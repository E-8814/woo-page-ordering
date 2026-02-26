#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TEST CLAIM 2: Local vs Global Attention Patterns

Tests whether "local pairwise relationships dominate 2-5 page documents,
while global document structure matters more for 20+ pages"

Strategy:
1. Load specialized models: tiny (2-5 pages) and xlarge (21-25 pages)
2. Extract attention weights from all layers
3. Analyze attention distance (do models focus locally or globally?)
4. Visualize attention patterns
5. Quantify: avg attention distance, locality score, etc.

Expected time: 1-2 hours (inference only, no training)
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
import random
from scipy.stats import kendalltau
from datetime import datetime
import math
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import List, Tuple

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Define classes needed for model loading (even though we don't use them)
@dataclass
class CurriculumStage:
    """Configuration for a curriculum stage (needed for unpickling)"""
    name: str
    train_range: Tuple[int, int]
    epochs: int
    learning_rate: float
    description: str = ""
    target_weight: float = 1.0
    non_target_weight: float = 1.0


@dataclass
class ModelConfig:
    """Configuration for each specialized model (needed for unpickling)"""
    name: str
    target_range: Tuple[int, int]
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    mlp_ratio: float = 4.0
    description: str = ""
    curriculum_stages: List[CurriculumStage] = field(default_factory=list)


class FullAttentionBlock(nn.Module):
    """Full self-attention block with attention weight extraction"""

    def __init__(self, hidden_dim: int, num_heads: int = 16,
                 dropout: float = 0.3, mlp_ratio: float = 4.0):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        # Store attention weights for analysis
        self.last_attention_weights = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, return_attention: bool = False) -> torch.Tensor:
        B, N, C = x.shape

        x_norm = self.norm1(x)

        q = self.q_proj(x_norm).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x_norm).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x_norm).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)

        # Store attention weights
        self.last_attention_weights = attn.detach()

        attn = self.dropout(attn)

        x_attn = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.out_proj(x_attn)
        x_attn = self.dropout(x_attn)

        x = x + x_attn
        x = x + self.mlp(self.norm2(x))

        if return_attention:
            return x, self.last_attention_weights

        return x


class SpecializedOrderingModel(nn.Module):
    """Model specialized for specific document length range"""

    def __init__(self, embedding_dim: int, hidden_dim: int, num_layers: int,
                 num_heads: int, dropout: float, mlp_ratio: float = 4.0,
                 target_max_pages: int = 25):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout * 0.5)
        )

        self.attention_blocks = nn.ModuleList([
            FullAttentionBlock(hidden_dim, num_heads, dropout, mlp_ratio)
            for _ in range(num_layers)
        ])

        self.pos_encoding = nn.Parameter(
            torch.randn(1, 100, hidden_dim) * 0.02
        )

        self.final_norm = nn.LayerNorm(hidden_dim)

        # Scoring network (not used in attention analysis)
        scorer_hidden = hidden_dim * 2 if target_max_pages > 15 else hidden_dim
        self.diff_scorer = nn.Sequential(
            nn.Linear(hidden_dim, scorer_hidden),
            nn.GELU(),
            nn.LayerNorm(scorer_hidden),
            nn.Dropout(dropout),
            nn.Linear(scorer_hidden, scorer_hidden // 2),
            nn.GELU(),
            nn.LayerNorm(scorer_hidden // 2),
            nn.Dropout(dropout * 0.8),
            nn.Linear(scorer_hidden // 2, scorer_hidden // 4),
            nn.GELU(),
            nn.LayerNorm(scorer_hidden // 4),
            nn.Dropout(dropout * 0.5),
            nn.Linear(scorer_hidden // 4, 1)
        )

        # Only for long documents
        if target_max_pages > 15:
            self.global_scorer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.LayerNorm(hidden_dim // 2),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
        else:
            self.global_scorer = None

    def encode(self, embeddings: torch.Tensor, return_attention: bool = False):
        """Encode embeddings and optionally return attention weights"""
        B, N, _ = embeddings.shape

        x = self.input_projection(embeddings)

        # Positional encoding
        if N <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :N, :]

        attention_weights_per_layer = []

        # Attention blocks
        for block in self.attention_blocks:
            if return_attention:
                x = block(x, return_attention=False)
                # Get attention weights from last computation
                if block.last_attention_weights is not None:
                    attention_weights_per_layer.append(block.last_attention_weights)
            else:
                x = block(x)

        x = self.final_norm(x)

        if return_attention:
            return x, attention_weights_per_layer
        return x

    def forward(self, embeddings: torch.Tensor, return_attention: bool = False):
        if return_attention:
            encoded, attention_weights = self.encode(embeddings, return_attention=True)
            return encoded, attention_weights
        else:
            encoded = self.encode(embeddings)
            return encoded


def load_model(model_path: str, embedding_dim: int, device: str):
    """Load a trained model"""
    logger.info(f"Loading model from: {model_path}")

    # Load checkpoint to get actual config
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract config from checkpoint
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        config = {
            'hidden_dim': saved_config.hidden_dim,
            'num_layers': saved_config.num_layers,
            'num_heads': saved_config.num_heads,
            'dropout': saved_config.dropout,
            'mlp_ratio': saved_config.mlp_ratio,
            'target_range': saved_config.target_range
        }
    else:
        raise ValueError(f"No config found in checkpoint: {model_path}")

    model = SpecializedOrderingModel(
        embedding_dim=embedding_dim,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        mlp_ratio=config['mlp_ratio'],
        target_max_pages=config['target_range'][1]
    ).to(device)

    # Extract state_dict from checkpoint (already loaded above)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    logger.info(f"  Hidden dim: {config['hidden_dim']}")
    logger.info(f"  Num layers: {config['num_layers']}")
    logger.info(f"  Num heads: {config['num_heads']}")

    return model, config


def extract_attention_patterns(model, embeddings: torch.Tensor, device: str):
    """Extract attention patterns from a document"""
    model.eval()

    with torch.no_grad():
        embeddings = embeddings.unsqueeze(0).to(device)
        _, attention_weights_per_layer = model(embeddings, return_attention=True)

    # attention_weights_per_layer: list of (B, num_heads, N, N)
    # Average across batch and heads
    averaged_attention_per_layer = []

    for attn in attention_weights_per_layer:
        # (B, num_heads, N, N) -> (N, N)
        avg_attn = attn.mean(dim=(0, 1)).cpu().numpy()
        averaged_attention_per_layer.append(avg_attn)

    return averaged_attention_per_layer


def compute_attention_distance(attention_matrix: np.ndarray) -> float:
    """
    Compute average attention distance.

    For each position i, compute weighted average distance to positions it attends to.
    Lower distance = more local attention
    Higher distance = more global attention
    """
    N = len(attention_matrix)
    total_distance = 0.0

    for i in range(N):
        weighted_distance = 0.0
        for j in range(N):
            distance = abs(j - i)
            weighted_distance += distance * attention_matrix[i, j]

        total_distance += weighted_distance

    avg_distance = total_distance / N
    return avg_distance


def compute_locality_score(attention_matrix: np.ndarray, window: int = 2) -> float:
    """
    Compute locality score: fraction of attention within local window.

    Higher score = more local attention
    Lower score = more global attention
    """
    N = len(attention_matrix)
    local_attention = 0.0
    total_attention = 0.0

    for i in range(N):
        for j in range(N):
            total_attention += attention_matrix[i, j]
            if abs(j - i) <= window:
                local_attention += attention_matrix[i, j]

    locality = local_attention / total_attention if total_attention > 0 else 0.0
    return locality


def analyze_model_attention(model, dataset, model_name: str, device: str, num_samples: int = 50):
    """Analyze attention patterns for a model on its target documents"""

    logger.info(f"\nAnalyzing attention patterns for {model_name}...")

    all_avg_distances = []
    all_locality_scores = []
    all_attention_matrices = []

    for idx in range(min(num_samples, len(dataset))):
        doc = dataset[idx]
        embeddings = doc['embeddings']

        # Extract attention
        attention_per_layer = extract_attention_patterns(model, embeddings, device)

        # Analyze each layer
        for layer_idx, attn_matrix in enumerate(attention_per_layer):
            avg_dist = compute_attention_distance(attn_matrix)
            locality = compute_locality_score(attn_matrix, window=2)

            all_avg_distances.append(avg_dist)
            all_locality_scores.append(locality)

            if idx < 3:  # Save first few for visualization
                all_attention_matrices.append((layer_idx, attn_matrix))

    results = {
        'model_name': model_name,
        'avg_attention_distance': np.mean(all_avg_distances),
        'std_attention_distance': np.std(all_avg_distances),
        'avg_locality_score': np.mean(all_locality_scores),
        'std_locality_score': np.std(all_locality_scores),
        'sample_matrices': all_attention_matrices[:10]  # First 10 for viz
    }

    logger.info(f"  Avg attention distance: {results['avg_attention_distance']:.2f} ± {results['std_attention_distance']:.2f}")
    logger.info(f"  Avg locality score (±2): {results['avg_locality_score']:.3f} ± {results['std_locality_score']:.3f}")

    return results


def visualize_attention_comparison(short_results, long_results, output_dir: Path):
    """Create visualization comparing short vs long model attention patterns"""

    logger.info("\nCreating visualizations...")

    output_dir.mkdir(exist_ok=True)

    # 1. Bar chart: Attention Distance Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    models = ['Short\n(2-5 pages)', 'Long\n(21-25 pages)']
    distances = [short_results['avg_attention_distance'], long_results['avg_attention_distance']]
    distance_stds = [short_results['std_attention_distance'], long_results['std_attention_distance']]

    ax1.bar(models, distances, yerr=distance_stds, capsize=10,
            color=['#3498db', '#e74c3c'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Average Attention Distance', fontweight='bold')
    ax1.set_title('Attention Distance (Higher = More Global)', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Locality Score Comparison
    localities = [short_results['avg_locality_score'], long_results['avg_locality_score']]
    locality_stds = [short_results['std_locality_score'], long_results['std_locality_score']]

    ax2.bar(models, localities, yerr=locality_stds, capsize=10,
            color=['#3498db', '#e74c3c'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Locality Score (within ±2 positions)', fontweight='bold')
    ax2.set_title('Local Attention Fraction (Higher = More Local)', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'attention_distance_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'attention_distance_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"  Saved: {output_dir / 'attention_distance_comparison.pdf'}")
    plt.close()

    # 3. Attention Heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Short model samples
    for idx, (layer_idx, attn_matrix) in enumerate(short_results['sample_matrices'][:2]):
        ax = axes[0, idx]
        sns.heatmap(attn_matrix, cmap='Blues', ax=ax, cbar_kws={'label': 'Attention Weight'},
                    square=True, vmin=0, vmax=attn_matrix.max())
        ax.set_title(f'Short Model (2-5 pages) - Layer {layer_idx}', fontweight='bold')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')

    # Long model samples
    for idx, (layer_idx, attn_matrix) in enumerate(long_results['sample_matrices'][:2]):
        ax = axes[1, idx]
        sns.heatmap(attn_matrix, cmap='Reds', ax=ax, cbar_kws={'label': 'Attention Weight'},
                    square=True, vmin=0, vmax=attn_matrix.max())
        ax.set_title(f'Long Model (21-25 pages) - Layer {layer_idx}', fontweight='bold')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')

    plt.tight_layout()
    plt.savefig(output_dir / 'attention_heatmaps.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'attention_heatmaps.png', dpi=300, bbox_inches='tight')
    logger.info(f"  Saved: {output_dir / 'attention_heatmaps.pdf'}")
    plt.close()


def load_test_documents(data_dir: str, min_pages: int, max_pages: int, max_docs: int = 50):
    """Load test documents for a specific range"""
    data_dir = Path(data_dir)
    documents = []

    for file_path in data_dir.glob("*_embeddings.json"):
        if len(documents) >= max_docs:
            break

        try:
            with open(file_path, 'r') as f:
                doc_data = json.load(f)

            doc_length = len(doc_data['embeddings'])

            if min_pages <= doc_length <= max_pages:
                documents.append({
                    'embeddings': torch.tensor(doc_data['embeddings'], dtype=torch.float32),
                    'length': doc_length
                })

        except Exception as e:
            continue

    logger.info(f"Loaded {len(documents)} documents ({min_pages}-{max_pages} pages)")
    return documents


def main():
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    # Model checkpoints produced by models/specialized_pairwise_ranking_direct.py
    # Train that model first: python models/specialized_pairwise_ranking_direct.py --data_dir data/
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ultimate_output_no_curriculum")
    OUTPUT_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info("="*80)
    logger.info("ATTENTION PATTERN ANALYSIS: LOCAL vs GLOBAL")
    logger.info("="*80)
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Model directory: {MODEL_DIR}")
    logger.info(f"Device: {device}")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info("HYPOTHESIS TO TEST:")
    logger.info("  'Local pairwise relationships dominate 2-5 page documents,")
    logger.info("   while global document structure matters more for 20+ pages'")
    logger.info("")
    logger.info("STRATEGY:")
    logger.info("  1. Load tiny (2-5 pages) and xlarge (21-25 pages) models")
    logger.info("  2. Extract attention weights from all layers")
    logger.info("  3. Compute attention distance (local vs global)")
    logger.info("  4. Compare patterns between short and long models")
    logger.info("="*80)

    # Get embedding dimension
    sample_file = next(Path(DATA_DIR).glob("*_embeddings.json"))
    with open(sample_file, 'r') as f:
        sample_data = json.load(f)
    embedding_dim = len(sample_data['embeddings'][0])
    logger.info(f"\nEmbedding dimension: {embedding_dim}")

    # Load models
    short_model, _ = load_model(f"{MODEL_DIR}/model_tiny_2_5.pt", embedding_dim, device)
    long_model, _ = load_model(f"{MODEL_DIR}/model_xlarge_21_25.pt", embedding_dim, device)

    # Load test documents
    logger.info("\nLoading test documents...")
    short_docs = load_test_documents(DATA_DIR, min_pages=2, max_pages=5, max_docs=50)
    long_docs = load_test_documents(DATA_DIR, min_pages=21, max_pages=25, max_docs=50)

    # Analyze attention patterns
    short_results = analyze_model_attention(short_model, short_docs, "Short (2-5 pages)", device, num_samples=50)
    long_results = analyze_model_attention(long_model, long_docs, "Long (21-25 pages)", device, num_samples=50)

    # Visualize
    visualize_attention_comparison(short_results, long_results, OUTPUT_DIR)

    # Final interpretation
    logger.info("\n" + "="*80)
    logger.info("RESULTS & INTERPRETATION")
    logger.info("="*80)

    logger.info(f"\nShort Model (2-5 pages):")
    logger.info(f"  Attention distance: {short_results['avg_attention_distance']:.2f}")
    logger.info(f"  Locality score: {short_results['avg_locality_score']:.3f}")

    logger.info(f"\nLong Model (21-25 pages):")
    logger.info(f"  Attention distance: {long_results['avg_attention_distance']:.2f}")
    logger.info(f"  Locality score: {long_results['avg_locality_score']:.3f}")

    # Compare
    distance_ratio = long_results['avg_attention_distance'] / short_results['avg_attention_distance']
    locality_ratio = short_results['avg_locality_score'] / long_results['avg_locality_score']

    logger.info(f"\nComparison:")
    logger.info(f"  Long model attention distance is {distance_ratio:.2f}x that of short model")
    logger.info(f"  Short model locality is {locality_ratio:.2f}x that of long model")

    logger.info("="*80)


if __name__ == "__main__":
    main()
