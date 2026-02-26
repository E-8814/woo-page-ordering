#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TRAIN SHORT (2-5) → TEST LONG (21-25)
Transfer learning experiment: Train only on short documents, test on long ones
Based on universal_training_no_curriculum.py architecture
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import torch

# Enable TF32 for A100/A6000 GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Faster but non-deterministic; results may vary slightly between runs

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import argparse
import logging
from pathlib import Path
import random
from typing import Tuple, List, Dict
import math
from collections import defaultdict
import time
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class ShortDocDataset(Dataset):
    """Dataset for SHORT documents (2-5 pages) for training"""

    def __init__(self, data_dir: str, split: str = 'train',
                 train_ratio: float = 0.85, augment: bool = False, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment and (split == 'train')
        self.documents = []

        # Load files
        embedding_files = list(self.data_dir.glob("*_embeddings.json"))
        logger.info(f"Found {len(embedding_files)} total files")

        # Load documents in 2-5 page range
        valid_docs = []

        for file_path in embedding_files:
            try:
                with open(file_path, 'r') as f:
                    doc_data = json.load(f)

                if 'correct_order' not in doc_data:
                    continue

                doc_length = len(doc_data['embeddings'])

                # ONLY 2-5 pages
                if 2 <= doc_length <= 5:
                    doc_data['length'] = doc_length
                    doc_data['file_path'] = str(file_path)
                    valid_docs.append(doc_data)

            except Exception as e:
                pass

        # Split data
        random.seed(seed)
        random.shuffle(valid_docs)
        n_train = int(len(valid_docs) * train_ratio)

        if split == 'train':
            selected = valid_docs[:n_train]
        else:
            selected = valid_docs[n_train:]

        logger.info(f"Short doc (2-5 pages) {split} set: {len(selected)} documents")

        # Process documents
        for doc_data in selected:
            processed_doc = {
                'embeddings': torch.tensor(doc_data['embeddings'], dtype=torch.float32),
                'page_numbers': torch.tensor(doc_data['page_numbers'], dtype=torch.long),
                'correct_order': torch.tensor(doc_data['correct_order'], dtype=torch.long),
                'doc_id': Path(doc_data['file_path']).stem.replace('_embeddings', ''),
                'length': doc_data['length']
            }
            self.documents.append(processed_doc)

        if split == 'train':
            random.shuffle(self.documents)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        doc_data = self.documents[idx]
        n = doc_data['length']

        embeddings = doc_data['embeddings'].clone()

        # Augmentation
        if self.augment:
            if random.random() < 0.5:
                noise_scale = 0.015 * (1 + n / 30)
                noise = torch.randn_like(embeddings) * noise_scale
                embeddings = embeddings + noise

            if random.random() < 0.3:
                dropout_mask = torch.rand(embeddings.shape[-1]) > 0.1
                embeddings = embeddings * dropout_mask.float()

        # Shuffle for ordering task
        perm = torch.randperm(n)

        return {
            'embeddings': embeddings[perm],
            'page_numbers': doc_data['page_numbers'][perm],
            'correct_order': doc_data['correct_order'],
            'original_indices': perm,
            'doc_id': doc_data['doc_id'],
            'split': self.split,
            'length': n
        }


class LongDocDataset(Dataset):
    """Dataset for LONG documents (21-25 pages) for testing"""

    def __init__(self, data_dir: str, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.documents = []

        # Load files
        embedding_files = list(self.data_dir.glob("*_embeddings.json"))

        # Load documents in 21-25 page range
        for file_path in embedding_files:
            try:
                with open(file_path, 'r') as f:
                    doc_data = json.load(f)

                if 'correct_order' not in doc_data:
                    continue

                doc_length = len(doc_data['embeddings'])

                # ONLY 21-25 pages
                if 21 <= doc_length <= 25:
                    processed_doc = {
                        'embeddings': torch.tensor(doc_data['embeddings'], dtype=torch.float32),
                        'page_numbers': torch.tensor(doc_data['page_numbers'], dtype=torch.long),
                        'correct_order': torch.tensor(doc_data['correct_order'], dtype=torch.long),
                        'doc_id': Path(file_path).stem.replace('_embeddings', ''),
                        'length': doc_length
                    }
                    self.documents.append(processed_doc)

            except Exception as e:
                pass

        logger.info(f"Long doc (21-25 pages) test set: {len(self.documents)} documents")

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        doc_data = self.documents[idx]
        n = doc_data['length']

        # Shuffle for ordering task
        perm = torch.randperm(n)

        return {
            'embeddings': doc_data['embeddings'][perm],
            'page_numbers': doc_data['page_numbers'][perm],
            'correct_order': doc_data['correct_order'],
            'original_indices': perm,
            'doc_id': doc_data['doc_id'],
            'split': 'test',
            'length': n
        }


class FullAttentionBlock(nn.Module):
    """Full self-attention block"""

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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape

        x_norm = self.norm1(x)

        q = self.q_proj(x_norm).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x_norm).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x_norm).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x_attn = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.out_proj(x_attn)
        x_attn = self.dropout(x_attn)

        x = x + x_attn
        x = x + self.mlp(self.norm2(x))

        return x


class OrderingModel(nn.Module):
    """Pairwise ranking model for document ordering"""

    def __init__(self, embedding_dim: int, hidden_dim: int = 2048,
                 num_layers: int = 10, num_heads: int = 16,
                 dropout: float = 0.3, mlp_ratio: float = 4.0):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout * 0.5)
        )

        self.attention_blocks = nn.ModuleList([
            FullAttentionBlock(
                hidden_dim,
                num_heads,
                dropout,
                mlp_ratio
            )
            for _ in range(num_layers)
        ])

        self.pos_encoding = nn.Parameter(
            torch.randn(1, 100, hidden_dim) * 0.02
        )

        self.final_norm = nn.LayerNorm(hidden_dim)

        # Scoring network
        scorer_hidden = hidden_dim * 2

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

        self.global_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, embeddings: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, _ = embeddings.shape

        x = self.input_projection(embeddings)

        # Positional encoding
        if N <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :N, :]
        else:
            pos_enc = F.interpolate(
                self.pos_encoding.transpose(1, 2),
                size=N,
                mode='linear'
            ).transpose(1, 2)
            x = x + pos_enc

        # Attention blocks
        for block in self.attention_blocks:
            x = block(x, mask)

        x = self.final_norm(x)

        return x

    def compute_pairwise_scores(self, encoded: torch.Tensor) -> torch.Tensor:
        n = encoded.shape[1] if encoded.dim() == 3 else encoded.shape[0]
        device = encoded.device

        if encoded.dim() == 3:
            encoded = encoded.squeeze(0)

        encoded = F.normalize(encoded, p=2, dim=-1)

        # Compute pairwise differences
        encoded_i = encoded.unsqueeze(1)
        encoded_j = encoded.unsqueeze(0)
        diffs = encoded_j - encoded_i

        # Score differences
        diffs_flat = diffs.view(-1, encoded.shape[-1])
        scores_flat = self.diff_scorer(diffs_flat).squeeze(-1)
        scores = scores_flat.view(n, n)

        # Zero diagonal
        scores = scores * (1 - torch.eye(n, device=device))

        # Add global scores
        global_scores = self.global_scorer(encoded).squeeze(-1)
        scores = scores + global_scores.unsqueeze(0) - global_scores.unsqueeze(1)

        return scores

    def forward(self, embeddings: torch.Tensor,
                page_numbers: torch.Tensor = None,
                correct_order: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)

        encoded = self.encode(embeddings)
        pairwise_scores = self.compute_pairwise_scores(encoded)

        # Position scores
        outgoing_scores = pairwise_scores.mean(dim=1)
        incoming_scores = pairwise_scores.mean(dim=0)
        position_scores = incoming_scores - outgoing_scores

        return position_scores, pairwise_scores

    def predict_order(self, embeddings: torch.Tensor) -> torch.Tensor:
        position_scores, _ = self.forward(embeddings)
        predicted_order = torch.argsort(position_scores)
        return predicted_order


def pairwise_loss(pairwise_scores: torch.Tensor,
                  page_numbers: torch.Tensor,
                  correct_order: torch.Tensor,
                  doc_length: int,
                  margin: float = 1.0) -> torch.Tensor:
    """Loss for pairwise ranking"""
    device = pairwise_scores.device

    # Build mapping
    page_to_idx = {int(page): idx for idx, page in enumerate(page_numbers.cpu().numpy())}

    correct_indices = []
    for page in correct_order.cpu().numpy():
        if int(page) in page_to_idx:
            correct_indices.append(page_to_idx[int(page)])

    if len(correct_indices) < 2:
        return torch.tensor(0.0, device=device)

    # Adjust margin based on document length
    adjusted_margin = margin * (1.3 + doc_length / 25)

    ranking_loss = 0.0
    count = 0

    # Pairwise ranking loss
    for i in range(len(correct_indices) - 1):
        idx_i = correct_indices[i]
        idx_j = correct_indices[i + 1]

        correct_score = pairwise_scores[idx_i, idx_j]
        incorrect_score = pairwise_scores[idx_j, idx_i]

        ranking_loss += F.relu(adjusted_margin - correct_score)
        ranking_loss += F.relu(adjusted_margin + incorrect_score)
        count += 2

        # Additional pairs for longer documents
        if doc_length > 10:
            for k in range(i + 2, min(i + 5, len(correct_indices))):
                idx_k = correct_indices[k]
                ranking_loss += F.relu(adjusted_margin/2 - pairwise_scores[idx_i, idx_k])
                ranking_loss += F.relu(adjusted_margin/3 - pairwise_scores[idx_j, idx_k])
                count += 2

    if count > 0:
        ranking_loss = ranking_loss / count

    # Regularization
    reg_weight = 0.001 * (1 + doc_length / 50)
    total_loss = ranking_loss + reg_weight * pairwise_scores.abs().mean()

    return total_loss


def compute_metrics(predicted_order, correct_order, page_numbers):
    """Compute accuracy and Kendall's tau"""
    predicted_pages = [page_numbers[idx].item() for idx in predicted_order]

    correct = sum(1 for pred, true in zip(predicted_pages, correct_order.cpu().numpy())
                  if pred == true)
    accuracy = correct / len(predicted_pages) if predicted_pages else 0

    # Kendall's tau
    from scipy.stats import kendalltau

    true_positions = {page: i for i, page in enumerate(correct_order.cpu().numpy())}
    pred_positions = {page: i for i, page in enumerate(predicted_pages)}

    common_pages = set(true_positions.keys()) & set(pred_positions.keys())
    if len(common_pages) > 1:
        true_pos = [true_positions[p] for p in common_pages]
        pred_pos = [pred_positions[p] for p in common_pages]
        tau, _ = kendalltau(true_pos, pred_pos)
    else:
        tau = 0.0

    return {'accuracy': accuracy, 'kendall_tau': tau}


def evaluate_on_long_docs(model, long_test_dataset, device='cuda'):
    """Evaluate model on LONG documents (21-25 pages)"""
    logger.info("\n" + "="*80)
    logger.info("EVALUATING ON LONG DOCUMENTS (21-25 pages)")
    logger.info("="*80)

    model.eval()
    metrics = []

    with torch.no_grad():
        for idx, doc_data in enumerate(long_test_dataset):
            embeddings = doc_data['embeddings'].to(device)
            page_numbers = doc_data['page_numbers']
            correct_order = doc_data['correct_order']

            if len(embeddings) < 2:
                continue

            predicted_order = model.predict_order(embeddings)
            metric = compute_metrics(predicted_order, correct_order, page_numbers)
            metrics.append(metric)

            if (idx + 1) % 20 == 0:
                logger.info(f"  Processed {idx + 1}/{len(long_test_dataset)} documents...")

    avg_tau = np.mean([m['kendall_tau'] for m in metrics]) if metrics else 0.0
    avg_acc = np.mean([m['accuracy'] for m in metrics]) if metrics else 0.0

    logger.info(f"\n{'='*80}")
    logger.info(f"LONG DOCUMENT (21-25 pages) TEST RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"  Average Kendall's Tau: {avg_tau:.4f}")
    logger.info(f"  Average Accuracy: {avg_acc:.4f}")
    logger.info(f"  Number of documents: {len(metrics)}")
    logger.info(f"{'='*80}")

    return avg_tau, avg_acc


def train_model(data_dir: str, output_dir: str, device: str = 'cuda',
                epochs: int = 20, use_amp: bool = True):
    """Train model on SHORT docs (2-5 pages), test on LONG docs (21-25 pages)"""

    logger.info("="*80)
    logger.info("TRAIN SHORT (2-5 pages) → TEST LONG (21-25 pages)")
    logger.info("="*80)
    logger.info(f"Training epochs: {epochs}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info("="*80)

    # Load training data (SHORT docs only)
    train_dataset = ShortDocDataset(
        data_dir, split='train', augment=True
    )
    val_dataset = ShortDocDataset(
        data_dir, split='val', augment=False
    )

    if len(train_dataset) == 0:
        logger.error("No training data available")
        return None

    first_doc = train_dataset[0]
    embedding_dim = first_doc['embeddings'].shape[1]
    logger.info(f"Embedding dimension: {embedding_dim}")

    # Create model
    logger.info("Creating model...")
    model = OrderingModel(
        embedding_dim,
        hidden_dim=2048,
        num_layers=10,
        num_heads=16,
        dropout=0.3,
        mlp_ratio=4.0
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,} ({total_params*4/1024**2:.1f} MB)")

    if device == 'cuda':
        logger.info(f"Model on GPU: {next(model.parameters()).is_cuda}")
        logger.info(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=2, pin_memory=(device == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-5,
        weight_decay=1e-4, betas=(0.9, 0.999)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=5e-6
    )

    scaler = GradScaler('cuda') if use_amp and device == 'cuda' else None

    best_val_tau = -1.0
    grad_accum_steps = 4

    logger.info(f"\nStarting training...")
    logger.info("="*80)

    # Training loop
    for epoch in range(epochs):
        epoch_start = time.time()

        # Training
        model.train()
        train_losses = []
        train_metrics = []
        accumulation_counter = 0

        for batch_idx, doc_data in enumerate(train_loader):
            embeddings = doc_data['embeddings'].squeeze(0).to(device)
            page_numbers = doc_data['page_numbers'].squeeze(0)
            correct_order = doc_data['correct_order'].squeeze(0)
            doc_length = doc_data['length'].item()

            if len(embeddings) < 2:
                continue

            # Forward pass
            if scaler:
                with autocast('cuda'):
                    position_scores, pairwise_scores = model(embeddings, page_numbers, correct_order)
                    loss = pairwise_loss(
                        pairwise_scores, page_numbers, correct_order,
                        doc_length
                    )
            else:
                position_scores, pairwise_scores = model(embeddings, page_numbers, correct_order)
                loss = pairwise_loss(
                    pairwise_scores, page_numbers, correct_order,
                    doc_length
                )

            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps

            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_counter += 1

            # Update weights
            if accumulation_counter % grad_accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                optimizer.zero_grad()

            train_losses.append(loss.item() * grad_accum_steps)

            # Compute metrics
            with torch.no_grad():
                predicted_order = model.predict_order(embeddings)
                metrics = compute_metrics(predicted_order, correct_order, page_numbers)
                train_metrics.append(metrics)

        # Validation on SHORT docs
        model.eval()
        val_metrics = []

        with torch.no_grad():
            for doc_data in val_loader:
                embeddings = doc_data['embeddings'].squeeze(0).to(device)
                page_numbers = doc_data['page_numbers'].squeeze(0)
                correct_order = doc_data['correct_order'].squeeze(0)

                if len(embeddings) < 2:
                    continue

                predicted_order = model.predict_order(embeddings)
                metric = compute_metrics(predicted_order, correct_order, page_numbers)
                val_metrics.append(metric)

        train_loss = np.mean(train_losses) if train_losses else 0
        train_tau = np.mean([m['kendall_tau'] for m in train_metrics]) if train_metrics else 0
        val_tau = np.mean([m['kendall_tau'] for m in val_metrics]) if val_metrics else 0

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Tau: {train_tau:.4f} | "
            f"Val Tau (short): {val_tau:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model based on SHORT doc validation
        if val_tau > best_val_tau:
            best_val_tau = val_tau

            model_path = Path(output_dir) / "train_short_test_long_best.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_tau_short': val_tau,
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)

            logger.info(f"  ✅ NEW BEST | Val Tau (short): {val_tau:.4f}")

        scheduler.step()

        if device == 'cuda' and epoch % 5 == 0:
            logger.info(f"  GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

    logger.info("\n" + "="*80)
    logger.info(f"Training complete! Best val tau on SHORT docs: {best_val_tau:.4f}")
    logger.info("="*80)

    # Load best model
    checkpoint = torch.load(Path(output_dir) / "train_short_test_long_best.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Now test on LONG documents
    logger.info("\nLoading LONG documents (21-25 pages) for testing...")
    long_test_dataset = LongDocDataset(data_dir)

    if len(long_test_dataset) > 0:
        long_tau, long_acc = evaluate_on_long_docs(model, long_test_dataset, device)

        # Final summary
        logger.info("\n" + "="*80)
        logger.info("FINAL SUMMARY")
        logger.info("="*80)
        logger.info(f"Trained on: 2-5 page documents")
        logger.info(f"Best validation tau (short docs): {best_val_tau:.4f}")
        logger.info(f"")
        logger.info(f"Tested on: 21-25 page documents")
        logger.info(f"Test tau (long docs): {long_tau:.4f}")
        logger.info(f"Test accuracy (long docs): {long_acc:.4f}")
        logger.info("="*80)

        if long_tau < 0.05:
            logger.info("INTERPRETATION: ❌ CATASTROPHIC FAILURE - Random performance")
            logger.info("  → Strong evidence of ZERO TRANSFER from short to long docs")
        elif long_tau < 0.15:
            logger.info("INTERPRETATION: ⚠️  POOR PERFORMANCE - Minimal transfer")
        elif long_tau < 0.30:
            logger.info("INTERPRETATION: 🟡 WEAK TRANSFER")
        else:
            logger.info("INTERPRETATION: ✅ GOOD TRANSFER")

    else:
        logger.warning("No long documents found for testing")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train Short Test Long")
    parser.add_argument("--data_dir", type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
                       help="Directory containing embedding files")
    parser.add_argument("--output_dir", type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "transfer_output"),
                       help="Directory to save models and logs")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("TRANSFER LEARNING EXPERIMENT")
    logger.info("Train on SHORT (2-5 pages) → Test on LONG (21-25 pages)")
    logger.info("="*80)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Training epochs: {args.epochs}")

    if torch.cuda.is_available() and args.device == 'cuda':
        num_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"    Memory: {torch.cuda.get_device_properties(i).total_memory/1024**3:.1f} GB")

    logger.info("="*80)

    # Train model
    model = train_model(
        args.data_dir, args.output_dir,
        device=args.device, epochs=args.epochs, use_amp=True
    )

    logger.info("\n✅ Experiment complete!")


if __name__ == "__main__":
    main()
