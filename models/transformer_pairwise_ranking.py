#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UNIVERSAL MODEL TRAINING - NO CURRICULUM VERSION
Trains a single model optimized for all document lengths.
Direct training on all data simultaneously (no curriculum).
Comprehensive logging across all ranges.
Memory efficient with GPU monitoring.

Key Features:
- Single universal model trained from scratch
- Trains directly on all document lengths (2-25)
- Detailed metrics logging for analysis
- GPU memory management
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import torch

# Enable TF32 for A100/A6000 GPUs (after importing torch)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
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
from typing import Tuple, List, Dict, Optional
import math
from collections import defaultdict
import time
from dataclasses import dataclass, field
import csv
from datetime import datetime
import psutil
import gc

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_file_logging(output_dir, model_name):
    """Setup file logging with timestamp"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"universal_{model_name}_{timestamp}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)

    return log_file


@dataclass
class CurriculumStage:
    """Configuration for a curriculum stage"""
    name: str
    train_range: Tuple[int, int]
    epochs: int
    learning_rate: float
    description: str
    target_weight: float = 1.0  # Weight for target documents
    non_target_weight: float = 1.0  # Weight for non-target documents


@dataclass
class ModelConfig:
    """Configuration for the universal model"""
    name: str
    target_range: Tuple[int, int]
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    mlp_ratio: float
    description: str
    curriculum_stages: List[CurriculumStage] = field(default_factory=list)


# Evaluation ranges for comprehensive logging
EVAL_RANGES = [
    ("2-5", (2, 5)),
    ("6-10", (6, 10)),
    ("11-15", (11, 15)),
    ("16-20", (16, 20)),
    ("21-25", (21, 25)),
    ("26-30", (26, 30)),
    ("31-40", (31, 40)),
    ("41+", (41, 999))
]


def get_universal_model_config():
    """Get configuration for universal model"""
    config = ModelConfig(
        name="universal",
        target_range=(2, 25),
        hidden_dim=2048,
        num_layers=10,
        num_heads=16,
        dropout=0.3,
        mlp_ratio=4.0,
        description="Universal model for all document lengths",
        curriculum_stages=[
            CurriculumStage(
                name="direct",
                train_range=(2, 25),
                epochs=60,
                learning_rate=5e-5,
                description="Direct training on all data",
                target_weight=5.0,
                non_target_weight=1.0
            )
        ]
    )

    return config


class ComprehensiveMetricsLogger:
    """Logs metrics for ALL document ranges"""

    def __init__(self, output_dir: str, model_name: str):
        self.output_dir = Path(output_dir) / "metrics"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"universal_{model_name}_{timestamp}.csv"

        # Create header
        header = ['epoch', 'stage', 'timestamp', 'train_loss', 'train_tau']

        # Add columns for each evaluation range
        for range_name, _ in EVAL_RANGES:
            header.extend([
                f'val_{range_name}_tau',
                f'val_{range_name}_acc',
                f'val_{range_name}_count'
            ])

        header.extend(['learning_rate', 'gpu_memory_mb', 'epoch_time'])

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def log_epoch(self, epoch: int, stage: str, train_metrics: dict,
                  val_metrics_by_range: dict,
                  learning_rate: float, epoch_time: float):
        """Log metrics for all ranges"""

        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2

        row = [
            epoch, stage, datetime.now().isoformat(),
            train_metrics.get('loss', 0),
            train_metrics.get('tau', 0)
        ]

        # Add metrics for each range
        for range_name, _ in EVAL_RANGES:
            if range_name in val_metrics_by_range:
                m = val_metrics_by_range[range_name]
                row.extend([m.get('tau', -1), m.get('acc', -1), m.get('count', 0)])
            else:
                row.extend([-1, -1, 0])

        row.extend([learning_rate, gpu_memory, epoch_time])

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


class UniversalDataset(Dataset):
    """Dataset for universal model training"""

    def __init__(self, data_dir: str, stage: CurriculumStage,
                 split: str = 'train',
                 train_ratio: float = 0.85, augment: bool = False,
                 verbose: int = 1, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.stage = stage
        self.split = split
        self.augment = augment and (split == 'train')
        self.documents = []

        # Load files
        embedding_files = list(self.data_dir.glob("*_embeddings.json"))

        if verbose >= 1:
            logger.info(f"Found {len(embedding_files)} total files")

        # Load documents in training range
        valid_docs = []

        for file_path in embedding_files:
            try:
                with open(file_path, 'r') as f:
                    doc_data = json.load(f)

                if 'correct_order' not in doc_data:
                    continue

                doc_length = len(doc_data['embeddings'])

                # Check if in training range
                if stage.train_range[0] <= doc_length <= stage.train_range[1]:
                    doc_data['length'] = doc_length
                    doc_data['file_path'] = str(file_path)
                    valid_docs.append(doc_data)

            except Exception as e:
                if verbose >= 3:
                    logger.warning(f"Error loading {file_path}: {e}")

        # Split data
        random.seed(seed)
        random.shuffle(valid_docs)
        n_train = int(len(valid_docs) * train_ratio)

        if split == 'train':
            selected = valid_docs[:n_train]
        else:
            selected = valid_docs[n_train:]

        if verbose >= 1 and split == 'train':
            logger.info(f"  Stage {stage.name} training set:")
            logger.info(f"    Total docs: {len(selected)}")

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

        if verbose >= 1:
            logger.info(f"  Final dataset size: {len(self.documents)}")

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


class UniversalOrderingModel(nn.Module):
    """Universal model for all document lengths"""

    def __init__(self, embedding_dim: int, config: ModelConfig):
        super().__init__()

        self.config = config
        self.embedding_dim = embedding_dim

        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout * 0.5)
        )

        self.attention_blocks = nn.ModuleList([
            FullAttentionBlock(
                config.hidden_dim,
                config.num_heads,
                config.dropout,
                config.mlp_ratio
            )
            for _ in range(config.num_layers)
        ])

        self.pos_encoding = nn.Parameter(
            torch.randn(1, 100, config.hidden_dim) * 0.02
        )

        self.final_norm = nn.LayerNorm(config.hidden_dim)

        # Scoring network
        scorer_hidden = config.hidden_dim * 2

        self.diff_scorer = nn.Sequential(
            nn.Linear(config.hidden_dim, scorer_hidden),
            nn.GELU(),
            nn.LayerNorm(scorer_hidden),
            nn.Dropout(config.dropout),

            nn.Linear(scorer_hidden, scorer_hidden // 2),
            nn.GELU(),
            nn.LayerNorm(scorer_hidden // 2),
            nn.Dropout(config.dropout * 0.8),

            nn.Linear(scorer_hidden // 2, scorer_hidden // 4),
            nn.GELU(),
            nn.LayerNorm(scorer_hidden // 4),
            nn.Dropout(config.dropout * 0.5),

            nn.Linear(scorer_hidden // 4, 1)
        )

        self.global_scorer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
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


def universal_loss(pairwise_scores: torch.Tensor,
                   page_numbers: torch.Tensor,
                   correct_order: torch.Tensor,
                   doc_length: int,
                   margin: float = 1.0) -> torch.Tensor:
    """Loss for universal model"""
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


def evaluate_all_ranges(model, data_dir, device='cuda', verbose=False):
    """Evaluate model on ALL document length ranges"""
    results = {}

    for range_name, (min_len, max_len) in EVAL_RANGES:
        try:
            # Create a fake stage for dataset initialization
            fake_stage = CurriculumStage(
                name="eval",
                train_range=(min_len, max_len),
                epochs=1,
                learning_rate=0,
                description="Evaluation",
                target_weight=1.0,
                non_target_weight=1.0
            )
            val_dataset = UniversalDataset(
                data_dir=data_dir,
                stage=fake_stage,
                split='val',
                augment=False,
                verbose=0
            )

            if len(val_dataset) == 0:
                results[range_name] = {'tau': -1, 'acc': -1, 'count': 0}
                continue

            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

            model.eval()
            metrics = []

            with torch.no_grad():
                for doc_data in val_loader:
                    embeddings = doc_data['embeddings'].squeeze(0).to(device)
                    page_numbers = doc_data['page_numbers'].squeeze(0)
                    correct_order = doc_data['correct_order'].squeeze(0)

                    if len(embeddings) < 2:
                        continue

                    predicted_order = model.predict_order(embeddings)
                    metric = compute_metrics(predicted_order, correct_order, page_numbers)
                    metrics.append(metric)

            if metrics:
                results[range_name] = {
                    'tau': np.mean([m['kendall_tau'] for m in metrics]),
                    'acc': np.mean([m['accuracy'] for m in metrics]),
                    'count': len(metrics)
                }
            else:
                results[range_name] = {'tau': -1, 'acc': -1, 'count': 0}

        except Exception as e:
            if verbose:
                logger.warning(f"Error evaluating range {range_name}: {e}")
            results[range_name] = {'tau': -1, 'acc': -1, 'count': 0}

    return results


def train_universal_model(config: ModelConfig, data_dir: str, output_dir: str,
                          device: str = 'cuda', use_amp: bool = True):
    """Train universal model"""

    logger.info("="*80)
    logger.info(f"TRAINING UNIVERSAL MODEL")
    logger.info(f"Description: {config.description}")
    logger.info(f"Architecture: {config.hidden_dim}D, {config.num_layers}L, {config.num_heads}H")
    logger.info(f"Total Stages: {len(config.curriculum_stages)}")
    logger.info("="*80)

    # Setup logging
    log_file = setup_file_logging(output_dir, config.name)
    logger.info(f"Logging to: {log_file}")

    # Create metrics logger
    metrics_logger = ComprehensiveMetricsLogger(output_dir, config.name)

    # Get embedding dimension
    fake_stage = CurriculumStage(
        name="temp",
        train_range=(2, 25),
        epochs=1,
        learning_rate=0,
        description="Temp",
        target_weight=1.0,
        non_target_weight=1.0
    )
    temp_dataset = UniversalDataset(
        data_dir=data_dir,
        stage=fake_stage,
        split='train',
        augment=False,
        verbose=0
    )
    if len(temp_dataset) == 0:
        logger.error("No training data available")
        return None, 0

    first_doc = temp_dataset[0]
    embedding_dim = first_doc['embeddings'].shape[1]
    logger.info(f"Embedding dimension: {embedding_dim}")

    # Create model FROM SCRATCH
    logger.info("Creating new model from scratch...")
    model = UniversalOrderingModel(embedding_dim, config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,} ({total_params*4/1024**2:.1f} MB)")

    if device == 'cuda':
        logger.info(f"Model on GPU: {next(model.parameters()).is_cuda}")
        logger.info(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

    best_avg_tau = -1.0
    global_epoch = 0

    # Gradient accumulation for memory efficiency
    grad_accum_steps = 4

    # Train through curriculum stages
    for stage_idx, stage in enumerate(config.curriculum_stages):
        logger.info("-" * 80)
        logger.info(f"STAGE {stage_idx + 1}/{len(config.curriculum_stages)}: {stage.name}")
        logger.info(f"Description: {stage.description}")
        logger.info(f"Training range: {stage.train_range}")
        logger.info(f"Learning rate: {stage.learning_rate}")
        logger.info(f"Epochs: {stage.epochs}")
        logger.info("-" * 80)

        # Create dataset for this stage
        train_dataset = UniversalDataset(
            data_dir, stage,
            split='train', augment=True, verbose=1
        )

        if len(train_dataset) == 0:
            logger.warning(f"No training data for stage {stage.name}")
            continue

        train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True,
            num_workers=2, pin_memory=(device == 'cuda')
        )

        # Optimizer for this stage
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=stage.learning_rate,
            weight_decay=1e-4, betas=(0.9, 0.999)
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage.epochs, eta_min=stage.learning_rate * 0.1
        )

        scaler = GradScaler('cuda') if use_amp and device == 'cuda' else None

        stage_patience = 0
        stage_max_patience = max(20, stage.epochs // 5)

        # Training loop for this stage
        for epoch in range(stage.epochs):
            global_epoch += 1
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
                        loss = universal_loss(
                            pairwise_scores, page_numbers, correct_order,
                            doc_length
                        )
                else:
                    position_scores, pairwise_scores = model(embeddings, page_numbers, correct_order)
                    loss = universal_loss(
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

                # Clear cache periodically
                if batch_idx % 50 == 0 and device == 'cuda':
                    torch.cuda.empty_cache()

            # Evaluate on ALL ranges
            val_metrics_by_range = evaluate_all_ranges(model, data_dir, device)

            # Calculate average tau across all ranges
            avg_tau = np.mean([m['tau'] for m in val_metrics_by_range.values() if m['count'] > 0])

            # Calculate metrics
            train_summary = {
                'loss': np.mean(train_losses) if train_losses else 0,
                'tau': np.mean([m['kendall_tau'] for m in train_metrics]) if train_metrics else 0
            }

            current_lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start

            # Log to CSV
            metrics_logger.log_epoch(
                global_epoch, stage.name, train_summary,
                val_metrics_by_range,
                current_lr, epoch_time
            )

            # Console logging
            logger.info(f"Epoch {epoch+1:3d}/{stage.epochs} | Stage: {stage.name} | Time: {epoch_time:6.2f}s")
            logger.info(f"  Train | Loss: {train_summary['loss']:7.4f} | Tau: {train_summary['tau']:6.3f}")

            # Log validation performance for each range
            logger.info("  Validation Performance by Range:")
            for range_name, (min_l, max_l) in EVAL_RANGES:
                if range_name in val_metrics_by_range:
                    m = val_metrics_by_range[range_name]
                    if m['count'] > 0:
                        logger.info(f"    {range_name:5s} ({min_l:2d}-{max_l:3d}): "
                                  f"Tau={m['tau']:6.3f} Acc={m['acc']:5.3f} N={m['count']:3d}")

            logger.info(f"  Average Tau across all ranges: {avg_tau:.4f}")

            # Save best model based on average performance
            if avg_tau > best_avg_tau:
                best_avg_tau = avg_tau
                stage_patience = 0

                model_path = Path(output_dir) / f"model_{config.name}.pt"
                model_path.parent.mkdir(parents=True, exist_ok=True)

                # Save the model state
                state_dict = model.state_dict()

                torch.save({
                    'epoch': global_epoch,
                    'stage': stage.name,
                    'model_state_dict': state_dict,
                    'config': config,
                    'avg_tau': avg_tau,
                    'all_metrics': val_metrics_by_range
                }, model_path)

                logger.info(f"  ✅ NEW BEST MODEL | Average Tau: {avg_tau:.4f}")
            else:
                stage_patience += 1

            # GPU memory logging
            if device == 'cuda' and epoch % 10 == 0:
                logger.info(f"  GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

            # Early stopping
            if stage_patience >= stage_max_patience:
                logger.info(f"Early stopping stage {stage.name} at epoch {epoch+1}")
                break

            if avg_tau > 0.95:
                logger.info(f"Excellent performance achieved! Moving to next stage.")
                break

            scheduler.step()

        logger.info(f"Stage {stage.name} complete")

        # Clean up memory between stages
        if device == 'cuda':
            torch.cuda.empty_cache()

    logger.info("="*80)
    logger.info(f"Training complete for {config.name}")
    logger.info(f"Best average Tau: {best_avg_tau:.4f}")
    logger.info(f"Model saved to: {output_dir}/model_{config.name}.pt")
    logger.info(f"Metrics saved to: {metrics_logger.csv_path}")

    return model, best_avg_tau


def main():
    parser = argparse.ArgumentParser(description="Universal Model Training")
    parser.add_argument("--data_dir", type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
                       help="Directory containing embedding files")
    parser.add_argument("--output_dir", type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "universal_output"),
                       help="Directory to save models and logs")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("UNIVERSAL MODEL TRAINING")
    logger.info("Training single model for all document ranges")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {args.device}")

    if torch.cuda.is_available() and args.device == 'cuda':
        num_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"    Memory: {torch.cuda.get_device_properties(i).total_memory/1024**3:.1f} GB")
        logger.info(f"PyTorch CUDA: {torch.version.cuda}")
        logger.info(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
        logger.info(f"Cudnn benchmark: {torch.backends.cudnn.benchmark}")

    logger.info("="*80)

    # Get model config
    config = get_universal_model_config()

    logger.info(f"Training universal model:")
    logger.info(f"  - {config.name}: {config.description}")
    logger.info("")

    # Train model
    model, best_tau = train_universal_model(
        config, args.data_dir, args.output_dir,
        device=args.device, use_amp=True
    )

    # Final summary
    logger.info("="*80)
    logger.info("TRAINING COMPLETE - FINAL RESULTS")
    logger.info("="*80)
    logger.info(f"Universal model: Average Tau = {best_tau:.4f}")
    logger.info("="*80)
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info(f"Metrics saved to: {args.output_dir}/metrics/")


if __name__ == "__main__":
    main()
