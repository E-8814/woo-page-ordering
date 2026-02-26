#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM-based Baseline for Document Page Ordering

Approach: Train a bidirectional LSTM to predict absolute page positions
- Input: Page embedding (3072-dim vector)
- Output: Position score (0-1 normalized)
- Learns: "What does page 1 look like vs page N?"

This baseline tests whether documents have consistent semantic structure
that allows position prediction from embedding features alone.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from scipy.stats import kendalltau
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import sys
from datetime import datetime
import time

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Setup logging
def setup_logger():
    logger = logging.getLogger('lstm_baseline')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logger()


class DocumentDataset(Dataset):
    """Dataset for loading document embeddings"""

    def __init__(self, data_dir: str, min_pages: int = 2, max_pages: int = 999,
                 shuffle: bool = True, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.documents = []
        self.shuffle = shuffle
        self.seed = seed

        logger.info(f"Loading documents from {data_dir}")
        logger.info(f"Page range filter: {min_pages}-{max_pages}")
        logger.info(f"Shuffle embeddings: {shuffle}")

        json_files = list(self.data_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} total files")

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                num_pages = data['num_pages']
                if min_pages <= num_pages <= max_pages:
                    embeddings = np.array(data['embeddings'], dtype=np.float32)
                    correct_order = data['correct_order']
                    page_numbers = data['page_numbers']

                    self.documents.append({
                        'doc_id': data['doc_id'],
                        'embeddings': embeddings,
                        'correct_order': correct_order,
                        'page_numbers': page_numbers,
                        'num_pages': num_pages
                    })
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue

        logger.info(f"Loaded {len(self.documents)} documents ({min_pages}-{max_pages} pages)")

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        doc = self.documents[idx]

        if self.shuffle:
            # Shuffle embeddings to create the ordering task
            num_pages = len(doc['embeddings'])
            shuffle_indices = list(range(num_pages))
            random.shuffle(shuffle_indices)

            shuffled_embeddings = doc['embeddings'][shuffle_indices]
            shuffled_page_numbers = [doc['page_numbers'][i] for i in shuffle_indices]

            return {
                'doc_id': doc['doc_id'],
                'embeddings': shuffled_embeddings,
                'page_numbers': shuffled_page_numbers,
                'correct_order': doc['correct_order'],
                'num_pages': doc['num_pages']
            }
        else:
            # No shuffle - for sanity checking
            return {
                'doc_id': doc['doc_id'],
                'embeddings': doc['embeddings'],
                'page_numbers': doc['page_numbers'],
                'correct_order': doc['correct_order'],
                'num_pages': doc['num_pages']
            }


class LSTMOrderingModel(nn.Module):
    """
    Bidirectional LSTM for page position prediction

    Architecture:
    - Input projection: embedding_dim -> hidden_dim
    - Bidirectional LSTM: processes sequence in both directions
    - Position scoring head: hidden_dim*2 -> 1 (position score)

    The model learns to predict absolute positions from embedding features.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        logger.info(f"Building LSTM model:")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Num layers: {num_layers}")
        logger.info(f"  Dropout: {dropout}")

        # Input projection
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # Divide by 2 since bidirectional
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Position scoring head
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"  Total parameters: {total_params:,}")

    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch_size, num_pages, embedding_dim)
        Returns:
            scores: (batch_size, num_pages) - position scores for each page
        """
        # Project embeddings
        x = self.input_proj(embeddings)  # (B, N, hidden_dim)

        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (B, N, hidden_dim)

        # Score each position
        scores = self.scorer(lstm_out).squeeze(-1)  # (B, N)

        return scores


class LSTMBaseline:
    """LSTM baseline with comprehensive training and evaluation"""

    def __init__(self, embedding_dim: int, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.3, device: str = 'cuda'):
        self.name = "LSTM"
        self.device = device
        self.model = LSTMOrderingModel(
            embedding_dim, hidden_dim, num_layers, dropout
        ).to(device)
        self.optimizer = None
        self.trained = False

        logger.info(f"Model moved to device: {device}")

    def train_model(self, train_dataset: Dataset, val_dataset: Dataset = None,
                   epochs: int = 30, lr: float = 1e-4, batch_size: int = 1):
        """
        Train the LSTM model

        Training objective: MSE between predicted positions and target positions
        Target positions are normalized to [0, 1] range
        """
        logger.info("=" * 80)
        logger.info("TRAINING LSTM BASELINE")
        logger.info("=" * 80)
        logger.info(f"Training set size: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"Validation set size: {len(val_dataset)}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Learning rate: {lr}")
        logger.info(f"Batch size: {batch_size}")
        logger.info("")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience = 0
        max_patience = 5

        for epoch in range(epochs):
            epoch_start = time.time()

            # ==================== TRAINING ====================
            self.model.train()
            train_losses = []

            for doc in train_dataset:
                embeddings = torch.FloatTensor(doc['embeddings']).unsqueeze(0).to(self.device)
                page_numbers = doc['page_numbers']
                correct_order = doc['correct_order']
                num_pages = len(doc['embeddings'])

                # Create target positions for the SHUFFLED embeddings
                # For each position i in the shuffled array, what position should it be in?
                target_positions = torch.zeros(num_pages, dtype=torch.float32)

                # Build mapping: page_number -> correct position in final sequence
                correct_pos = {page: pos for pos, page in enumerate(correct_order)}

                # For each shuffled position, assign its target position
                for i, page_num in enumerate(page_numbers):
                    if page_num in correct_pos:
                        target_positions[i] = correct_pos[page_num]

                # Normalize to [0, 1]
                if num_pages > 1:
                    target_positions = target_positions / (num_pages - 1)

                target_positions = target_positions.unsqueeze(0).to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                scores = self.model(embeddings)
                loss = criterion(scores, target_positions)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # ==================== VALIDATION ====================
            val_loss = None
            if val_dataset:
                self.model.eval()
                val_losses = []

                with torch.no_grad():
                    for doc in val_dataset:
                        embeddings = torch.FloatTensor(doc['embeddings']).unsqueeze(0).to(self.device)
                        page_numbers = doc['page_numbers']
                        correct_order = doc['correct_order']
                        num_pages = len(doc['embeddings'])

                        # Create target positions
                        target_positions = torch.zeros(num_pages, dtype=torch.float32)
                        correct_pos = {page: pos for pos, page in enumerate(correct_order)}

                        for i, page_num in enumerate(page_numbers):
                            if page_num in correct_pos:
                                target_positions[i] = correct_pos[page_num]

                        if num_pages > 1:
                            target_positions = target_positions / (num_pages - 1)

                        target_positions = target_positions.unsqueeze(0).to(self.device)

                        scores = self.model(embeddings)
                        loss = criterion(scores, target_positions)
                        val_losses.append(loss.item())

                val_loss = np.mean(val_losses)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

            # ==================== LOGGING ====================
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']

            if val_loss is not None:
                logger.info(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {epoch_time:.1f}s"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {epoch_time:.1f}s"
                )

            scheduler.step()

            # Early stopping check
            if patience >= max_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        self.trained = True
        logger.info("")
        logger.info("LSTM training complete")
        logger.info("=" * 80)
        logger.info("")

    def predict_order(self, embeddings: np.ndarray) -> List[int]:
        """
        Predict page order using trained LSTM

        Returns indices that would sort the pages correctly
        """
        if not self.trained:
            logger.warning("Model not trained! Returning random order.")
            order = list(range(len(embeddings)))
            random.shuffle(order)
            return order

        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(embeddings).unsqueeze(0).to(self.device)
            scores = self.model(x).squeeze(0).cpu().numpy()

            # Sort by scores to get predicted order
            order = np.argsort(scores).tolist()

        return order


def calculate_metrics(predicted_order: List[int], correct_order: List[int],
                     page_numbers: List[int]) -> Tuple[float, float]:
    """
    Calculate Kendall's Tau and accuracy metrics
    """
    # Map predicted indices to page numbers
    predicted_pages = [page_numbers[idx] for idx in predicted_order]

    # Calculate accuracy - exact position matches
    correct = sum(1 for pred, true in zip(predicted_pages, correct_order)
                  if pred == true)
    accuracy = correct / len(predicted_pages) if predicted_pages else 0

    # Kendall's tau - compare ranking
    true_positions = {page: i for i, page in enumerate(correct_order)}
    pred_positions = {page: i for i, page in enumerate(predicted_pages)}

    common_pages = set(true_positions.keys()) & set(pred_positions.keys())
    if len(common_pages) > 1:
        true_pos = [true_positions[p] for p in common_pages]
        pred_pos = [pred_positions[p] for p in common_pages]
        tau, _ = kendalltau(true_pos, pred_pos)
    else:
        tau = 0.0

    return tau, accuracy


def evaluate_baseline(baseline: LSTMBaseline, test_dataset: Dataset,
                      page_ranges: List[Tuple[int, int, str]]) -> Dict:
    """Evaluate LSTM baseline on test dataset"""

    logger.info("Evaluating LSTM baseline on test set...")
    logger.info("")

    # Group documents by page range
    range_results = {name: {'taus': [], 'accs': [], 'count': 0}
                    for _, _, name in page_ranges}

    for idx, doc in enumerate(test_dataset):
        if (idx + 1) % 500 == 0:
            logger.info(f"  Processed {idx + 1}/{len(test_dataset)} documents...")

        embeddings = doc['embeddings']
        correct_order = doc['correct_order']
        page_numbers = doc['page_numbers']
        num_pages = doc['num_pages']

        # Predict order
        predicted_order = baseline.predict_order(embeddings)

        # Calculate metrics
        tau, acc = calculate_metrics(predicted_order, correct_order, page_numbers)

        # Add to appropriate range
        for min_p, max_p, name in page_ranges:
            if min_p <= num_pages <= max_p:
                range_results[name]['taus'].append(tau)
                range_results[name]['accs'].append(acc)
                range_results[name]['count'] += 1
                break

    logger.info(f"  Completed evaluation on {len(test_dataset)} documents")
    logger.info("")

    # Compute averages
    results = {}
    for name, data in range_results.items():
        if data['count'] > 0:
            results[name] = {
                'tau': np.mean(data['taus']),
                'acc': np.mean(data['accs']),
                'count': data['count']
            }
        else:
            results[name] = {'tau': 0.0, 'acc': 0.0, 'count': 0}

    return results


def print_results(baseline_name: str, results: Dict, page_ranges: List[Tuple[int, int, str]]):
    """Print results in format similar to training output"""

    logger.info("=" * 80)
    logger.info(f"{baseline_name.upper()} BASELINE RESULTS")
    logger.info("=" * 80)
    logger.info("Validation Performance by Range:")

    for min_p, max_p, name in page_ranges:
        if name in results and results[name]['count'] > 0:
            tau = results[name]['tau']
            acc = results[name]['acc']
            count = results[name]['count']

            # Format similar to training output
            prefix = "    <�" if name in ["2-5", "6-10"] else "       "
            logger.info(
                f"{prefix} {name:8s} ({min_p:2d}-{max_p:3d}): "
                f"Tau={tau:6.3f} Acc={acc:5.3f} N={count:3d}"
            )


def sanity_check(baseline: LSTMBaseline, device: str):
    """
    Sanity check: Test on unshuffled data
    Should get near-perfect accuracy if model is working correctly
    """
    logger.info("=" * 80)
    logger.info("SANITY CHECK: Testing on unshuffled data")
    logger.info("=" * 80)
    logger.info("If the model is working correctly, it should get high accuracy")
    logger.info("on unshuffled data (embeddings already in correct order).")
    logger.info("")

    # Create small test dataset without shuffling
    test_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    sanity_dataset = DocumentDataset(test_data_dir, min_pages=2, max_pages=10, shuffle=False)

    # Take only first 50 docs for quick test
    sanity_subset = [sanity_dataset[i] for i in range(min(50, len(sanity_dataset)))]

    taus = []
    accs = []

    for doc in sanity_subset:
        embeddings = doc['embeddings']
        page_numbers = doc['page_numbers']
        correct_order = doc['correct_order']

        predicted_order = baseline.predict_order(embeddings)
        tau, acc = calculate_metrics(predicted_order, correct_order, page_numbers)

        taus.append(tau)
        accs.append(acc)

    avg_tau = np.mean(taus)
    avg_acc = np.mean(accs)

    logger.info(f"Sanity check results (N={len(sanity_subset)}):")
    logger.info(f"  Average Tau: {avg_tau:.4f}")
    logger.info(f"  Average Acc: {avg_acc:.4f}")
    logger.info("")

    if avg_tau > 0.9:
        logger.info(" PASS: Model correctly predicts order on unshuffled data")
    elif avg_tau > 0.7:
        logger.info("�  WARNING: Model shows moderate performance on unshuffled data")
    else:
        logger.info("L FAIL: Model performs poorly even on unshuffled data - something may be wrong")

    logger.info("=" * 80)
    logger.info("")


def main():
    # Configuration
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define page ranges (matching training output)
    page_ranges = [
        (2, 5, "2-5"),
        (6, 10, "6-10"),
        (11, 15, "11-15"),
        (16, 20, "16-20"),
        (21, 25, "21-25"),
        (26, 30, "26-30"),
        (31, 40, "31-40"),
        (41, 999, "41+")
    ]

    logger.info("=" * 80)
    logger.info("LSTM BASELINE FOR DOCUMENT PAGE ORDERING")
    logger.info("=" * 80)
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Device: {device}")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    # Load datasets with PROPER train/val/test split (NO LEAKAGE!)
    logger.info("Loading datasets with proper 70/15/15 split...")
    logger.info("")

    # Load ALL documents in training range (2-25 pages)
    full_dataset = DocumentDataset(DATA_DIR, min_pages=2, max_pages=25, shuffle=True)
    total_size = len(full_dataset)

    logger.info(f"Total documents (2-25 pages): {total_size}")

    # Create NON-OVERLAPPING train/val/test splits
    random.seed(42)  # For reproducibility
    all_indices = list(range(total_size))
    random.shuffle(all_indices)

    # Split: 70% train, 15% val, 15% test
    train_size = int(0.70 * total_size)
    train_size = min(train_size, 1000)  # Cap at 1000 for faster training
    val_size = int(0.15 * total_size)
    val_size = min(val_size, 200)
    test_size = total_size - train_size - val_size

    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]

    logger.info(f"  Train set: {train_size} documents")
    logger.info(f"  Val set: {val_size} documents")
    logger.info(f"  Test set (held-out): {test_size} documents")
    logger.info("")

    # SubsetDataset class that calls base dataset's __getitem__ for fresh shuffles
    class SubsetDataset:
        def __init__(self, base_dataset, indices):
            self.base_dataset = base_dataset
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            # Call base_dataset's __getitem__ to get FRESH shuffle each time
            return self.base_dataset[self.indices[idx]]

    train_dataset_limited = SubsetDataset(full_dataset, train_indices)
    val_dataset = SubsetDataset(full_dataset, val_indices)

    # Store held-out test set for final evaluation
    held_out_test_set = [full_dataset.documents[idx] for idx in test_indices]

    logger.info(f"Training set: {len(train_dataset_limited)} documents")
    logger.info(f"Validation set: {len(val_dataset)} documents")
    logger.info("")

    # Get embedding dimension
    embedding_dim = full_dataset[0]['embeddings'].shape[1]
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info("")

    # Initialize model
    lstm_baseline = LSTMBaseline(
        embedding_dim=embedding_dim,
        hidden_dim=512,
        num_layers=2,
        dropout=0.3,
        device=device
    )
    logger.info("")

    # Train model
    lstm_baseline.train_model(
        train_dataset=train_dataset_limited,
        val_dataset=val_dataset,
        epochs=30,
        lr=1e-4,
        batch_size=1
    )

    # Sanity check
    sanity_check(lstm_baseline, device)

    # Evaluate on HELD-OUT test set (never seen during training!)
    logger.info("Evaluating on held-out test set...")
    logger.info(f"Held-out test set size: {len(held_out_test_set)} documents")
    logger.info("")

    # Create a simple wrapper for held-out test set
    class TestDataset:
        def __init__(self, documents):
            self.documents = documents
        def __len__(self):
            return len(self.documents)
        def __getitem__(self, idx):
            doc = self.documents[idx]
            # Return with shuffled embeddings
            num_pages = len(doc['embeddings'])
            shuffle_indices = list(range(num_pages))
            random.shuffle(shuffle_indices)

            shuffled_embeddings = doc['embeddings'][shuffle_indices]
            shuffled_page_numbers = [doc['page_numbers'][i] for i in shuffle_indices]

            return {
                'doc_id': doc['doc_id'],
                'embeddings': shuffled_embeddings,
                'page_numbers': shuffled_page_numbers,
                'correct_order': doc['correct_order'],
                'num_pages': doc['num_pages']
            }

    test_dataset = TestDataset(held_out_test_set)

    # Evaluate
    lstm_results = evaluate_baseline(lstm_baseline, test_dataset, page_ranges)
    print_results("LSTM", lstm_results, page_ranges)
    logger.info("")

    # Final summary (matching specialized models format)
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE - FINAL RESULTS")
    logger.info("=" * 80)

    logger.info("LSTM BASELINE:")
    key_ranges = [
        ("2-5", (2, 5)),
        ("6-10", (6, 10)),
        ("11-15", (11, 15)),
        ("16-20", (16, 20)),
        ("21-25", (21, 25))
    ]

    for range_name, (min_p, max_p) in key_ranges:
        if range_name in lstm_results and lstm_results[range_name]['count'] > 0:
            tau = lstm_results[range_name]['tau']
            logger.info(f"{range_name:15s} ({min_p:2d}-{max_p:2d} pages): Target Tau = {tau:.4f}")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
