#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POINTER NETWORK BASELINE
========================

Implementation of Pointer Networks (Vinyals et al., 2015) for document ordering.
This is the classic neural baseline for combinatorial ordering tasks.

Architecture:
- Encoder: Bidirectional LSTM that processes all page embeddings
- Decoder: LSTM with attention mechanism that "points" to pages sequentially
- At each decoding step, attention scores determine which page comes next

Reference: "Pointer Networks" (Vinyals, Fortunato, Jaitly, 2015)
https://arxiv.org/abs/1506.03134
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import math
import random
from scipy.stats import kendalltau
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import sys
from datetime import datetime
import time

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Setup logging
def setup_logger():
    logger = logging.getLogger('pointer_network')
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
                 shuffle: bool = True):
        self.data_dir = Path(data_dir)
        self.documents = []
        self.shuffle = shuffle

        json_files = list(self.data_dir.glob("*.json"))

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
                continue

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        doc = self.documents[idx]

        if self.shuffle:
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
                'num_pages': doc['num_pages'],
                'shuffle_indices': shuffle_indices
            }
        else:
            return {
                'doc_id': doc['doc_id'],
                'embeddings': doc['embeddings'],
                'page_numbers': doc['page_numbers'],
                'correct_order': doc['correct_order'],
                'num_pages': doc['num_pages'],
                'shuffle_indices': list(range(len(doc['embeddings'])))
            }


class PointerNetworkAttention(nn.Module):
    """
    Attention mechanism for Pointer Networks.
    Computes attention scores between decoder state and all encoder outputs.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Attention parameters (additive attention)
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)  # For encoder states
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)  # For decoder state
        self.v = nn.Linear(hidden_dim, 1, bias=False)  # Final attention projection

    def forward(self, decoder_state: torch.Tensor, encoder_outputs: torch.Tensor,
                mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention scores (pointers).

        Args:
            decoder_state: (batch_size, hidden_dim) - current decoder state
            encoder_outputs: (batch_size, seq_len, hidden_dim) - all encoder outputs
            mask: (batch_size, seq_len) - mask for already selected positions

        Returns:
            attention_weights: (batch_size, seq_len) - pointer distribution
            context: (batch_size, hidden_dim) - attended context vector
        """
        batch_size, seq_len, hidden_dim = encoder_outputs.size()

        # Expand decoder state to match encoder outputs
        # (batch_size, hidden_dim) -> (batch_size, 1, hidden_dim) -> (batch_size, seq_len, hidden_dim)
        decoder_state_expanded = decoder_state.unsqueeze(1).expand(-1, seq_len, -1)

        # Additive attention: tanh(W1 * encoder + W2 * decoder)
        encoder_features = self.W1(encoder_outputs)  # (batch_size, seq_len, hidden_dim)
        decoder_features = self.W2(decoder_state_expanded)  # (batch_size, seq_len, hidden_dim)

        combined = torch.tanh(encoder_features + decoder_features)  # (batch_size, seq_len, hidden_dim)

        # Compute attention scores
        scores = self.v(combined).squeeze(-1)  # (batch_size, seq_len)

        # Apply mask (prevent pointing to already selected positions)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax to get pointer distribution
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len)

        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return attention_weights, context


class PointerNetwork(nn.Module):
    """
    Pointer Network for document ordering.

    Encoder: Bidirectional LSTM processes all page embeddings
    Decoder: LSTM with attention mechanism points to pages in order
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)

        # Encoder: Bidirectional LSTM
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # Will be doubled by bidirectional
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Decoder: Unidirectional LSTM
        self.decoder = nn.LSTM(
            input_size=hidden_dim,  # Takes context from attention
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = PointerNetworkAttention(hidden_dim)

        # Initialize decoder hidden state from encoder
        self.decoder_state_init = nn.Linear(hidden_dim * num_layers, hidden_dim * num_layers)

        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"  Pointer Network parameters: {total_params:,}")

    def encode(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        Encode all page embeddings.

        Args:
            embeddings: (batch_size, seq_len, embedding_dim)

        Returns:
            encoder_outputs: (batch_size, seq_len, hidden_dim)
            encoder_hidden: tuple of (h, c) for LSTM
        """
        # Project embeddings
        x = self.input_proj(embeddings)  # (batch_size, seq_len, hidden_dim)

        # Encode with bidirectional LSTM
        encoder_outputs, (h, c) = self.encoder(x)

        return encoder_outputs, (h, c)

    def decode_step(self, context: torch.Tensor, decoder_hidden: Tuple,
                   encoder_outputs: torch.Tensor, mask: torch.Tensor = None):
        """
        Single decoding step: point to next page.

        Args:
            context: (batch_size, hidden_dim) - previous context
            decoder_hidden: tuple of (h, c) for decoder LSTM
            encoder_outputs: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, seq_len) - mask for already selected positions

        Returns:
            pointer_probs: (batch_size, seq_len) - pointer distribution
            decoder_hidden: updated decoder hidden state
            context: new context vector
        """
        # Decoder LSTM step
        context_input = context.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        decoder_output, decoder_hidden = self.decoder(context_input, decoder_hidden)
        decoder_state = decoder_output.squeeze(1)  # (batch_size, hidden_dim)

        # Attention: compute pointers
        pointer_probs, new_context = self.attention(decoder_state, encoder_outputs, mask)

        return pointer_probs, decoder_hidden, new_context

    def forward(self, embeddings: torch.Tensor, target_order: torch.Tensor = None):
        """
        Full forward pass: encode all pages, then decode the ordering.

        Args:
            embeddings: (batch_size, seq_len, embedding_dim)
            target_order: (batch_size, seq_len) - ground truth ordering indices

        Returns:
            pointer_log_probs: (batch_size, seq_len, seq_len) - log probabilities for training
            predicted_order: (batch_size, seq_len) - predicted ordering
        """
        batch_size, seq_len, _ = embeddings.size()
        device = embeddings.device

        # Encode
        encoder_outputs, (h, c) = self.encode(embeddings)

        # Initialize decoder hidden state from encoder
        # Reshape encoder hidden: (num_layers * 2, batch, hidden//2) -> (num_layers, batch, hidden)
        h = h.view(self.num_layers, 2, batch_size, self.hidden_dim // 2)
        h = torch.cat([h[:, 0, :, :], h[:, 1, :, :]], dim=-1)  # Concat forward/backward

        c = c.view(self.num_layers, 2, batch_size, self.hidden_dim // 2)
        c = torch.cat([c[:, 0, :, :], c[:, 1, :, :]], dim=-1)

        decoder_hidden = (h, c)

        # Initial context: mean of encoder outputs
        context = encoder_outputs.mean(dim=1)  # (batch_size, hidden_dim)

        # Decode step by step
        pointer_log_probs = []
        predicted_order = []
        mask = torch.ones(batch_size, seq_len, device=device)  # 1 = available, 0 = selected

        for step in range(seq_len):
            # Decode one step
            pointer_probs, decoder_hidden, context = self.decode_step(
                context, decoder_hidden, encoder_outputs, mask
            )

            # Log probabilities for loss
            pointer_log_probs.append(torch.log(pointer_probs + 1e-10))

            # Greedy selection for prediction
            selected_idx = pointer_probs.argmax(dim=-1)  # (batch_size,)
            predicted_order.append(selected_idx)

            # Update mask (mark selected position as unavailable)
            mask.scatter_(1, selected_idx.unsqueeze(1), 0)

        # Stack results
        pointer_log_probs = torch.stack(pointer_log_probs, dim=1)  # (batch_size, seq_len, seq_len)
        predicted_order = torch.stack(predicted_order, dim=1)  # (batch_size, seq_len)

        return pointer_log_probs, predicted_order

    def predict_order(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict ordering for inference"""
        _, predicted_order = self.forward(embeddings)
        return predicted_order


class PointerNetworkBaseline:
    """Pointer Network baseline with training"""

    def __init__(self, embedding_dim: int, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.2,
                 device: str = 'cuda', model_name: str = "Pointer-Network"):
        self.name = model_name
        self.device = device
        self.model = PointerNetwork(
            embedding_dim, hidden_dim, num_layers, dropout
        ).to(device)
        self.optimizer = None
        self.trained = False

    def compute_loss(self, pointer_log_probs: torch.Tensor,
                    target_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.

        Args:
            pointer_log_probs: (batch_size, seq_len, seq_len) - log P(pointer | step)
            target_indices: (batch_size, seq_len) - ground truth indices to point to

        Returns:
            loss: scalar
        """
        batch_size, seq_len, _ = pointer_log_probs.size()

        # Gather log probabilities of target indices
        # For each decoding step, get log prob of correct pointer
        target_indices = target_indices.unsqueeze(-1)  # (batch_size, seq_len, 1)
        log_probs = pointer_log_probs.gather(2, target_indices).squeeze(-1)  # (batch_size, seq_len)

        # Average negative log likelihood
        loss = -log_probs.mean()

        return loss

    def train_model(self, train_dataset: Dataset, val_dataset: Dataset = None,
                   epochs: int = 40, lr: float = 1e-4):
        """
        Train Pointer Network.

        Training is more challenging than position prediction because we need to
        learn the sequential pointing behavior.
        """
        logger.info(f"Training {self.name}")
        logger.info(f"  Training samples: {len(train_dataset)}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Learning rate: {lr}")
        logger.info("")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999)
        )

        # Cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=lr * 0.1
        )

        best_val_tau = 0.0
        patience = 0
        max_patience = 10

        # Print header for epoch logs
        logger.info("=" * 145)
        logger.info("Epoch | Train Loss | Val Loss | 2-5 Tau | 2-5 Acc | 6-10 Tau | 6-10 Acc | 11-15 Tau | 16-20 Tau | 21+ Tau | Avg Tau |   LR   | Time")
        logger.info("-" * 145)

        for epoch in range(epochs):
            epoch_start = time.time()

            # ==================== TRAINING ====================
            self.model.train()
            train_losses = []

            for doc in train_dataset:
                embeddings = torch.FloatTensor(doc['embeddings']).unsqueeze(0).to(self.device)
                page_numbers = doc['page_numbers']
                correct_order = doc['correct_order']
                shuffle_indices = doc['shuffle_indices']

                # Create target indices: for each position in correct order, find its index in shuffled
                # We need to map: "which shuffled position should we point to at each step?"
                target_indices = []
                correct_pos = {page: pos for pos, page in enumerate(correct_order)}

                # For each step, find which shuffled index corresponds to correct page
                shuffled_pages = [page_numbers[i] for i in range(len(page_numbers))]
                for correct_page in correct_order:
                    if correct_page in shuffled_pages:
                        target_idx = shuffled_pages.index(correct_page)
                        target_indices.append(target_idx)

                if len(target_indices) != len(embeddings[0]):
                    continue  # Skip if mismatch

                target_indices = torch.LongTensor(target_indices).unsqueeze(0).to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                pointer_log_probs, _ = self.model(embeddings)
                loss = self.compute_loss(pointer_log_probs, target_indices)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses) if train_losses else 0

            # ==================== VALIDATION ====================
            val_loss = None
            val_tau = 0.0
            tau_by_range = {}
            acc_by_range = {}

            if val_dataset:
                self.model.eval()
                val_losses = []

                # Track Tau and Accuracy by page range
                range_taus = {
                    '2-5': [],
                    '6-10': [],
                    '11-15': [],
                    '16-20': [],
                    '21+': []
                }
                range_accs = {
                    '2-5': [],
                    '6-10': [],
                    '11-15': [],
                    '16-20': [],
                    '21+': []
                }

                with torch.no_grad():
                    for doc in val_dataset:
                        embeddings = torch.FloatTensor(doc['embeddings']).unsqueeze(0).to(self.device)
                        page_numbers = doc['page_numbers']
                        correct_order = doc['correct_order']
                        num_pages = len(doc['embeddings'])

                        # Compute loss
                        shuffled_pages = page_numbers
                        target_indices = []
                        for correct_page in correct_order:
                            if correct_page in shuffled_pages:
                                target_idx = shuffled_pages.index(correct_page)
                                target_indices.append(target_idx)

                        if len(target_indices) == len(embeddings[0]):
                            target_indices = torch.LongTensor(target_indices).unsqueeze(0).to(self.device)
                            pointer_log_probs, predicted_order = self.model(embeddings)
                            loss = self.compute_loss(pointer_log_probs, target_indices)
                            val_losses.append(loss.item())

                            # Compute metrics
                            predicted_order = predicted_order.squeeze(0).cpu().numpy()
                            predicted_pages = [page_numbers[idx] for idx in predicted_order]

                            true_positions = {page: i for i, page in enumerate(correct_order)}
                            pred_positions = {page: i for i, page in enumerate(predicted_pages)}
                            common_pages = set(true_positions.keys()) & set(pred_positions.keys())

                            if len(common_pages) > 1:
                                true_pos = [true_positions[p] for p in common_pages]
                                pred_pos = [pred_positions[p] for p in common_pages]
                                tau, _ = kendalltau(true_pos, pred_pos)

                                # Calculate accuracy
                                correct = sum(1 for pred, true in zip(predicted_pages, correct_order) if pred == true)
                                accuracy = correct / len(predicted_pages)

                                # Add to appropriate range
                                if 2 <= num_pages <= 5:
                                    range_taus['2-5'].append(tau)
                                    range_accs['2-5'].append(accuracy)
                                elif 6 <= num_pages <= 10:
                                    range_taus['6-10'].append(tau)
                                    range_accs['6-10'].append(accuracy)
                                elif 11 <= num_pages <= 15:
                                    range_taus['11-15'].append(tau)
                                    range_accs['11-15'].append(accuracy)
                                elif 16 <= num_pages <= 20:
                                    range_taus['16-20'].append(tau)
                                    range_accs['16-20'].append(accuracy)
                                elif num_pages >= 21:
                                    range_taus['21+'].append(tau)
                                    range_accs['21+'].append(accuracy)

                val_loss = np.mean(val_losses) if val_losses else 0

                # Compute average Tau and Accuracy per range
                for range_name, taus in range_taus.items():
                    if taus:
                        tau_by_range[range_name] = np.mean(taus)
                        acc_by_range[range_name] = np.mean(range_accs[range_name])
                    else:
                        tau_by_range[range_name] = 0.0
                        acc_by_range[range_name] = 0.0

                # Overall average Tau (for early stopping)
                all_taus = [tau for taus in range_taus.values() for tau in taus]
                val_tau = np.mean(all_taus) if all_taus else 0.0

                # Track best based on Tau
                if val_tau > best_val_tau:
                    best_val_tau = val_tau
                    patience = 0
                else:
                    patience += 1

            # ==================== LOGGING ====================
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']

            if val_loss is not None:
                logger.info(
                    f"{epoch+1:5d} | "
                    f"{avg_train_loss:10.4f} | "
                    f"{val_loss:8.4f} | "
                    f"{tau_by_range.get('2-5', 0.0):8.4f} | "
                    f"{acc_by_range.get('2-5', 0.0):8.4f} | "
                    f"{tau_by_range.get('6-10', 0.0):9.4f} | "
                    f"{acc_by_range.get('6-10', 0.0):9.4f} | "
                    f"{tau_by_range.get('11-15', 0.0):10.4f} | "
                    f"{tau_by_range.get('16-20', 0.0):10.4f} | "
                    f"{tau_by_range.get('21+', 0.0):8.4f} | "
                    f"{val_tau:8.4f} | "
                    f"{current_lr:.5f} | "
                    f"{epoch_time:5.1f}s"
                )
            else:
                logger.info(
                    f"{epoch+1:5d} | "
                    f"{avg_train_loss:10.4f} | "
                    f"   N/A   | "
                    f"  N/A   | "
                    f"  N/A   | "
                    f"   N/A    | "
                    f"   N/A    | "
                    f"    N/A     | "
                    f"    N/A     | "
                    f"  N/A   | "
                    f"  N/A    | "
                    f"{current_lr:.5f} | "
                    f"{epoch_time:5.1f}s"
                )

            scheduler.step()

            if patience >= max_patience:
                logger.info(f"\nEarly stopping at epoch {epoch+1} (best val tau: {best_val_tau:.4f})")
                break

        logger.info("-" * 145)
        logger.info("=" * 145)

        self.trained = True
        logger.info(f"{self.name} training complete")
        logger.info("")

    def predict_order(self, embeddings: np.ndarray) -> List[int]:
        """Predict page order"""
        if not self.trained:
            order = list(range(len(embeddings)))
            random.shuffle(order)
            return order

        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(embeddings).unsqueeze(0).to(self.device)
            predicted_order = self.model.predict_order(x).squeeze(0).cpu().numpy()
            order = predicted_order.tolist()

        return order


class SinglePointerNetwork:
    """
    Single Pointer Network for all document lengths

    Strategy: Train 1 general model for all documents (2-25 pages)
    """

    def __init__(self, embedding_dim: int, device: str = 'cuda'):
        self.device = device
        self.embedding_dim = embedding_dim

        # Track held-out test set (reserved BEFORE training)
        self.test_set = []

        logger.info("=" * 80)
        logger.info("CREATING SINGLE GENERAL POINTER NETWORK")
        logger.info("=" * 80)
        logger.info("")

        # Create single general model
        self.model = PointerNetworkBaseline(
            embedding_dim, hidden_dim=512, num_layers=2, dropout=0.2,
            device=device, model_name="General-PointerNet (2-25 pages)"
        )

        logger.info("Single general model created")
        logger.info("")

    def train_all(self, data_dir: str):
        """Train single general model with proper 70/15/15 split"""

        logger.info("=" * 80)
        logger.info("TRAINING GENERAL MODEL WITH 70/15/15 SPLIT")
        logger.info("=" * 80)
        logger.info("")

        # Fixed SubsetDataset - re-shuffles each epoch instead of memorizing!
        class SubsetDataset:
            def __init__(self, base_dataset, indices):
                self.base_dataset = base_dataset
                self.indices = indices
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, idx):
                # Call base_dataset's __getitem__ to get FRESH shuffle each time!
                return self.base_dataset[self.indices[idx]]

        # ==================== Train General Model ====================
        logger.info("=" * 80)
        logger.info("TRAINING GENERAL MODEL (2-25 pages)")
        logger.info("=" * 80)
        logger.info("")

        all_data = DocumentDataset(data_dir, min_pages=2, max_pages=25, shuffle=True)
        total_size = len(all_data)

        # 70/15/15 split (train/val/test)
        random.seed(42)
        all_indices = list(range(total_size))
        random.shuffle(all_indices)

        train_size = int(0.70 * total_size)
        train_size = min(train_size, 1000)  # Cap at 1000 for faster training
        val_size = int(0.15 * total_size)
        val_size = min(val_size, 200)
        test_size = total_size - train_size - val_size

        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size:train_size + val_size]
        test_indices = all_indices[train_size + val_size:]

        # Store held-out test set for later
        self.test_set = [all_data.documents[idx] for idx in test_indices]

        logger.info(f"  Split: Train={train_size}, Val={val_size}, Test={test_size}")

        self.model.train_model(
            SubsetDataset(all_data, train_indices),
            SubsetDataset(all_data, val_indices),
            epochs=40,
            lr=1e-4
        )

        logger.info("=" * 80)
        logger.info("MODEL TRAINED")
        logger.info(f"Total held-out test documents: {len(self.test_set)}")
        logger.info("=" * 80)
        logger.info("")

    def predict_order(self, embeddings: np.ndarray) -> List[int]:
        """
        Predict order using the general model
        """
        return self.model.predict_order(embeddings)


def calculate_metrics(predicted_order: List[int], correct_order: List[int],
                     page_numbers: List[int]) -> Tuple[float, float]:
    """Calculate Kendall's Tau and accuracy"""
    predicted_pages = [page_numbers[idx] for idx in predicted_order]

    correct = sum(1 for pred, true in zip(predicted_pages, correct_order)
                  if pred == true)
    accuracy = correct / len(predicted_pages) if predicted_pages else 0

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


def evaluate_single_model_on_held_out_test(model_wrapper: SinglePointerNetwork,
                                            page_ranges: List[Tuple[int, int, str]]) -> Dict:
    """Evaluate single model on held-out test set (reserved during training)"""

    logger.info("=" * 80)
    logger.info("EVALUATING MODEL ON HELD-OUT TEST SET")
    logger.info("=" * 80)
    logger.info("")

    all_test_docs = model_wrapper.test_set
    logger.info(f"Total held-out test documents: {len(all_test_docs)}")
    logger.info("")

    range_results = {name: {'taus': [], 'accs': [], 'count': 0}
                    for _, _, name in page_ranges}

    logger.info(f"Evaluating on {len(all_test_docs)} held-out test documents...")

    for idx, doc in enumerate(all_test_docs):
        if (idx + 1) % 100 == 0:
            logger.info(f"  Processed {idx + 1}/{len(all_test_docs)} documents...")

        embeddings = doc['embeddings']
        correct_order = doc['correct_order']
        page_numbers = doc['page_numbers']
        num_pages = doc['num_pages']

        predicted_order = model_wrapper.predict_order(embeddings)
        tau, acc = calculate_metrics(predicted_order, correct_order, page_numbers)

        for min_p, max_p, name in page_ranges:
            if min_p <= num_pages <= max_p:
                range_results[name]['taus'].append(tau)
                range_results[name]['accs'].append(acc)
                range_results[name]['count'] += 1
                break

    logger.info(f"Evaluated on {sum(d['count'] for d in range_results.values())} test documents")
    logger.info("")

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


def print_results(results: Dict, page_ranges: List[Tuple[int, int, str]]):
    """Print results"""

    logger.info("=" * 80)
    logger.info("POINTER NETWORK - FINAL RESULTS")
    logger.info("=" * 80)
    logger.info("Validation Performance by Range:")
    logger.info("")

    for min_p, max_p, name in page_ranges:
        if name in results and results[name]['count'] > 0:
            tau = results[name]['tau']
            acc = results[name]['acc']
            count = results[name]['count']

            logger.info(
                f"    {name:8s} ({min_p:2d}-{max_p:3d}): "
                f"Tau={tau:6.4f} Acc={acc:5.3f} N={count:3d}"
            )

    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)

    key_ranges = [
        ("2-5", (2, 5)),
        ("6-10", (6, 10)),
        ("11-15", (11, 15)),
        ("16-20", (16, 20)),
        ("21-25", (21, 25))
    ]

    for range_name, (min_p, max_p) in key_ranges:
        if range_name in results and results[range_name]['count'] > 0:
            tau = results[range_name]['tau']
            logger.info(f"{range_name:15s} ({min_p:2d}-{max_p:2d} pages): Target Tau = {tau:.4f}")

    logger.info("=" * 80)


def main():
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    page_ranges = [
        (2, 5, "2-5"),
        (6, 10, "6-10"),
        (11, 15, "11-15"),
        (16, 20, "16-20"),
        (21, 25, "21-25")
    ]

    logger.info("=" * 80)
    logger.info("POINTER NETWORK BASELINE (Vinyals et al., 2015)")
    logger.info("=" * 80)
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Device: {device}")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info("Strategy: Single general Pointer Network")
    logger.info("  - General model: 2-25 pages")
    logger.info("")
    logger.info("Architecture:")
    logger.info("  - Encoder: Bidirectional LSTM (2 layers, 512 hidden)")
    logger.info("  - Decoder: LSTM with attention mechanism")
    logger.info("  - Attention: Points to next page at each decoding step")
    logger.info("=" * 80)
    logger.info("")

    # Get embedding dimension
    temp_dataset = DocumentDataset(DATA_DIR, min_pages=2, max_pages=5, shuffle=True)
    embedding_dim = temp_dataset[0]['embeddings'].shape[1]
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info("")

    # Create and train single model (reserves held-out test set during training)
    model_wrapper = SinglePointerNetwork(embedding_dim, device)
    model_wrapper.train_all(DATA_DIR)

    # Evaluate on held-out test set (reserved during training, NEVER seen by model)
    results = evaluate_single_model_on_held_out_test(model_wrapper, page_ranges)
    print_results(results, page_ranges)


if __name__ == "__main__":
    main()
