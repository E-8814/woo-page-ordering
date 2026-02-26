#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformer Encoder-Decoder WITHOUT Positional Encodings
Ablation study: Does removing positional encodings fix the collapse?
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import logging
from pathlib import Path
import random
from scipy.stats import kendalltau
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class OrderingDataset(Dataset):
    def __init__(self, data_dir, min_len=2, max_len=25, split='train',
                 train_ratio=0.85, seed=42):
        self.data_dir = Path(data_dir)
        self.documents = []

        embedding_files = list(self.data_dir.glob("*_embeddings.json"))

        valid_docs = []
        for file_path in embedding_files:
            try:
                with open(file_path, 'r') as f:
                    doc_data = json.load(f)

                if 'correct_order' not in doc_data:
                    continue

                doc_length = len(doc_data['embeddings'])
                if min_len <= doc_length <= max_len:
                    valid_docs.append({
                        'embeddings': torch.tensor(doc_data['embeddings'], dtype=torch.float32),
                        'page_numbers': torch.tensor(doc_data['page_numbers'], dtype=torch.long),
                        'correct_order': torch.tensor(doc_data['correct_order'], dtype=torch.long),
                        'length': doc_length
                    })
            except:
                continue

        # Split
        random.seed(seed)
        random.shuffle(valid_docs)
        n_train = int(len(valid_docs) * train_ratio)

        if split == 'train':
            self.documents = valid_docs[:n_train]
        else:
            self.documents = valid_docs[n_train:]

        logger.info(f"{split} set: {len(self.documents)} documents ({min_len}-{max_len} pages)")

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        doc = self.documents[idx]
        n = doc['length']

        # Shuffle input
        perm = torch.randperm(n)
        shuffled_embeddings = doc['embeddings'][perm]
        shuffled_pages = doc['page_numbers'][perm]

        # Target: indices that would sort shuffled_pages to match correct_order
        page_to_shuffled_idx = {page.item(): i for i, page in enumerate(shuffled_pages)}
        target_indices = torch.tensor([
            page_to_shuffled_idx[page.item()]
            for page in doc['correct_order']
            if page.item() in page_to_shuffled_idx
        ], dtype=torch.long)

        return {
            'embeddings': shuffled_embeddings,
            'page_numbers': shuffled_pages,
            'correct_order': doc['correct_order'],
            'target_indices': target_indices,
            'length': n
        }


class TransformerSeq2SeqNoPosEncoding(nn.Module):
    """
    Transformer Encoder-Decoder WITHOUT positional encodings
    Ablation: Test if positional encoding is the failure point
    """

    def __init__(self, embedding_dim, hidden_dim=512, num_layers=6,
                 num_heads=8, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)

        # NO POSITIONAL ENCODINGS - this is the key change!

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output: pointer to input positions
        self.output_proj = nn.Linear(hidden_dim, 1)

        # Learnable start token
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def encode(self, embeddings):
        """Encode shuffled input - NO positional encoding added"""
        B, N, _ = embeddings.shape

        x = self.input_proj(embeddings)
        # NO positional encoding added here!

        memory = self.encoder(x)
        return memory

    def decode_step(self, decoder_input, memory, tgt_mask=None):
        """Single decoder step"""
        decoder_out = self.decoder(decoder_input, memory, tgt_mask=tgt_mask)

        # Compute attention scores over input positions
        scores = torch.matmul(decoder_out, memory.transpose(1, 2))  # (B, L, N)

        return scores

    def forward(self, embeddings, target_indices=None):
        """
        embeddings: (B, N, embedding_dim) - shuffled pages
        target_indices: (B, L) - target ordering as indices into embeddings
        """
        B, N, _ = embeddings.shape
        device = embeddings.device

        # Encode
        memory = self.encode(embeddings)

        if target_indices is not None:
            # Training mode
            target_len = target_indices.shape[1]

            # Build decoder input with start token
            decoder_input = self.start_token.expand(B, 1, -1)

            # Teacher forcing
            for i in range(target_len):
                idx = target_indices[:, i].unsqueeze(1).unsqueeze(2).expand(-1, -1, memory.shape[2])
                selected = torch.gather(memory, 1, idx)
                decoder_input = torch.cat([decoder_input, selected], dim=1)

            scores = self.decode_step(decoder_input[:, :-1, :], memory)  # (B, target_len, N)

            return scores
        else:
            # Inference mode - greedy decoding
            decoder_input = self.start_token.expand(B, 1, -1)
            selected_indices = []
            used_mask = torch.zeros(B, N, device=device, dtype=torch.bool)

            for i in range(N):
                scores = self.decode_step(decoder_input, memory)[:, -1, :]  # (B, N)

                # Mask already selected
                scores = scores.masked_fill(used_mask, -1e9)

                # Select best
                idx = scores.argmax(dim=-1)  # (B,)
                selected_indices.append(idx)

                # Update mask
                used_mask.scatter_(1, idx.unsqueeze(1), True)

                # Add to decoder input
                idx_expanded = idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, memory.shape[2])
                selected = torch.gather(memory, 1, idx_expanded)
                decoder_input = torch.cat([decoder_input, selected], dim=1)

            predicted_order = torch.stack(selected_indices, dim=1)  # (B, N)
            return predicted_order


def compute_loss(scores, target_indices):
    """
    scores: (B, L, N) - logits over input positions
    target_indices: (B, L) - ground truth indices
    """
    B, L, N = scores.shape

    # Handle shape mismatch: only use first L target indices
    target_indices = target_indices[:, :L]

    # Cross entropy loss
    loss = F.cross_entropy(
        scores.view(B * L, N),
        target_indices.view(B * L)
    )

    return loss


def compute_metrics(predicted_order, correct_order, page_numbers):
    """Compute accuracy and Kendall's tau"""
    predicted_pages = [page_numbers[idx].item() for idx in predicted_order]

    correct = sum(1 for pred, true in zip(predicted_pages, correct_order.cpu().numpy())
                  if pred == true)
    accuracy = correct / len(predicted_pages) if predicted_pages else 0

    # Kendall's tau
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


def evaluate(model, data_loader, device):
    """Evaluate model"""
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for batch in data_loader:
            embeddings = batch['embeddings'].to(device)
            page_numbers = batch['page_numbers'].squeeze(0)
            correct_order = batch['correct_order'].squeeze(0)

            if len(embeddings.squeeze(0)) < 2:
                continue

            # Get prediction
            predicted_indices = model(embeddings)
            predicted_indices = predicted_indices.squeeze(0).cpu()

            metrics = compute_metrics(predicted_indices, correct_order, page_numbers)
            all_metrics.append(metrics)

    if all_metrics:
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
            'kendall_tau': np.mean([m['kendall_tau'] for m in all_metrics])
        }
    else:
        avg_metrics = {'accuracy': 0.0, 'kendall_tau': 0.0}

    return avg_metrics


def train_model(data_dir, device='cuda', epochs=50, hidden_dim=512, num_layers=6):
    """Train transformer seq2seq model WITHOUT positional encodings"""

    # Load data
    train_dataset = OrderingDataset(data_dir, min_len=2, max_len=25, split='train')
    val_dataset = OrderingDataset(data_dir, min_len=2, max_len=25, split='val')

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Get embedding dimension
    embedding_dim = train_dataset[0]['embeddings'].shape[1]
    logger.info(f"Embedding dimension: {embedding_dim}")

    # Create model WITHOUT positional encodings
    model = TransformerSeq2SeqNoPosEncoding(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=8,
        dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda') if device == 'cuda' else None

    best_tau = -1.0

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []

        for batch in train_loader:
            embeddings = batch['embeddings'].to(device)
            target_indices = batch['target_indices'].to(device)

            if len(embeddings.squeeze(0)) < 2:
                continue

            optimizer.zero_grad()

            if scaler:
                with autocast('cuda'):
                    scores = model(embeddings, target_indices)
                    loss = compute_loss(scores, target_indices)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                scores = model(embeddings, target_indices)
                loss = compute_loss(scores, target_indices)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_losses.append(loss.item())

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        avg_loss = np.mean(train_losses) if train_losses else 0
        logger.info(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                   f"Val Tau: {val_metrics['kendall_tau']:.4f} | "
                   f"Val Acc: {val_metrics['accuracy']:.4f}")

        # Save best
        if val_metrics['kendall_tau'] > best_tau:
            best_tau = val_metrics['kendall_tau']
            torch.save(model.state_dict(), os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'best_transformer_seq2seq_no_pos.pt'))
            logger.info(f"  ✓ New best model: {best_tau:.4f}")

        # NO EARLY STOPPING - run all epochs

        scheduler.step()

    return model, best_tau


def evaluate_by_length(model, data_dir, device):
    """Evaluate on different length ranges"""
    ranges = [
        ("2-5", 2, 5),
        ("6-10", 6, 10),
        ("11-15", 11, 15),
        ("16-20", 16, 20),
        ("21-25", 21, 25)
    ]

    logger.info("\n" + "="*60)
    logger.info("EVALUATION BY DOCUMENT LENGTH (NO POSITIONAL ENCODINGS)")
    logger.info("="*60)

    for name, min_len, max_len in ranges:
        val_dataset = OrderingDataset(data_dir, min_len=min_len, max_len=max_len, split='val')

        if len(val_dataset) == 0:
            continue

        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        metrics = evaluate(model, val_loader, device)

        logger.info(f"{name:15s} ({min_len:2d}-{max_len:2d} pages): Tau = {metrics['kendall_tau']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Transformer Seq2Seq WITHOUT Positional Encodings")
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("ABLATION: Transformer WITHOUT Positional Encodings")
    logger.info("="*60)

    model, best_tau = train_model(
        args.data_dir,
        device=args.device,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )

    logger.info(f"\nBest validation Tau: {best_tau:.4f}")

    # Load best model
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'best_transformer_seq2seq_no_pos.pt')))

    # Evaluate by length
    evaluate_by_length(model, args.data_dir, args.device)


if __name__ == "__main__":
    main()
