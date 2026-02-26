"""
MLP-based Pointer Network for Document Page Ordering
====================================================

Pointer mechanism with simple feedforward networks (no LSTM, no attention).
Just MLPs for encoding and decoding, with a pointer mechanism for selection.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.stats import kendalltau
from typing import List, Tuple, Dict
import random
import json
import time
import math
from pathlib import Path


class MLPPointerNetwork(nn.Module):
    """
    Pointer network using MLPs instead of LSTM.

    - MLP encoder to process page embeddings
    - MLP decoder that processes previous selection
    - Pointer mechanism to select next page
    """
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Encoder: MLP to process each page embedding
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Decoder state tracker (processes previously selected page)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Pointer mechanism components
        self.W_query = nn.Linear(hidden_dim, hidden_dim)
        self.W_key = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def encode(self, embeddings):
        """
        Encode page embeddings using MLP.

        Args:
            embeddings: (batch_size, num_pages, embedding_dim)
        Returns:
            encoded: (batch_size, num_pages, hidden_dim)
        """
        batch_size, num_pages, _ = embeddings.shape

        # Process each page through encoder MLP
        encoded = self.encoder(embeddings)  # (batch, num_pages, hidden)

        return encoded

    def pointer_attention(self, query, keys, mask=None):
        """
        Compute pointer probabilities using attention mechanism.

        Args:
            query: (batch_size, hidden_dim) - decoder state
            keys: (batch_size, num_pages, hidden_dim) - encoded pages
            mask: (batch_size, num_pages) - 1 for available, 0 for used
        Returns:
            pointer_probs: (batch_size, num_pages)
        """
        # Transform query and keys
        query_transformed = self.W_query(query).unsqueeze(1)  # (batch, 1, hidden)
        keys_transformed = self.W_key(keys)  # (batch, num_pages, hidden)

        # Compute attention scores
        scores = self.v(torch.tanh(query_transformed + keys_transformed))  # (batch, num_pages, 1)
        scores = scores.squeeze(-1)  # (batch, num_pages)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Convert to probabilities
        pointer_probs = torch.softmax(scores, dim=-1)

        return pointer_probs

    def forward(self, embeddings, target_order=None):
        """
        Forward pass through pointer network.

        Args:
            embeddings: (batch_size, num_pages, embedding_dim)
            target_order: (batch_size, num_pages) - ground truth order for training
        Returns:
            all_pointer_probs: List of (batch_size, num_pages) for each step
            predicted_order: (batch_size, num_pages)
        """
        batch_size, num_pages, _ = embeddings.shape
        device = embeddings.device

        # Encode all pages
        encoded_pages = self.encode(embeddings)  # (batch, num_pages, hidden)

        # Initialize decoder state (start from mean of all pages)
        decoder_state = encoded_pages.mean(dim=1)  # (batch, hidden)

        # Track which positions have been selected
        mask = torch.ones(batch_size, num_pages, device=device)

        all_pointer_probs = []
        predicted_order = []

        # Decode step by step
        for step in range(num_pages):
            # Get pointer probabilities for this step
            pointer_probs = self.pointer_attention(decoder_state, encoded_pages, mask)
            all_pointer_probs.append(pointer_probs)

            # Select page
            if target_order is not None and self.training:
                # Teacher forcing during training
                selected_idx = target_order[:, step]
            else:
                # Greedy selection during inference
                selected_idx = torch.argmax(pointer_probs, dim=-1)

            predicted_order.append(selected_idx)

            # Update mask
            mask = mask.scatter(1, selected_idx.unsqueeze(1), 0)

            # Update decoder state with selected page
            selected_pages = torch.gather(
                encoded_pages,
                1,
                selected_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.hidden_dim)
            ).squeeze(1)  # (batch, hidden)

            decoder_state = self.decoder(selected_pages)

        predicted_order = torch.stack(predicted_order, dim=1)

        return all_pointer_probs, predicted_order


class DocumentDataset(Dataset):
    """Dataset of shuffled documents"""
    def __init__(self, documents: List[Dict]):
        self.documents = documents

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        doc = self.documents[idx]

        # Get original order and embeddings
        original_order = torch.tensor(doc['original_order'], dtype=torch.long)
        embeddings = torch.tensor(doc['page_embeddings'], dtype=torch.float32)

        # Apply a fresh random shuffle each time
        num_pages = len(original_order)
        shuffle_indices = torch.randperm(num_pages)

        shuffled_embeddings = embeddings[shuffle_indices]
        shuffled_order = original_order[shuffle_indices]

        # Target order: argsort to get positions that would restore correct order
        target_order = torch.argsort(shuffled_order)

        return shuffled_embeddings, target_order, doc['doc_id']


class SubsetDataset(Dataset):
    """Wrapper to create train/val/test splits with dynamic shuffling"""
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # CRITICAL: Call base_dataset's __getitem__ to get fresh shuffle
        return self.base_dataset[self.indices[idx]]


def collate_fn(batch):
    """Collate function for variable-length documents"""
    # For batch_size=1, just return the single item properly formatted
    if len(batch) == 1:
        embeddings, targets, doc_ids = batch[0]
        return embeddings.unsqueeze(0), targets.unsqueeze(0), [doc_ids]

    # For larger batches, documents must have same length
    embeddings, targets, doc_ids = zip(*batch)

    # Check if all have same size
    sizes = [e.shape[0] for e in embeddings]
    if len(set(sizes)) > 1:
        raise ValueError(f"Cannot batch documents with different lengths: {sizes}")

    embeddings = torch.stack(embeddings)
    targets = torch.stack(targets)

    return embeddings, targets, doc_ids


class SingleMLPPointer:
    """
    Single MLP pointer network for all page ranges.
    """
    def __init__(self, embedding_dim: int = 768, device: str = 'cuda'):
        self.device = device
        self.embedding_dim = embedding_dim

        # Single general model for all document sizes
        self.model = MLPPointerNetwork(
            embedding_dim=embedding_dim,
            hidden_dim=512,
            dropout=0.15
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Store held-out test set for final evaluation
        self.test_set = []

    def train_model(self, train_loader: DataLoader,
                   val_loader: DataLoader, num_epochs: int = 50):
        """Train the model"""
        model = self.model
        optimizer = self.optimizer

        best_val_tau = -1.0
        patience = 10
        patience_counter = 0

        print(f"\nTraining general model...")

        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_taus = []

            for embeddings, targets, _ in train_loader:
                embeddings = embeddings.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                pointer_probs, predicted_order = model(embeddings, targets)

                # Compute cross-entropy loss for each decoding step
                loss = 0
                for step, probs in enumerate(pointer_probs):
                    target_indices = targets[:, step]
                    loss += nn.functional.cross_entropy(
                        torch.log(probs + 1e-10),
                        target_indices
                    )

                loss = loss / len(pointer_probs)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                train_loss += loss.item()

                # Calculate Kendall's tau for training batch
                for pred, tgt in zip(predicted_order, targets):
                    tau, _ = kendalltau(pred.cpu().numpy(), tgt.cpu().numpy())
                    train_taus.append(tau if not np.isnan(tau) else 0.0)

            avg_train_loss = train_loss / len(train_loader)
            avg_train_tau = np.mean(train_taus)

            # Validation
            model.eval()
            val_taus = []

            with torch.no_grad():
                for embeddings, targets, _ in val_loader:
                    embeddings = embeddings.to(self.device)
                    targets = targets.to(self.device)

                    _, predicted_order = model(embeddings)

                    for pred, tgt in zip(predicted_order, targets):
                        tau, _ = kendalltau(pred.cpu().numpy(), tgt.cpu().numpy())
                        val_taus.append(tau if not np.isnan(tau) else 0.0)

            avg_val_tau = np.mean(val_taus)

            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, Train Tau: {avg_train_tau:.4f}, "
                  f"Val Tau: {avg_val_tau:.4f}")

            # Early stopping based on validation tau
            if avg_val_tau > best_val_tau:
                best_val_tau = avg_val_tau
                patience_counter = 0
                # Save best model checkpoint
                model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlp_pointer_general_best.pt')
                torch.save(model.state_dict(), model_save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlp_pointer_general_best.pt')
        model.load_state_dict(torch.load(model_save_path))
        print(f"Best validation tau: {best_val_tau:.4f}")

        return best_val_tau

    def predict(self, embeddings: torch.Tensor):
        """Predict ordering for a document"""
        self.model.eval()
        with torch.no_grad():
            embeddings = embeddings.unsqueeze(0).to(self.device)
            _, predicted_order = self.model(embeddings)
            return predicted_order.squeeze(0).cpu().numpy()


def load_documents(data_dir: str) -> List[Dict]:
    """Load documents from JSON files"""
    print(f"Loading documents from {data_dir}...")
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*_embeddings.json"))

    documents = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            embeddings = np.array(data['embeddings'], dtype=np.float32)
            correct_order = data['correct_order']

            # Convert to expected format
            documents.append({
                'doc_id': data['doc_id'],
                'page_embeddings': embeddings,
                'original_order': list(range(len(embeddings))),
                'num_pages': len(embeddings)
            })
        except Exception as e:
            continue

    print(f"Loaded {len(documents)} documents")
    return documents


def train_single_model(data_dir: str,
                       device: str = 'cuda',
                       num_epochs: int = 50,
                       batch_size: int = 32,
                       embedding_dim: int = 3072,
                       all_documents: List[Dict] = None):
    """Train a single general model on all documents"""
    # Load all documents if not provided
    if all_documents is None:
        all_documents = load_documents(data_dir)

    # Filter to documents in the 2-25 page range
    filtered_docs = [
        doc for doc in all_documents
        if 2 <= len(doc['original_order']) <= 25
    ]

    print(f"\n{'='*80}")
    print(f"Training single general model on all documents (2-25 pages)")
    print(f"{'='*80}")
    print(f"Total documents: {len(filtered_docs)}")

    # Create single model with correct embedding dimension
    model_wrapper = SingleMLPPointer(embedding_dim=embedding_dim, device=device)

    # Create dataset
    dataset = DocumentDataset(filtered_docs)

    # 70/15/15 split for train/val/test
    random.seed(42)
    total_size = len(dataset)
    all_indices = list(range(total_size))
    random.shuffle(all_indices)

    train_size = int(0.70 * total_size)
    train_size = min(train_size, 1000)  # Cap at 1000 for memory
    val_size = int(0.15 * total_size)
    val_size = min(val_size, 200)
    test_size = total_size - train_size - val_size

    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]

    print(f"Split: {train_size} train, {val_size} val, {test_size} test")

    # Store held-out test set for later evaluation
    model_wrapper.test_set = [filtered_docs[idx] for idx in test_indices]

    # Create subset datasets with dynamic shuffling
    train_dataset = SubsetDataset(dataset, train_indices)
    val_dataset = SubsetDataset(dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Disable multiprocessing for batch_size=1
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0  # Disable multiprocessing for batch_size=1
    )

    # Train the model
    best_tau = model_wrapper.train_model(train_loader, val_loader, num_epochs)

    return model_wrapper


def evaluate_single_model_on_held_out_test(model_wrapper: SingleMLPPointer,
                                            page_ranges: List[Tuple[int, int, str]]) -> Dict:
    """Evaluate single model on held-out test set across different page ranges"""
    from datetime import datetime

    print("\n" + "="*80)
    print("EVALUATING ON HELD-OUT TEST SET")
    print("="*80)

    all_test_docs = model_wrapper.test_set
    print(f"Total held-out test documents: {len(all_test_docs)}")

    # Evaluate by page range
    results_by_range = {}

    for min_pages, max_pages, model_name in page_ranges:
        # Filter test docs for this range
        range_docs = [
            doc for doc in all_test_docs
            if min_pages <= len(doc['original_order']) <= max_pages
        ]

        if not range_docs:
            continue

        taus = []
        for doc in range_docs:
            embeddings = torch.tensor(doc['page_embeddings'], dtype=torch.float32)
            num_pages = len(doc['original_order'])

            # Get shuffled version
            shuffle_indices = torch.randperm(num_pages)
            shuffled_embeddings = embeddings[shuffle_indices]
            original_order = torch.tensor(doc['original_order'], dtype=torch.long)
            shuffled_order = original_order[shuffle_indices]

            # Predict
            predicted_order = model_wrapper.predict(shuffled_embeddings)

            # Map predicted positions back to original page numbers
            predicted_pages = shuffled_order[predicted_order].numpy()
            actual_pages = np.array(doc['original_order'])

            # Calculate Kendall's tau
            tau, _ = kendalltau(predicted_pages, actual_pages)
            if not np.isnan(tau):
                taus.append(tau)

        avg_tau = np.mean(taus) if taus else 0.0
        results_by_range[f"{min_pages}-{max_pages}"] = {
            'tau': avg_tau,
            'n_docs': len(range_docs)
        }

    # Overall result
    all_taus = []
    for doc in all_test_docs:
        embeddings = torch.tensor(doc['page_embeddings'], dtype=torch.float32)
        num_pages = len(doc['original_order'])

        shuffle_indices = torch.randperm(num_pages)
        shuffled_embeddings = embeddings[shuffle_indices]
        original_order = torch.tensor(doc['original_order'], dtype=torch.long)
        shuffled_order = original_order[shuffle_indices]

        predicted_order = model_wrapper.predict(shuffled_embeddings)
        predicted_pages = shuffled_order[predicted_order].numpy()
        actual_pages = np.array(doc['original_order'])

        tau, _ = kendalltau(predicted_pages, actual_pages)
        if not np.isnan(tau):
            all_taus.append(tau)

    overall_tau = np.mean(all_taus) if all_taus else 0.0
    print(f"\n{'='*80}")
    print(f"OVERALL HELD-OUT TEST TAU: {overall_tau:.4f} (N={len(all_test_docs)})")
    print(f"{'='*80}")

    # Print results in the desired format matching results_baseline.txt
    print("\n" + "="*80)
    print("RESULTS IN STANDARD FORMAT")
    print("="*80)
    print("baseline_pointer_mlp:")
    for min_pages, max_pages, model_name in page_ranges:
        range_key = f"{min_pages}-{max_pages}"
        if range_key in results_by_range:
            tau = results_by_range[range_key]['tau']
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Format: model_name with proper spacing
            label = f"{min_pages}-{max_pages}"
            spacing = " " * (16 - len(label))
            print(f"{timestamp} | INFO | {label}{spacing}({min_pages:2d}-{max_pages:2d} pages): Target Tau = {tau:.4f}")

    return {
        'overall': overall_tau,
        'by_range': results_by_range,
        'n_total': len(all_test_docs)
    }


def main():
    """Main training and evaluation script"""
    import os
    os.makedirs('models', exist_ok=True)

    # Configuration
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCHS = 50
    BATCH_SIZE = 1  # Use batch_size=1 due to variable-length documents

    # Define page ranges for evaluation
    PAGE_RANGES = [
        (2, 5, 'tiny_2_5'),
        (6, 10, 'small_6_10'),
        (11, 15, 'medium_11_15'),
        (16, 20, 'large_16_20'),
        (21, 25, 'xlarge_21_25')
    ]

    print(f"Using device: {DEVICE}")
    print(f"Training single MLP pointer network...")

    # Load documents first to get embedding dimension
    all_docs = load_documents(DATA_DIR)
    if len(all_docs) == 0:
        print("Error: No documents loaded!")
        return

    embedding_dim = all_docs[0]['page_embeddings'].shape[1]
    print(f"Detected embedding dimension: {embedding_dim}")

    # Train single model
    start_time = time.time()
    model_wrapper = train_single_model(
        data_dir=DATA_DIR,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        embedding_dim=embedding_dim,
        all_documents=all_docs
    )

    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time/60:.2f} minutes")

    # Evaluate on held-out test set across all page ranges
    results = evaluate_single_model_on_held_out_test(model_wrapper, PAGE_RANGES)

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Overall Test Tau: {results['overall']:.4f}")
    print("\nBy page range:")
    for range_name, metrics in results['by_range'].items():
        print(f"  {range_name}: {metrics['tau']:.4f} (N={metrics['n_docs']})")


if __name__ == '__main__':
    main()
