#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
"""
RANDOM BASELINE
Orders documents randomly (sanity check)
"""

import json
import random
import numpy as np
from pathlib import Path
from scipy.stats import kendalltau
from typing import Tuple, Dict


def random_ordering(embeddings: np.ndarray) -> np.ndarray:
    """
    Randomly shuffle document order.
    
    Args:
        embeddings: [n_docs, embedding_dim] - not actually used
        
    Returns:
        predicted_order: [n_docs] - random permutation
    """
    n = len(embeddings)
    order = list(range(n))
    random.shuffle(order)
    return np.array(order)


def evaluate_document(doc_data: dict) -> Dict[str, float]:
    """Evaluate random ordering on a single document compilation."""
    embeddings = np.array(doc_data['embeddings'])
    correct_order = doc_data['correct_order']
    page_numbers = doc_data['page_numbers']
    
    # Get random prediction
    predicted_order = random_ordering(embeddings)
    
    # Convert to actual page numbers
    predicted_pages = [page_numbers[idx] for idx in predicted_order]
    
    # Compute metrics
    # Accuracy
    correct = sum(1 for p, t in zip(predicted_pages, correct_order) if p == t)
    accuracy = correct / len(predicted_pages)
    
    # Kendall's tau
    tau, _ = kendalltau(correct_order, predicted_pages)
    
    return {
        'accuracy': accuracy,
        'kendall_tau': tau,
        'length': len(embeddings)
    }


def evaluate_on_range(data_dir: str, min_docs: int, max_docs: int, 
                     split: str = 'val', seed: int = 42) -> Dict[str, float]:
    """
    Evaluate random baseline on documents of specific length range.
    
    Args:
        data_dir: Path to directory with embedding files
        min_docs: Minimum number of documents in compilation
        max_docs: Maximum number of documents in compilation
        split: 'train' or 'val'
        seed: Random seed
        
    Returns:
        Dictionary with average metrics
    """
    random.seed(seed)
    np.random.seed(seed)
    
    data_dir = Path(data_dir)
    embedding_files = list(data_dir.glob("*_embeddings.json"))

    print(f"  Found {len(embedding_files)} total embedding files")
    print(f"  Scanning files for range {min_docs}-{max_docs}...")

    # First pass: collect file paths that match our criteria (memory efficient)
    # We still need to scan files to check document length, but we don't keep them in memory
    valid_file_paths = []
    for idx, file_path in enumerate(embedding_files):
        try:
            # Show progress every 1000 files
            if (idx + 1) % 1000 == 0:
                print(f"    Scanned {idx+1}/{len(embedding_files)} files... (found {len(valid_file_paths)} matches)", end='\r', flush=True)

            with open(file_path, 'r') as f:
                doc_data = json.load(f)

            if 'correct_order' not in doc_data:
                continue

            doc_length = len(doc_data['embeddings'])

            if min_docs <= doc_length <= max_docs:
                valid_file_paths.append(file_path)
        except Exception as e:
            continue

    print(f"\n    Found {len(valid_file_paths)} documents in range {min_docs}-{max_docs}")

    # Split train/val on file paths
    random.shuffle(valid_file_paths)
    n_train = int(len(valid_file_paths) * 0.85)
    
    if split == 'train':
        selected_paths = valid_file_paths[:n_train]
    else:
        selected_paths = valid_file_paths[n_train:]

    if len(selected_paths) == 0:
        return {'tau': -1, 'acc': -1, 'count': 0}

    # Evaluate with progress (load files one at a time)
    results = []
    total_docs = len(selected_paths)
    for idx, file_path in enumerate(selected_paths):
        # Load document data on demand
        with open(file_path, 'r') as f:
            doc_data = json.load(f)

        metrics = evaluate_document(doc_data)
        results.append(metrics)
        
        # Show progress every 20 documents or at end (faster than others)
        if (idx + 1) % 20 == 0 or (idx + 1) == total_docs:
            avg_tau_so_far = np.mean([r['kendall_tau'] for r in results])
            print(f"    Progress: {idx+1}/{total_docs} docs | Avg Tau: {avg_tau_so_far:.3f}", 
                  end='\r', flush=True)
    
    if total_docs > 0:
        print()  # New line after progress
    
    # Aggregate
    avg_tau = np.mean([r['kendall_tau'] for r in results])
    avg_acc = np.mean([r['accuracy'] for r in results])
    
    return {
        'tau': avg_tau,
        'acc': avg_acc,
        'count': len(results)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Random Baseline")
    parser.add_argument("--data_dir", type=str, 
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
                       help="Directory with embedding files")
    args = parser.parse_args()
    
    print("=" * 80)
    print("RANDOM BASELINE EVALUATION")
    print("=" * 80)
    
    # Evaluation ranges
    ranges = [
        ("2-5", 2, 5),
        ("6-10", 6, 10),
        ("11-15", 11, 15),
        ("16-20", 16, 20),
        ("21-25", 21, 25),
    ]
    
    # Collect results for visualization
    all_results = []
    for range_name, min_docs, max_docs in ranges:
        results = evaluate_on_range(args.data_dir, min_docs, max_docs)
        all_results.append((range_name, results))
        
        if results['count'] > 0:
            print(f"{range_name:8s} | Tau: {results['tau']:6.3f} | "
                  f"Acc: {results['acc']:5.3f} | N: {results['count']:4d}")
        else:
            print(f"{range_name:8s} | No documents in range")
    
    print("=" * 80)
    
    # Create beautiful visualization
    print("\n📊 Creating beautiful visualization...")
    import matplotlib.pyplot as plt
    import numpy as np
    
    range_names = [r for r, _ in all_results]
    tau_values = [res['tau'] if res['count'] > 0 else 0 for _, res in all_results]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create gradient colors
    colors = ['#e74c3c', '#e67e73', '#ea9999', '#f4b4b4', '#f9cece']
    bars = ax.bar(range_names, tau_values, color=colors, alpha=0.85, 
                  edgecolor='#c0392b', linewidth=2.5)
    
    # Add value labels with styling
    for bar, tau in zip(bars, tau_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{tau:.3f}', ha='center', va='bottom', 
                fontsize=13, fontweight='bold', color='#c0392b')
    
    ax.set_xlabel('Document Range', fontweight='bold', fontsize=15)
    ax.set_ylabel("Kendall's Tau Correlation", fontweight='bold', fontsize=15)
    ax.set_title('Random Baseline - Sanity Check Performance', 
                fontweight='bold', fontsize=18, pad=20, color='#2c3e50')
    ax.set_ylim(-0.2, 1.0)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.6, linewidth=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    
    # Add explanation text
    ax.text(0.98, 0.95, 'Expected: ~0% correlation\n(Pure randomness)', 
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           horizontalalignment='right', style='italic',
           bbox=dict(boxstyle='round', facecolor='#fff5f5', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('random_baseline_results.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Beautiful visualization saved to: random_baseline_results.png")
    plt.close()
    
    # Save results to JSON for later use
    results_dict = {r[0]: res['tau'] for r, res in all_results if res['count'] > 0}
    with open('random_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    print("💾 Results saved to: random_results.json")

