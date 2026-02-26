#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
"""
BASELINE: Simple Similarity-Based Document Ordering
Uses cosine similarity between embeddings to order pages.
No neural networks, no training - just similarity metrics.
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import random
from typing import List, Dict, Tuple
from scipy.stats import kendalltau
import argparse
import logging
from collections import defaultdict
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SimpleOrderingBaseline:
    """Multiple baseline methods for document ordering"""

    def __init__(self, method='sequential'):
        """
        method: 'sequential', 'centroid', 'tsp', 'spectral'
        """
        self.method = method

    def order_by_sequential_similarity(self, embeddings):
        """
        Greedy approach: Start with most "introduction-like" page,
        then repeatedly add the most similar unvisited page.
        """
        n = len(embeddings)
        if n <= 1:
            return list(range(n))

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Find starting page (lowest norm often = introduction)
        norms = torch.norm(embeddings, dim=1)
        current = torch.argmin(norms).item()

        ordered = [current]
        remaining = set(range(n)) - {current}

        # Greedily add most similar pages
        while remaining:
            similarities = torch.matmul(embeddings[current:current+1], embeddings.t()).squeeze()

            # Mask out already selected pages
            for idx in ordered:
                similarities[idx] = -1

            next_page = torch.argmax(similarities).item()
            ordered.append(next_page)
            remaining.remove(next_page)
            current = next_page

        return ordered

    def order_by_centroid_distance(self, embeddings):
        """
        Order pages by their distance to document centroid.
        Theory: Introduction is far from center, conclusion too,
        middle pages are closer to center.
        """
        n = len(embeddings)
        if n <= 1:
            return list(range(n))

        # Compute document centroid
        centroid = embeddings.mean(dim=0, keepdim=True)

        # Compute distances to centroid
        distances = torch.cdist(embeddings, centroid).squeeze()

        # Sort by distance (furthest first, assuming intro/conclusion are outliers)
        ordered = torch.argsort(distances, descending=True).tolist()

        # Heuristic: First page should be the one with lowest norm (often intro)
        norms = torch.norm(embeddings, dim=1)
        intro_candidate = torch.argmin(norms).item()

        # Move intro candidate to beginning if not already there
        if intro_candidate in ordered:
            ordered.remove(intro_candidate)
            ordered.insert(0, intro_candidate)

        return ordered

    def order_by_tsp(self, embeddings):
        """
        Treat as traveling salesman problem:
        Find path through all pages minimizing total distance.
        """
        n = len(embeddings)
        if n <= 1:
            return list(range(n))

        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings)

        # Nearest neighbor heuristic for TSP
        norms = torch.norm(embeddings, dim=1)
        current = torch.argmin(norms).item()  # Start with "intro"

        ordered = [current]
        unvisited = set(range(n)) - {current}

        while unvisited:
            # Find nearest unvisited page
            dists = distances[current].clone()
            for visited in ordered:
                dists[visited] = float('inf')

            nearest = torch.argmin(dists).item()
            ordered.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        return ordered

    def order_by_spectral(self, embeddings):
        """
        Use spectral ordering based on similarity matrix.
        Compute eigenvector of similarity matrix and sort by it.
        """
        n = len(embeddings)
        if n <= 1:
            return list(range(n))

        # Normalize and compute similarity matrix
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        similarity = torch.matmul(embeddings_norm, embeddings_norm.t())

        # Make it a proper affinity matrix
        similarity = (similarity + 1) / 2  # Scale to [0, 1]
        similarity.fill_diagonal_(0)  # No self-connections

        # Compute eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(similarity)

        # Use second smallest eigenvector (Fiedler vector) for ordering
        fiedler = eigenvectors[:, 1]

        # Sort by Fiedler vector values
        ordered = torch.argsort(fiedler).tolist()

        return ordered

    def order_by_sliding_window(self, embeddings, window_size=3):
        """
        Advanced: Use sliding window correlations.
        Assumes adjacent pages have higher similarity than distant pages.
        """
        n = len(embeddings)
        if n <= 1:
            return list(range(n))

        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Build scoring matrix based on window similarities
        scores = torch.zeros(n)

        for i in range(n):
            # Compute average similarity to potential neighbors
            window_sim = 0
            count = 0
            for j in range(max(0, i-window_size), min(n, i+window_size+1)):
                if i != j:
                    window_sim += torch.matmul(embeddings[i:i+1], embeddings[j:j+1].t()).item()
                    count += 1
            if count > 0:
                scores[i] = window_sim / count

        # Order by scores (pages with high local similarity go in middle)
        ordered = torch.argsort(scores).tolist()

        # Ensure first page is most "intro-like"
        norms = torch.norm(embeddings, dim=1)
        intro_candidate = torch.argmin(norms).item()
        if intro_candidate in ordered:
            ordered.remove(intro_candidate)
            ordered.insert(0, intro_candidate)

        return ordered

    def predict_order(self, embeddings):
        """Main entry point for ordering"""
        if self.method == 'sequential':
            return self.order_by_sequential_similarity(embeddings)
        elif self.method == 'centroid':
            return self.order_by_centroid_distance(embeddings)
        elif self.method == 'tsp':
            return self.order_by_tsp(embeddings)
        elif self.method == 'spectral':
            return self.order_by_spectral(embeddings)
        elif self.method == 'sliding_window':
            return self.order_by_sliding_window(embeddings)
        else:
            raise ValueError(f"Unknown method: {self.method}")


def compute_metrics(predicted_order, correct_order, page_numbers):
    """Compute accuracy and Kendall's tau"""
    # Map predicted indices to page numbers
    predicted_pages = [page_numbers[idx].item() for idx in predicted_order]

    # Calculate accuracy
    correct = sum(1 for pred, true in zip(predicted_pages, correct_order.numpy())
                  if pred == true)
    accuracy = correct / len(predicted_pages) if predicted_pages else 0

    # Kendall's tau
    true_positions = {page: i for i, page in enumerate(correct_order.numpy())}
    pred_positions = {page: i for i, page in enumerate(predicted_pages)}

    common_pages = set(true_positions.keys()) & set(pred_positions.keys())
    if len(common_pages) > 1:
        true_pos = [true_positions[p] for p in common_pages]
        pred_pos = [pred_positions[p] for p in common_pages]
        tau, _ = kendalltau(true_pos, pred_pos)
    else:
        tau = 0.0

    return {'accuracy': accuracy, 'kendall_tau': tau}


def evaluate_baselines(data_dir: str, methods: List[str] = None):
    """Evaluate all baseline methods on the dataset"""

    if methods is None:
        methods = ['sequential', 'centroid', 'tsp', 'spectral', 'sliding_window']

    data_path = Path(data_dir)
    embedding_files = list(data_path.glob("*_embeddings.json"))

    logger.info(f"Found {len(embedding_files)} documents")

    # Group documents by length
    docs_by_length = defaultdict(list)

    for file_path in embedding_files:
        try:
            with open(file_path, 'r') as f:
                doc_data = json.load(f)

            if 'correct_order' not in doc_data:
                continue

            doc_length = len(doc_data['embeddings'])
            docs_by_length[doc_length].append({
                'embeddings': torch.tensor(doc_data['embeddings'], dtype=torch.float32),
                'page_numbers': torch.tensor(doc_data['page_numbers'], dtype=torch.long),
                'correct_order': torch.tensor(doc_data['correct_order'], dtype=torch.long),
                'doc_id': file_path.stem.replace('_embeddings', ''),
                'length': doc_length
            })
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")

    # Evaluate each method
    results = defaultdict(lambda: defaultdict(list))

    for method in methods:
        logger.info(f"\nEvaluating method: {method}")
        baseline = SimpleOrderingBaseline(method=method)

        for length, docs in docs_by_length.items():
            if length < 2:
                continue

            method_metrics = []

            for doc in docs:
                # Shuffle pages to simulate test condition
                n = doc['length']
                perm = torch.randperm(n)
                shuffled_embeddings = doc['embeddings'][perm]
                shuffled_page_numbers = doc['page_numbers'][perm]

                # Predict order
                start_time = time.time()
                predicted_order = baseline.predict_order(shuffled_embeddings)
                pred_time = time.time() - start_time

                # Compute metrics
                metrics = compute_metrics(
                    predicted_order,
                    doc['correct_order'],
                    shuffled_page_numbers
                )
                metrics['time'] = pred_time
                method_metrics.append(metrics)

            if method_metrics:
                avg_tau = np.mean([m['kendall_tau'] for m in method_metrics])
                avg_acc = np.mean([m['accuracy'] for m in method_metrics])
                avg_time = np.mean([m['time'] for m in method_metrics])

                results[method][length] = {
                    'tau': avg_tau,
                    'accuracy': avg_acc,
                    'time_ms': avg_time * 1000,
                    'n_docs': len(method_metrics)
                }

    return results


def print_results_table(results):
    """Print results in a nice table format"""

    # Define length ranges for summary
    ranges = [
        ("2-5", (2, 5)),
        ("6-10", (6, 10)),
        ("11-15", (11, 15)),
        ("16-20", (16, 20)),
        ("21-25", (21, 25)),
        ("26-30", (26, 30)),
        ("31+", (31, 999))
    ]

    print("\n" + "="*90)
    print("BASELINE RESULTS - Kendall's Tau by Document Length Range")
    print("="*90)

    # Header
    print(f"{'Method':<20}", end="")
    for range_name, _ in ranges:
        print(f"{range_name:>10}", end="")
    print(f"{'Overall':>10}")
    print("-"*90)

    # Results for each method
    for method in results.keys():
        print(f"{method:<20}", end="")

        all_taus = []
        all_counts = []

        for range_name, (min_len, max_len) in ranges:
            range_taus = []
            range_counts = 0

            for length, metrics in results[method].items():
                if min_len <= length <= max_len:
                    tau = metrics['tau']
                    n_docs = metrics['n_docs']
                    range_taus.extend([tau] * n_docs)
                    range_counts += n_docs
                    all_taus.extend([tau] * n_docs)
                    all_counts.append(n_docs)

            if range_taus:
                avg_tau = np.mean(range_taus)
                print(f"{avg_tau:>10.3f}", end="")
            else:
                print(f"{'---':>10}", end="")

        # Overall average
        if all_taus:
            overall_tau = np.mean(all_taus)
            print(f"{overall_tau:>10.3f}")
        else:
            print(f"{'---':>10}")

    print("="*90)

    # Print detailed results by exact length
    print("\nDetailed Results by Exact Document Length:")
    print("-"*90)

    for method in results.keys():
        print(f"\n{method.upper()} Method:")
        print(f"{'Length':>8} {'Docs':>8} {'Tau':>10} {'Accuracy':>10} {'Time(ms)':>10}")
        print("-"*50)

        for length in sorted(results[method].keys()):
            metrics = results[method][length]
            print(f"{length:>8} {metrics['n_docs']:>8} {metrics['tau']:>10.3f} "
                  f"{metrics['accuracy']:>10.3f} {metrics['time_ms']:>10.1f}")


def main():
    parser = argparse.ArgumentParser(description="Baseline methods for document ordering")
    parser.add_argument("--data_dir", type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
                       help="Directory containing embedding files")
    parser.add_argument("--methods", nargs='+',
                       choices=['sequential', 'centroid', 'tsp', 'spectral',
                               'sliding_window', 'all'],
                       default=['all'],
                       help="Which methods to evaluate")
    parser.add_argument("--output_file", type=str,
                       default="baseline_results.json",
                       help="File to save results")

    args = parser.parse_args()

    # Determine methods to evaluate
    if 'all' in args.methods:
        methods = ['sequential', 'centroid', 'tsp', 'spectral', 'sliding_window']
    else:
        methods = args.methods

    logger.info(f"Evaluating methods: {methods}")
    logger.info(f"Data directory: {args.data_dir}")

    # Run evaluation
    results = evaluate_baselines(args.data_dir, methods)

    # Print results
    print_results_table(results)

    # Save results to JSON
    output_path = Path(args.output_file)

    # Convert to serializable format
    serializable_results = {}
    for method, method_results in results.items():
        serializable_results[method] = {}
        for length, metrics in method_results.items():
            serializable_results[method][str(length)] = metrics

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    # Print comparison with your model (if you have results)
    print("\n" + "="*90)
    print("COMPARISON NOTES:")
    print("-"*90)
    print("Your transformer model should significantly outperform these baselines.")
    print("If any baseline gets Tau > 0.7 on longer documents, it suggests:")
    print("  1. Your embeddings already encode strong sequential information")
    print("  2. Simple methods might be sufficient for this task")
    print("\nIf your model gets Tau > 0.9 while baselines get < 0.5:")
    print("  → Strong evidence that your approach adds value!")
    print("="*90)


if __name__ == "__main__":
    main()