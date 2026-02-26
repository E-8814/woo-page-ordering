"""
Greedy Nearest Neighbor Baseline for Document Page Ordering
===========================================================

Simple greedy heuristic baseline:
- Start with first page
- At each step, pick the most similar unselected page (by cosine similarity)
- Like solving TSP with greedy nearest neighbor

This is the "obvious" non-learning baseline that reviewers will expect.
"""

import os
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import json
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def greedy_nearest_neighbor(embeddings: np.ndarray) -> np.ndarray:
    """
    Greedy nearest neighbor baseline.

    Algorithm:
    1. Start with first page
    2. At each step, select unselected page with highest cosine similarity to current page
    3. Continue until all pages selected

    Args:
        embeddings: (num_pages, embedding_dim)
    Returns:
        order: (num_pages,) - ordering of pages
    """
    num_pages = len(embeddings)

    # Compute pairwise cosine similarities
    similarities = cosine_similarity(embeddings)

    # Start with the first page
    selected = [0]
    remaining = set(range(1, num_pages))

    # Greedily select most similar page at each step
    while remaining:
        current_page = selected[-1]

        # Find most similar unselected page
        best_similarity = -1
        best_page = None

        for page in remaining:
            sim = similarities[current_page, page]
            if sim > best_similarity:
                best_similarity = sim
                best_page = page

        selected.append(best_page)
        remaining.remove(best_page)

    return np.array(selected)




def evaluate_baseline(documents: List[Dict],
                      page_ranges: List[Tuple[int, int, str]]) -> Dict:
    """
    Evaluate greedy nearest neighbor baseline on different page ranges.

    Args:
        documents: List of documents with embeddings
        page_ranges: List of (min_pages, max_pages, label)
    Returns:
        results: Dict with tau scores by range
    """
    logger.info("\nEvaluating Greedy Nearest Neighbor...")
    logger.info("="*60)

    results = {}

    for min_pages, max_pages, label in page_ranges:
        # Filter documents by page range
        range_docs = [
            doc for doc in documents
            if min_pages <= len(doc['original_order']) <= max_pages
        ]

        if not range_docs:
            continue

        # Evaluate on all documents
        taus = []

        for doc in range_docs:
            embeddings = np.array(doc['page_embeddings'], dtype=np.float32)
            original_order = np.array(doc['original_order'], dtype=np.int64)

            # Shuffle the document
            num_pages = len(original_order)
            shuffle_indices = np.random.permutation(num_pages)
            shuffled_embeddings = embeddings[shuffle_indices]
            shuffled_order = original_order[shuffle_indices]

            # Predict ordering using greedy nearest neighbor
            predicted_order = greedy_nearest_neighbor(shuffled_embeddings)

            # Map predicted order back to original page numbers
            predicted_pages = shuffled_order[predicted_order]
            actual_pages = original_order

            # Calculate Kendall's tau
            tau, _ = kendalltau(predicted_pages, actual_pages)
            if not np.isnan(tau):
                taus.append(tau)

        avg_tau = np.mean(taus) if taus else 0.0

        results[label] = {
            'tau': avg_tau,
            'n_docs': len(range_docs)
        }

        # Format output like the example
        logger.info(f"{label:15} ({min_pages:2}-{max_pages:2} pages): Target Tau = {avg_tau:.4f}")

    # Calculate overall tau
    all_taus = []
    for label, metrics in results.items():
        all_taus.extend([metrics['tau']] * metrics['n_docs'])

    overall_tau = np.mean(all_taus) if all_taus else 0.0

    logger.info("="*60)
    logger.info(f"Overall Target Tau: {overall_tau:.4f}")

    return results


def load_documents(data_dir: str) -> List[Dict]:
    """Load documents from JSON files"""
    logger.info(f"Loading documents from {data_dir}...")
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*_embeddings.json"))

    documents = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            embeddings = np.array(data['embeddings'], dtype=np.float32)
            page_numbers = data['page_numbers']
            correct_order = data['correct_order']

            # Convert to format expected by baselines
            # original_order should be the correct ordering (0, 1, 2, ...)
            # page_embeddings are the embeddings in correct order
            documents.append({
                'doc_id': data['doc_id'],
                'page_embeddings': embeddings,
                'original_order': list(range(len(embeddings))),  # 0, 1, 2, ...
                'page_numbers': page_numbers
            })
        except Exception as e:
            continue

    logger.info(f"Loaded {len(documents)} documents")
    return documents


def main():
    """Main evaluation script"""

    # Configuration
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

    # Define page ranges
    PAGE_RANGES = [
        (2, 5, '2-5'),
        (6, 10, '6-10'),
        (11, 15, '11-15'),
        (16, 20, '16-20'),
        (21, 25, '21-25')
    ]

    # Load documents
    documents = load_documents(DATA_DIR)

    # Fix random seed for reproducibility
    np.random.seed(42)

    # Evaluate Greedy Nearest Neighbor
    logger.info("\n" + "="*60)
    logger.info("GREEDY NEAREST NEIGHBOR BASELINE")
    logger.info("="*60)
    results = evaluate_baseline(
        documents=documents,
        page_ranges=PAGE_RANGES
    )

    # Summary
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)

    for label, metrics in results.items():
        logger.info(f"  {label}: {metrics['tau']:.4f} (N={metrics['n_docs']})")


if __name__ == '__main__':
    main()
