#!/usr/bin/env python
import os
"""
COMPREHENSIVE POSITIONAL ENCODING COMPARISON
==============================================

Compares THREE positional encoding strategies:
1. LEARNED (original) - vulnerable to training imbalance
2. SINUSOIDAL (fixed) - standard solution
3. NONE - no positional information

Run this AFTER training all three models.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))
from seq2seq_transformer import TransformerSeq2Seq, OrderingDataset
from seq2seq_transformer_no_pos import TransformerSeq2SeqNoPosEncoding
from seq2seq_transformer_sinusoidal import TransformerSeq2SeqSinusoidal

sns.set_style("whitegrid")


def load_all_models():
    """Load all three model variants"""
    print("Loading models...")

    models = {}

    # 1. LEARNED positional encodings (original)
    try:
        model_learned = TransformerSeq2Seq(
            embedding_dim=3072, hidden_dim=512, num_layers=6,
            num_heads=8, dropout=0.1
        )
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'best_transformer_seq2seq.pt')
        model_learned.load_state_dict(torch.load(model_path, map_location='cpu'))
        model_learned.eval()
        models['Learned'] = model_learned
        print("✓ Learned model loaded")
    except FileNotFoundError:
        print("⚠ Learned model not found")

    # 2. SINUSOIDAL positional encodings
    try:
        model_sin = TransformerSeq2SeqSinusoidal(
            embedding_dim=3072, hidden_dim=512, num_layers=6,
            num_heads=8, dropout=0.1
        )
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'best_transformer_seq2seq_sinusoidal.pt')
        model_sin.load_state_dict(torch.load(model_path, map_location='cpu'))
        model_sin.eval()
        models['Sinusoidal'] = model_sin
        print("✓ Sinusoidal model loaded")
    except FileNotFoundError:
        print("⚠ Sinusoidal model not found. Train with:")
        print("  python transformer_seq2seq_sinusoidal.py")

    # 3. NO positional encodings
    try:
        model_nopos = TransformerSeq2SeqNoPosEncoding(
            embedding_dim=3072, hidden_dim=512, num_layers=6,
            num_heads=8, dropout=0.1
        )
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'best_transformer_seq2seq_no_pos.pt')
        model_nopos.load_state_dict(torch.load(model_path, map_location='cpu'))
        model_nopos.eval()
        models['No Position'] = model_nopos
        print("✓ No-pos model loaded")
    except FileNotFoundError:
        print("⚠ No-pos model not found. Train with:")
        print("  python transformer_seq2seq_no_pos.py")

    if len(models) < 2:
        print("\n❌ Need at least 2 models to compare. Train missing models first.")
        return None

    return models


def compute_metrics(predicted_order, correct_order, page_numbers):
    """Compute Kendall's tau"""
    predicted_pages = [page_numbers[idx].item() for idx in predicted_order]

    true_positions = {page: i for i, page in enumerate(correct_order.cpu().numpy())}
    pred_positions = {page: i for i, page in enumerate(predicted_pages)}

    common_pages = set(true_positions.keys()) & set(pred_positions.keys())
    if len(common_pages) > 1:
        true_pos = [true_positions[p] for p in common_pages]
        pred_pos = [pred_positions[p] for p in common_pages]
        tau, _ = kendalltau(true_pos, pred_pos)
    else:
        tau = 0.0

    return tau


def evaluate_model(model, data_dir, device='cpu'):
    """Evaluate model on different length ranges"""
    results = {}

    for min_len, max_len in [(2, 5), (6, 10), (11, 15), (16, 20), (21, 25)]:
        dataset = OrderingDataset(data_dir, min_len=min_len, max_len=max_len, split='val')

        taus = []
        for i in range(min(50, len(dataset))):
            doc = dataset[i]
            embeddings = doc['embeddings'].unsqueeze(0).to(device)
            page_numbers = doc['page_numbers']
            correct_order = doc['correct_order']

            with torch.no_grad():
                predicted_indices = model(embeddings).squeeze(0).cpu()

            tau = compute_metrics(predicted_indices, correct_order, page_numbers)
            taus.append(tau)

        results[f"{min_len}-{max_len}"] = {
            'mean': np.mean(taus),
            'std': np.std(taus),
            'all': taus
        }

    return results


def create_comparison_plots(all_results, output_dir):
    """Create comprehensive comparison plots"""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    ranges = ["2-5", "6-10", "11-15", "16-20", "21-25"]
    x_pos = np.arange(len(ranges))

    # PLOT 1: Side-by-side bars
    fig, ax = plt.subplots(figsize=(12, 6))

    width = 0.25
    colors = {'Learned': 'steelblue', 'Sinusoidal': 'coral', 'No Position': 'mediumseagreen'}

    for i, (name, results) in enumerate(all_results.items()):
        means = [results[r]['mean'] for r in ranges]
        offset = (i - len(all_results)/2 + 0.5) * width
        ax.bar(x_pos + offset, means, width, label=name, color=colors.get(name, 'gray'), alpha=0.8)

    ax.set_xlabel('Document Length (pages)')
    ax.set_ylabel('Kendall\'s τ')
    ax.set_title('Positional Encoding Strategy Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ranges)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'positional_encoding_comparison.pdf', bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir / 'positional_encoding_comparison.pdf'}")

    # PLOT 2: Line plot showing degradation
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, results in all_results.items():
        means = [results[r]['mean'] for r in ranges]
        ax.plot(ranges, means, marker='o', linewidth=2.5, label=name,
                color=colors.get(name, 'gray'), markersize=8)

    ax.set_xlabel('Document Length (pages)')
    ax.set_ylabel('Kendall\'s τ')
    ax.set_title('Performance Degradation by Positional Encoding Strategy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Threshold')

    plt.tight_layout()
    plt.savefig(output_dir / 'positional_encoding_degradation.pdf', bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'positional_encoding_degradation.pdf'}")

    # PLOT 3: Focus on long documents (21-25 pages)
    fig, ax = plt.subplots(figsize=(10, 6))

    long_doc_results = []
    labels = []

    for name, results in all_results.items():
        taus = results['21-25']['all']
        long_doc_results.append(taus)
        labels.append(f"{name}\nμ={np.mean(taus):.3f}")

    bp = ax.boxplot(long_doc_results, labels=labels, patch_artist=True)

    for patch, name in zip(bp['boxes'], all_results.keys()):
        patch.set_facecolor(colors.get(name, 'gray'))
        patch.set_alpha(0.7)

    ax.set_ylabel('Kendall\'s τ')
    ax.set_title('Performance on Long Documents (21-25 pages)')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='τ = 0.5')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'positional_encoding_long_docs.pdf', bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'positional_encoding_long_docs.pdf'}")


def generate_summary_report(all_results, output_dir):
    """Generate text report"""

    output_dir = Path(output_dir)
    report_path = output_dir / 'positional_encoding_report.txt'

    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("POSITIONAL ENCODING STRATEGY COMPARISON\n")
        f.write("="*70 + "\n\n")

        f.write("MODELS TESTED:\n")
        for i, name in enumerate(all_results.keys(), 1):
            f.write(f"  {i}. {name}\n")
        f.write("\n")

        f.write("="*70 + "\n")
        f.write("RESULTS BY DOCUMENT LENGTH:\n")
        f.write("="*70 + "\n\n")

        f.write(f"{'Length':<12}")
        for name in all_results.keys():
            f.write(f"{name:<15}")
        f.write("\n" + "-"*70 + "\n")

        ranges = ["2-5", "6-10", "11-15", "16-20", "21-25"]
        for r in ranges:
            f.write(f"{r:<12}")
            for name, results in all_results.items():
                tau = results[r]['mean']
                f.write(f"{tau:>6.4f}         ")
            f.write("\n")

        f.write("\n" + "="*70 + "\n")
        f.write("ANALYSIS:\n")
        f.write("="*70 + "\n\n")

        # Compare long document performance
        f.write("Long Document Performance (21-25 pages):\n\n")

        for name, results in all_results.items():
            tau = results['21-25']['mean']
            f.write(f"  {name}: τ = {tau:.4f}\n")

        f.write("\n")

        # Find best performer
        best_name = max(all_results.keys(),
                       key=lambda n: all_results[n]['21-25']['mean'])
        best_tau = all_results[best_name]['21-25']['mean']

        if 'Learned' in all_results:
            learned_tau = all_results['Learned']['21-25']['mean']
            improvement = best_tau - learned_tau

            f.write(f"Best performer: {best_name} (τ = {best_tau:.4f})\n")
            f.write(f"Improvement over Learned: Δτ = {improvement:+.4f}\n\n")

        f.write("\n" + "="*70 + "\n")
        f.write("RECOMMENDATION FOR PAPER:\n")
        f.write("="*70 + "\n\n")

        if best_name == 'Sinusoidal':
            f.write("Sinusoidal (fixed) positional encodings resolve the training data\n")
            f.write("imbalance issue. This is the standard solution used in the original\n")
            f.write("Transformer (Vaswani et al., 2017) and demonstrates why learned\n")
            f.write("positional encodings are problematic for variable-length data.\n")
        elif best_name == 'No Position':
            f.write("Removing positional encodings entirely resolves the issue. This\n")
            f.write("suggests the model can learn ordering from content alone, without\n")
            f.write("explicit positional information.\n")

        f.write("\n" + "="*70 + "\n")

    print(f"✓ Saved: {report_path}")


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    output_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "diagnosis_results"
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print("COMPREHENSIVE POSITIONAL ENCODING COMPARISON")
    print("="*70)

    # Load all models
    models = load_all_models()

    if models is None:
        return

    # Evaluate all models
    all_results = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        results = evaluate_model(model, data_dir)
        all_results[name] = results

    # Print quick summary
    print("\n" + "="*70)
    print("QUICK COMPARISON:")
    print("="*70)

    print(f"\n{'Length':<12}", end="")
    for name in all_results.keys():
        print(f"{name:<15}", end="")
    print()
    print("-" * 70)

    ranges = ["2-5", "6-10", "11-15", "16-20", "21-25"]
    for r in ranges:
        print(f"{r:<12}", end="")
        for name, results in all_results.items():
            tau = results[r]['mean']
            print(f"{tau:>6.4f}         ", end="")
        print()

    # Generate plots and report
    create_comparison_plots(all_results, output_dir)
    generate_summary_report(all_results, output_dir)

    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - positional_encoding_comparison.pdf")
    print("  - positional_encoding_degradation.pdf")
    print("  - positional_encoding_long_docs.pdf")
    print("  - positional_encoding_report.txt")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
