#!/usr/bin/env python
import os
"""
ABLATION COMPARISON: Original vs No-Positional-Encoding
=========================================================

Compare the original seq2seq transformer with the version without
positional encodings to prove/disprove the hypothesis that learned
positional encodings are the root cause of long-document failure.

Run this AFTER training transformer_seq2seq_no_pos.py
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

sns.set_style("whitegrid")


def load_models():
    """Load both models"""
    print("Loading models...")

    # Original model
    model_orig = TransformerSeq2Seq(
        embedding_dim=3072,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1
    )
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'best_transformer_seq2seq.pt')
    model_orig.load_state_dict(torch.load(model_path, map_location='cpu'))
    model_orig.eval()
    print("✓ Original model loaded")

    # No-pos model
    try:
        model_nopos = TransformerSeq2SeqNoPosEncoding(
            embedding_dim=3072,
            hidden_dim=512,
            num_layers=6,
            num_heads=8,
            dropout=0.1
        )
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'best_transformer_seq2seq_no_pos.pt')
        model_nopos.load_state_dict(torch.load(model_path, map_location='cpu'))
        model_nopos.eval()
        print("✓ No-pos model loaded")
    except FileNotFoundError:
        print("⚠ No-pos model not found. Train it first with:")
        print("  python transformer_seq2seq_no_pos.py")
        return None, None

    return model_orig, model_nopos


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


def create_comparison_plots(results_orig, results_nopos, output_dir):
    """Create side-by-side comparison plots"""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Plot 1: Performance comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ranges = ["2-5", "6-10", "11-15", "16-20", "21-25"]
    x_pos = np.arange(len(ranges))
    width = 0.35

    # Plot A: Bar chart comparison
    ax = axes[0]
    means_orig = [results_orig[r]['mean'] for r in ranges]
    means_nopos = [results_nopos[r]['mean'] for r in ranges]

    ax.bar(x_pos - width/2, means_orig, width, label='With Pos Encoding', color='steelblue', alpha=0.8)
    ax.bar(x_pos + width/2, means_nopos, width, label='Without Pos Encoding', color='coral', alpha=0.8)

    ax.set_xlabel('Document Length (pages)')
    ax.set_ylabel('Kendall\'s τ')
    ax.set_title('Performance Comparison: Original vs No-Pos')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ranges)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot B: Improvement/degradation
    ax = axes[1]
    improvements = [results_nopos[r]['mean'] - results_orig[r]['mean'] for r in ranges]

    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax.bar(x_pos, improvements, color=colors, alpha=0.7)

    ax.set_xlabel('Document Length (pages)')
    ax.set_ylabel('Δτ (No-Pos - Original)')
    ax.set_title('Performance Change After Removing Pos Encodings')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ranges)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_comparison.pdf', bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir / 'ablation_comparison.pdf'}")

    # Plot 2: Distribution comparison for long docs
    fig, ax = plt.subplots(figsize=(10, 6))

    taus_orig_long = results_orig['21-25']['all']
    taus_nopos_long = results_nopos['21-25']['all']

    ax.hist(taus_orig_long, bins=20, alpha=0.5, label='With Pos Encoding', color='steelblue', density=True)
    ax.hist(taus_nopos_long, bins=20, alpha=0.5, label='Without Pos Encoding', color='coral', density=True)

    ax.set_xlabel('Kendall\'s τ')
    ax.set_ylabel('Density')
    ax.set_title('Performance Distribution on Long Documents (21-25 pages)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add mean lines
    ax.axvline(np.mean(taus_orig_long), color='steelblue', linestyle='--', linewidth=2, label=f'Original mean: {np.mean(taus_orig_long):.3f}')
    ax.axvline(np.mean(taus_nopos_long), color='coral', linestyle='--', linewidth=2, label=f'No-pos mean: {np.mean(taus_nopos_long):.3f}')

    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_long_docs_distribution.pdf', bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'ablation_long_docs_distribution.pdf'}")


def generate_summary_report(results_orig, results_nopos, output_dir):
    """Generate text report of findings"""

    output_dir = Path(output_dir)
    report_path = output_dir / 'ablation_report.txt'

    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ABLATION STUDY: POSITIONAL ENCODING REMOVAL\n")
        f.write("="*70 + "\n\n")

        f.write("HYPOTHESIS:\n")
        f.write("Learned positional encodings cause seq2seq failure on long documents\n")
        f.write("due to insufficient training exposure at later positions.\n\n")

        f.write("METHOD:\n")
        f.write("Train two identical 6-layer encoder-decoder transformers:\n")
        f.write("  1. Original: WITH learned positional encodings\n")
        f.write("  2. Ablation: WITHOUT positional encodings\n\n")

        f.write("="*70 + "\n")
        f.write("RESULTS:\n")
        f.write("="*70 + "\n\n")

        f.write("Performance by Document Length:\n\n")
        f.write(f"{'Length':<15} {'Original':<12} {'No-Pos':<12} {'Δτ':<12} {'Improvement'}\n")
        f.write("-" * 70 + "\n")

        ranges = ["2-5", "6-10", "11-15", "16-20", "21-25"]
        for r in ranges:
            orig_tau = results_orig[r]['mean']
            nopos_tau = results_nopos[r]['mean']
            delta = nopos_tau - orig_tau
            improvement = "✓ YES" if delta > 0.05 else ("~ SAME" if abs(delta) < 0.05 else "✗ WORSE")

            f.write(f"{r:<15} {orig_tau:>6.4f}       {nopos_tau:>6.4f}       {delta:>+6.4f}     {improvement}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("ANALYSIS:\n")
        f.write("="*70 + "\n\n")

        # Check if removing pos encodings fixed long-doc performance
        orig_long = results_orig['21-25']['mean']
        nopos_long = results_nopos['21-25']['mean']
        improvement = nopos_long - orig_long

        f.write("\n" + "="*70 + "\n")
        f.write("CONCLUSION:\n")
        f.write("="*70 + "\n\n")

        f.write("See ablation_comparison.pdf for visual summary.\n")
        f.write("\n" + "="*70 + "\n")

    print(f"✓ Saved: {report_path}")


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    output_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "diagnosis_results"
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print("ABLATION STUDY: POSITIONAL ENCODING REMOVAL")
    print("="*70)

    # Load models
    model_orig, model_nopos = load_models()

    if model_orig is None or model_nopos is None:
        print("\n⚠ Cannot proceed without both models.")
        return

    # Evaluate both models
    print("\nEvaluating original model...")
    results_orig = evaluate_model(model_orig, data_dir)

    print("\nEvaluating no-pos model...")
    results_nopos = evaluate_model(model_nopos, data_dir)

    # Print quick summary
    print("\n" + "="*70)
    print("QUICK COMPARISON:")
    print("="*70)
    print(f"\n{'Length':<15} {'Original':<12} {'No-Pos':<12} {'Change'}")
    print("-" * 50)
    for r in ["2-5", "6-10", "11-15", "16-20", "21-25"]:
        orig = results_orig[r]['mean']
        nopos = results_nopos[r]['mean']
        delta = nopos - orig
        print(f"{r:<15} {orig:.4f}       {nopos:.4f}       {delta:+.4f}")

    # Generate plots and report
    create_comparison_plots(results_orig, results_nopos, output_dir)
    generate_summary_report(results_orig, results_nopos, output_dir)

    print("\n" + "="*70)
    print("ABLATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print("\nNext steps:")
    print("  1. Review ablation_comparison.pdf")
    print("  2. Read ablation_report.txt for interpretation")
    print("  3. Update paper Section 5.2 with findings")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
