#!/usr/bin/env python
import os
"""
ULTIMATE COMPREHENSIVE COMPARISON: ALL 3 POSITIONAL ENCODING VARIANTS
======================================================================

Runs the full diagnostic suite (exp1-4) on ALL THREE models:
1. Learned positional encodings (original)
2. No positional encodings
3. Sinusoidal positional encodings

Generates:
- Individual diagnostic plots for each model
- Side-by-side comparison plots showing all 3 models together
- Comprehensive report comparing all variants

This reveals EXACTLY where and how each positional encoding strategy fails.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from scipy.stats import kendalltau, entropy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import sys

# Import models
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))
from seq2seq_transformer import TransformerSeq2Seq, OrderingDataset
from seq2seq_transformer_no_pos import TransformerSeq2SeqNoPosEncoding
from seq2seq_transformer_sinusoidal import TransformerSeq2SeqSinusoidal

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class DecodingAnalyzer:
    """Analyzes step-by-step decoding process"""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()

    def analyze_decoding_steps(self, embeddings, correct_order, page_numbers):
        """
        Track performance at each decoding step
        Returns: step-by-step tau, confidence, errors
        """
        B, N, _ = embeddings.shape
        assert B == 1, "Batch size must be 1"

        embeddings = embeddings.to(self.device)

        # Store results
        step_results = []

        with torch.no_grad():
            # Encode
            memory = self.model.encode(embeddings)

            # Greedy decoding with tracking
            decoder_input = self.model.start_token.expand(1, 1, -1)
            selected_indices = []
            used_mask = torch.zeros(1, N, device=self.device, dtype=torch.bool)

            for step in range(N):
                # Get scores for this step
                scores = self.model.decode_step(decoder_input, memory)[:, -1, :]  # (1, N)
                scores = scores.masked_fill(used_mask, -1e9)

                # Get probabilities
                probs = F.softmax(scores, dim=-1).squeeze(0).cpu().numpy()

                # Select best
                idx = scores.argmax(dim=-1).item()
                selected_indices.append(idx)

                # Compute metrics for current partial sequence
                partial_predicted = page_numbers[[selected_indices]].cpu().numpy()
                partial_correct = correct_order[:len(selected_indices)].cpu().numpy()

                # Compute tau for partial sequence
                if len(selected_indices) > 1:
                    tau, _ = kendalltau(partial_predicted, partial_correct)
                else:
                    tau = 1.0 if partial_predicted[0] == partial_correct[0] else -1.0

                # Compute confidence (max prob)
                confidence = probs[idx]

                # Compute entropy of distribution
                prob_entropy = entropy(probs + 1e-10)

                # Was this selection correct?
                correct_page = correct_order[step].item()
                selected_page = page_numbers[idx].item()
                is_correct = (correct_page == selected_page)

                # Error distance (how far off was the selected page?)
                if not is_correct:
                    correct_idx = (page_numbers == correct_page).nonzero(as_tuple=True)[0].item()
                    error_distance = abs(idx - correct_idx)
                else:
                    error_distance = 0

                step_results.append({
                    'step': step,
                    'tau': tau,
                    'confidence': confidence,
                    'entropy': prob_entropy,
                    'is_correct': is_correct,
                    'error_distance': error_distance,
                    'selected_idx': idx,
                    'correct_idx': (page_numbers == correct_page).nonzero(as_tuple=True)[0].item() if not is_correct else idx,
                    'probs': probs.copy()
                })

                # Update for next step
                used_mask[0, idx] = True
                idx_expanded = torch.tensor([[idx]], device=self.device).unsqueeze(2).expand(-1, -1, memory.shape[2])
                selected = torch.gather(memory, 1, idx_expanded)
                decoder_input = torch.cat([decoder_input, selected], dim=1)

        return step_results


def load_models():
    """Load all three model variants"""
    print("\n" + "="*70)
    print("LOADING ALL THREE MODELS")
    print("="*70)

    models = {}

    # 1. LEARNED positional encodings (original)
    try:
        model = TransformerSeq2Seq(
            embedding_dim=3072, hidden_dim=512, num_layers=6,
            num_heads=8, dropout=0.1
        )
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'best_transformer_seq2seq.pt')
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        models['Learned'] = model
        print("✓ Learned positional encodings")
    except FileNotFoundError:
        print("⚠ Learned model not found")

    # 2. NO positional encodings
    try:
        model = TransformerSeq2SeqNoPosEncoding(
            embedding_dim=3072, hidden_dim=512, num_layers=6,
            num_heads=8, dropout=0.1
        )
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'best_transformer_seq2seq_no_pos.pt')
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        models['No Position'] = model
        print("✓ No positional encodings")
    except FileNotFoundError:
        print("⚠ No-pos model not found")

    # 3. SINUSOIDAL positional encodings
    try:
        model = TransformerSeq2SeqSinusoidal(
            embedding_dim=3072, hidden_dim=512, num_layers=6,
            num_heads=8, dropout=0.1
        )
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'best_transformer_seq2seq_sinusoidal.pt')
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        models['Sinusoidal'] = model
        print("✓ Sinusoidal positional encodings")
    except FileNotFoundError:
        print("⚠ Sinusoidal model not found")

    if len(models) == 0:
        print("\n❌ No models found! Train models first.")
        return None

    print(f"\n✓ Loaded {len(models)} model(s)")
    return models


def run_analysis_on_all_models(models, data_dir, n_docs_per_range=10):
    """Run decoding analysis on all models"""

    print("\n" + "="*70)
    print("RUNNING DIAGNOSTIC ANALYSIS ON ALL MODELS")
    print("="*70)

    all_results = {}

    for model_name, model in models.items():
        print(f"\n{'='*70}")
        print(f"Analyzing: {model_name}")
        print(f"{'='*70}")

        analyzer = DecodingAnalyzer(model)
        results_by_length = defaultdict(list)

        # Test on different lengths
        for min_len, max_len in [(2, 5), (6, 10), (11, 15), (16, 20), (21, 25)]:
            dataset = OrderingDataset(data_dir, min_len=min_len, max_len=max_len, split='val')

            print(f"  Analyzing {min_len}-{max_len} page documents...")

            for i in range(min(n_docs_per_range, len(dataset))):
                doc = dataset[i]
                embeddings = doc['embeddings'].unsqueeze(0)
                correct_order = doc['correct_order']
                page_numbers = doc['page_numbers']

                step_results = analyzer.analyze_decoding_steps(embeddings, correct_order, page_numbers)

                for result in step_results:
                    result['doc_length'] = doc['length']
                    result['length_range'] = f"{min_len}-{max_len}"

                results_by_length[f"{min_len}-{max_len}"].extend(step_results)

        all_results[model_name] = results_by_length

    return all_results


def create_comparison_exp1(all_results, output_dir):
    """
    EXPERIMENT 1 COMPARISON: Step-by-Step Decoding Analysis
    4-panel plot comparing all 3 models
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Step-by-Step Decoding Comparison")
    print("="*70)

    colors = {
        'Learned': '#e74c3c',      # Red
        'No Position': '#3498db',   # Blue
        'Sinusoidal': '#2ecc71'     # Green
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Shared length ranges to plot
    length_ranges = ["2-5", "6-10", "11-15", "21-25"]

    # Plot A: Tau over decoding steps
    ax = axes[0, 0]
    for model_name, results_by_length in all_results.items():
        # Only plot long docs to keep it readable
        length_range = "21-25"
        if length_range not in results_by_length:
            continue

        steps_data = defaultdict(list)
        for result in results_by_length[length_range]:
            steps_data[result['step']].append(result['tau'])

        steps = sorted(steps_data.keys())
        avg_tau = [np.mean(steps_data[s]) for s in steps]

        ax.plot(steps, avg_tau, marker='o', label=model_name,
                linewidth=2.5, color=colors.get(model_name, 'gray'), markersize=5)

    ax.set_xlabel('Decoding Step', fontsize=11)
    ax.set_ylabel('Kendall\'s τ (Partial Sequence)', fontsize=11)
    ax.set_title('Performance Degradation During Decoding (21-25 pages)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # Plot B: Confidence over decoding steps
    ax = axes[0, 1]
    for model_name, results_by_length in all_results.items():
        length_range = "21-25"
        if length_range not in results_by_length:
            continue

        steps_data = defaultdict(list)
        for result in results_by_length[length_range]:
            steps_data[result['step']].append(result['confidence'])

        steps = sorted(steps_data.keys())
        avg_conf = [np.mean(steps_data[s]) for s in steps]

        ax.plot(steps, avg_conf, marker='o', label=model_name,
                linewidth=2.5, color=colors.get(model_name, 'gray'), markersize=5)

    ax.set_xlabel('Decoding Step', fontsize=11)
    ax.set_ylabel('Selection Confidence (Max Probability)', fontsize=11)
    ax.set_title('Model Confidence During Decoding (21-25 pages)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot C: Entropy over decoding steps
    ax = axes[1, 0]
    for model_name, results_by_length in all_results.items():
        length_range = "21-25"
        if length_range not in results_by_length:
            continue

        steps_data = defaultdict(list)
        for result in results_by_length[length_range]:
            steps_data[result['step']].append(result['entropy'])

        steps = sorted(steps_data.keys())
        avg_entropy = [np.mean(steps_data[s]) for s in steps]

        ax.plot(steps, avg_entropy, marker='o', label=model_name,
                linewidth=2.5, color=colors.get(model_name, 'gray'), markersize=5)

    ax.set_xlabel('Decoding Step', fontsize=11)
    ax.set_ylabel('Attention Entropy (nats)', fontsize=11)
    ax.set_title('Attention Uncertainty During Decoding (21-25 pages)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot D: Accuracy over decoding steps
    ax = axes[1, 1]
    for model_name, results_by_length in all_results.items():
        length_range = "21-25"
        if length_range not in results_by_length:
            continue

        steps_data = defaultdict(list)
        for result in results_by_length[length_range]:
            steps_data[result['step']].append(1.0 if result['is_correct'] else 0.0)

        steps = sorted(steps_data.keys())
        avg_acc = [np.mean(steps_data[s]) * 100 for s in steps]

        ax.plot(steps, avg_acc, marker='o', label=model_name,
                linewidth=2.5, color=colors.get(model_name, 'gray'), markersize=5)

    ax.set_xlabel('Decoding Step', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Per-Step Selection Accuracy (21-25 pages)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_exp1_decoding_steps.pdf', bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'comparison_exp1_decoding_steps.pdf'}")


def create_comparison_exp2(all_results, output_dir):
    """
    EXPERIMENT 2 COMPARISON: Error Analysis
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Error Analysis Comparison")
    print("="*70)

    colors = {
        'Learned': '#e74c3c',
        'No Position': '#3498db',
        'Sinusoidal': '#2ecc71'
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot A: Error distance distribution (long docs only)
    ax = axes[0]
    for model_name, results_by_length in all_results.items():
        length_range = "21-25"
        if length_range not in results_by_length:
            continue

        error_distances = [r['error_distance'] for r in results_by_length[length_range]
                          if not r['is_correct']]

        if error_distances:
            ax.hist(error_distances, bins=20, alpha=0.4, label=model_name,
                   density=True, color=colors.get(model_name, 'gray'))

    ax.set_xlabel('Error Distance (positions)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Distribution of Selection Errors (21-25 pages)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot B: Error rate by position
    ax = axes[1]
    for model_name, results_by_length in all_results.items():
        length_range = "21-25"
        if length_range not in results_by_length:
            continue

        position_errors = defaultdict(list)
        for result in results_by_length[length_range]:
            position_errors[result['step']].append(0.0 if result['is_correct'] else 1.0)

        positions = sorted(position_errors.keys())
        error_rates = [np.mean(position_errors[p]) * 100 for p in positions]

        ax.plot(positions, error_rates, marker='o', label=model_name,
                linewidth=2.5, color=colors.get(model_name, 'gray'), markersize=5)

    ax.set_xlabel('Decoding Step (Position)', fontsize=11)
    ax.set_ylabel('Error Rate (%)', fontsize=11)
    ax.set_title('Error Rate by Position in Sequence (21-25 pages)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_exp2_error_analysis.pdf', bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'comparison_exp2_error_analysis.pdf'}")


def create_comparison_exp3(all_results, output_dir):
    """
    EXPERIMENT 3 COMPARISON: Per-Position Accuracy
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Per-Position Accuracy Comparison")
    print("="*70)

    colors = {
        'Learned': '#e74c3c',
        'No Position': '#3498db',
        'Sinusoidal': '#2ecc71'
    }

    fig, ax = plt.subplots(figsize=(14, 7))

    for model_name, results_by_length in all_results.items():
        # Aggregate across all long documents
        all_long_results = []
        for length_range in ["16-20", "21-25"]:
            if length_range in results_by_length:
                all_long_results.extend(results_by_length[length_range])

        if not all_long_results:
            continue

        position_acc = defaultdict(list)
        for result in all_long_results:
            position_acc[result['step']].append(1.0 if result['is_correct'] else 0.0)

        positions = sorted(position_acc.keys())
        accuracies = [np.mean(position_acc[p]) * 100 for p in positions]

        ax.plot(positions, accuracies, marker='o', linewidth=2.5,
                label=model_name, color=colors.get(model_name, 'gray'), markersize=6)

    ax.set_xlabel('Position in Sequence', fontsize=12)
    ax.set_ylabel('Selection Accuracy (%)', fontsize=12)
    ax.set_title('Per-Position Accuracy on Long Documents (16-25 pages)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random chance')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_exp3_position_accuracy.pdf', bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'comparison_exp3_position_accuracy.pdf'}")


def create_comparison_exp4(all_results, output_dir):
    """
    EXPERIMENT 4 COMPARISON: Confidence vs Correctness
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Confidence Analysis Comparison")
    print("="*70)

    colors = {
        'Learned': '#e74c3c',
        'No Position': '#3498db',
        'Sinusoidal': '#2ecc71'
    }

    fig, axes = plt.subplots(len(all_results), 2, figsize=(14, 5 * len(all_results)))

    if len(all_results) == 1:
        axes = axes.reshape(1, -1)

    for idx, (model_name, results_by_length) in enumerate(all_results.items()):
        # Get long doc results
        all_long_results = []
        for length_range in ["16-20", "21-25"]:
            if length_range in results_by_length:
                all_long_results.extend(results_by_length[length_range])

        if not all_long_results:
            continue

        correct_conf = [r['confidence'] for r in all_long_results if r['is_correct']]
        incorrect_conf = [r['confidence'] for r in all_long_results if not r['is_correct']]
        correct_entropy = [r['entropy'] for r in all_long_results if r['is_correct']]
        incorrect_entropy = [r['entropy'] for r in all_long_results if not r['is_correct']]

        # Plot confidence
        ax = axes[idx, 0]
        ax.hist(correct_conf, bins=30, alpha=0.5, label='Correct', color='green', density=True)
        ax.hist(incorrect_conf, bins=30, alpha=0.5, label='Incorrect', color='red', density=True)
        ax.set_xlabel('Selection Confidence', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{model_name}: Confidence Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot entropy
        ax = axes[idx, 1]
        ax.hist(correct_entropy, bins=30, alpha=0.5, label='Correct', color='green', density=True)
        ax.hist(incorrect_entropy, bins=30, alpha=0.5, label='Incorrect', color='red', density=True)
        ax.set_xlabel('Attention Entropy', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{model_name}: Entropy Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_exp4_confidence_analysis.pdf', bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'comparison_exp4_confidence_analysis.pdf'}")


def generate_comprehensive_report(all_results, output_dir):
    """Generate comprehensive comparison report"""

    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*70)

    report_path = output_dir / 'comprehensive_comparison_report.txt'

    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("COMPREHENSIVE POSITIONAL ENCODING COMPARISON REPORT\n")
        f.write("="*70 + "\n\n")

        f.write("MODELS COMPARED:\n")
        for i, model_name in enumerate(all_results.keys(), 1):
            f.write(f"  {i}. {model_name}\n")
        f.write("\n")

        f.write("="*70 + "\n")
        f.write("PERFORMANCE BY DOCUMENT LENGTH\n")
        f.write("="*70 + "\n\n")

        # Table header
        f.write(f"{'Length':<12}")
        for model_name in all_results.keys():
            f.write(f"{model_name:<15}")
        f.write("\n" + "-"*70 + "\n")

        # Table rows
        for length_range in ["2-5", "6-10", "11-15", "16-20", "21-25"]:
            f.write(f"{length_range:<12}")
            for model_name, results_by_length in all_results.items():
                if length_range not in results_by_length:
                    f.write(f"{'N/A':<15}")
                    continue

                # Get final tau
                final_taus = [r['tau'] for r in results_by_length[length_range]
                            if r['step'] == max(rr['step'] for rr in results_by_length[length_range])]
                avg_tau = np.mean(final_taus) if final_taus else 0
                f.write(f"{avg_tau:>6.4f}         ")
            f.write("\n")

        f.write("\n" + "="*70 + "\n")
        f.write("DETAILED ANALYSIS: LONG DOCUMENTS (21-25 pages)\n")
        f.write("="*70 + "\n\n")

        for model_name, results_by_length in all_results.items():
            f.write(f"\n{model_name}:\n")
            f.write("-" * 40 + "\n")

            length_range = "21-25"
            if length_range not in results_by_length:
                f.write("  No data available\n")
                continue

            all_results_lr = results_by_length[length_range]

            # Overall performance
            final_taus = [r['tau'] for r in all_results_lr
                         if r['step'] == max(rr['step'] for rr in all_results_lr)]
            f.write(f"  Final τ: {np.mean(final_taus):.4f}\n")

            # Error rate
            errors = [r for r in all_results_lr if not r['is_correct']]
            error_rate = 100 * len(errors) / len(all_results_lr) if all_results_lr else 0
            f.write(f"  Error rate: {error_rate:.1f}%\n")

            # Average error distance
            if errors:
                avg_error_dist = np.mean([e['error_distance'] for e in errors])
                f.write(f"  Avg error distance: {avg_error_dist:.2f} positions\n")

            # Per-position accuracy
            position_acc = defaultdict(list)
            for result in all_results_lr:
                position_acc[result['step']].append(1.0 if result['is_correct'] else 0.0)

            positions = sorted(position_acc.keys())
            if positions:
                early_acc = np.mean([np.mean(position_acc[p]) for p in positions[:5]]) * 100 if len(positions) >= 5 else 0
                f.write(f"  First 5 positions accuracy: {early_acc:.1f}%\n")

            # Confidence stats
            correct_results = [r for r in all_results_lr if r['is_correct']]
            incorrect_results = [r for r in all_results_lr if not r['is_correct']]

            if correct_results:
                f.write(f"  Confidence when correct: {np.mean([r['confidence'] for r in correct_results]):.3f}\n")
            if incorrect_results:
                f.write(f"  Confidence when incorrect: {np.mean([r['confidence'] for r in incorrect_results]):.3f}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*70 + "\n\n")

        # Find best and worst
        model_scores = {}
        for model_name, results_by_length in all_results.items():
            if "21-25" in results_by_length:
                final_taus = [r['tau'] for r in results_by_length["21-25"]
                             if r['step'] == max(rr['step'] for rr in results_by_length["21-25"])]
                model_scores[model_name] = np.mean(final_taus) if final_taus else 0

        if model_scores:
            best_model = max(model_scores, key=model_scores.get)
            worst_model = min(model_scores, key=model_scores.get)

            f.write(f"1. Best performer on long docs: {best_model} (τ = {model_scores[best_model]:.4f})\n")
            f.write(f"2. Worst performer on long docs: {worst_model} (τ = {model_scores[worst_model]:.4f})\n")

            if 'Learned' in model_scores and best_model != 'Learned':
                improvement = model_scores[best_model] - model_scores['Learned']
                f.write(f"3. Improvement over learned encodings: Δτ = {improvement:+.4f}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")

    print(f"✓ Saved: {report_path}")


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    output_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "diagnosis_results"
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print("ULTIMATE COMPREHENSIVE DIAGNOSTIC COMPARISON")
    print("="*70)
    print("\nThis will:")
    print("  1. Load all 3 positional encoding variants")
    print("  2. Run full diagnostic analysis on each")
    print("  3. Generate comparison plots (exp1-4)")
    print("  4. Create comprehensive report")
    print("\n" + "="*70)

    # Load all models
    models = load_models()
    if models is None or len(models) == 0:
        return

    # Run analysis on all models
    all_results = run_analysis_on_all_models(models, data_dir, n_docs_per_range=10)

    # Generate comparison plots
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70)

    create_comparison_exp1(all_results, output_dir)
    create_comparison_exp2(all_results, output_dir)
    create_comparison_exp3(all_results, output_dir)
    create_comparison_exp4(all_results, output_dir)

    # Generate comprehensive report
    generate_comprehensive_report(all_results, output_dir)

    print("\n" + "="*70)
    print("ALL DIAGNOSTICS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - comparison_exp1_decoding_steps.pdf")
    print("  - comparison_exp2_error_analysis.pdf")
    print("  - comparison_exp3_position_accuracy.pdf")
    print("  - comparison_exp4_confidence_analysis.pdf")
    print("  - comprehensive_comparison_report.txt")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
