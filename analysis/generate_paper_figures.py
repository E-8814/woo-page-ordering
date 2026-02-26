#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
"""
Create ALL publication-quality plots (PNG only)
Includes original plots + extended analysis plots
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from math import pi

# Set publication-quality style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Map internal names to clean paper names
NAME_MAPPING = {
    'baseline_random': 'Random',
    'baseline_greedy_nearest_neighbour': 'Greedy NN',
    'TSP_similarity': 'TSP',
    'lstm': 'BiLSTM Position',
    'pointer_mlp': 'Pointer MLP',
    'pointer_lstm': 'Pointer LSTM',
    'seq2seq_transformer': 'Seq2Seq Transformer',
    'seq2seq_transformer_no_positional_encodings': 'Seq2Seq (No Pos)',
    'seq2seq_transformer_sinusoidal': 'Seq2Seq (Sinusoidal)',
    'transformer_pairwise_ranking': 'Pairwise Ranking',
    'specialized_pairwise_ranking_direct': 'Specialized PR (Direct)',
    'specialized_pairwise_ranking_curriculum': 'Specialized PR (Curriculum)'
}

# Method categories for coloring
METHOD_CATEGORIES = {
    'Random': 'baseline',
    'Greedy NN': 'heuristic',
    'TSP': 'heuristic',
    'BiLSTM Position': 'neural_baseline',
    'Pointer MLP': 'neural_baseline',
    'Pointer LSTM': 'neural_baseline',
    'Seq2Seq Transformer': 'neural_baseline',
    'Seq2Seq (No Pos)': 'neural_baseline',
    'Seq2Seq (Sinusoidal)': 'neural_baseline',
    'Pairwise Ranking': 'proposed',
    'Specialized PR (Direct)': 'proposed',
    'Specialized PR (Curriculum)': 'proposed',
}

# Individual colors for each method
METHOD_COLORS = {
    'Random': '#808080',
    'Greedy NN': '#88CCEE',
    'TSP': '#44AA99',
    'BiLSTM Position': '#999933',
    'Pointer MLP': '#DDCC77',
    'Pointer LSTM': '#CC6677',
    'Seq2Seq Transformer': '#882255',
    'Seq2Seq (No Pos)': '#AA4499',
    'Seq2Seq (Sinusoidal)': '#9966CC',
    # Last 3 models - distinct shades of red (more contrast)
    'Pairwise Ranking': '#FF6B6B',                   # Light coral red
    'Specialized PR (Direct)': '#DC143C',            # Crimson (medium bright red)
    'Specialized PR (Curriculum)': '#8B0000',        # Dark red/maroon
}

# Line styles for proposed methods
PROPOSED_STYLES = {
    'Pairwise Ranking': '-',
    'Specialized PR (Direct)': '-',
    'Specialized PR (Curriculum)': '--',
}


def parse_results(file_path):
    """Parse results from text file"""
    results = {}
    current_method = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check for method name
            if not line.startswith('2025') and ':' in line and '|' not in line:
                method_match = re.match(r'^([^:]+):', line)
                if method_match:
                    current_method = method_match.group(1).strip()
                    results[current_method] = {}
                    continue

            # Check for results lines
            if 'Target Tau' in line and current_method:
                result_match = re.search(r'\(\s*(\d+)\s*-\s*(\d+)\s*pages\).*Target Tau\s*=\s*([-\d.]+)', line)
                if result_match:
                    min_page = result_match.group(1)
                    max_page = result_match.group(2)
                    tau = float(result_match.group(3))
                    page_range = f"{min_page}-{max_page}"
                    results[current_method][page_range] = tau

    return results


def results_to_dataframe(results):
    """Convert results dict to pandas DataFrame"""
    rows = []

    for method, ranges in results.items():
        # Skip centroid_similarity completely
        if method == 'centroid_similarity':
            continue

        clean_name = NAME_MAPPING.get(method, method)
        for page_range, tau in ranges.items():
            # Extract midpoint for x-axis
            if '-' in page_range:
                min_p, max_p = page_range.split('-')
                midpoint = (int(min_p) + int(max_p)) / 2
            else:
                midpoint = float(page_range.split()[0])

            rows.append({
                'Method': clean_name,
                'Range': page_range,
                'Midpoint': midpoint,
                'Tau': tau,
                'Category': METHOD_CATEGORIES.get(clean_name, 'other')
            })

    return pd.DataFrame(rows)


def save_figure(fig, base_path):
    """Helper function to save figure in both PNG and PDF formats"""
    png_path = str(base_path).replace('.png', '.png') if '.png' in str(base_path) else f"{base_path}.png"
    pdf_path = png_path.replace('.png', '.pdf')

    plt.savefig(png_path)
    plt.savefig(pdf_path)
    print(f"✓ Saved: {png_path}")
    print(f"✓ Saved: {pdf_path}")
    plt.close()


# ============================================================================
# ORIGINAL PLOTS
# ============================================================================

def create_line_plot(df, output_path):
    """Create main line plot: Performance vs Document Length"""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group methods by category for plotting order
    methods_by_category = {cat: [] for cat in ['baseline', 'heuristic', 'neural_baseline', 'proposed']}
    for method in df['Method'].unique():
        cat = METHOD_CATEGORIES.get(method, 'other')
        if cat in methods_by_category:
            methods_by_category[cat].append(method)

    # Plot in order: baseline, heuristic, neural, proposed (so proposed is on top)
    for category in ['baseline', 'heuristic', 'neural_baseline', 'proposed']:
        for method in methods_by_category[category]:
            method_data = df[df['Method'] == method].sort_values('Midpoint')

            color = METHOD_COLORS.get(method, '#000000')
            linewidth = 3 if category == 'proposed' else 1.5
            alpha = 1.0 if category == 'proposed' else 0.7
            linestyle = PROPOSED_STYLES.get(method, '-')
            marker = 'o' if category == 'proposed' else 's'
            markersize = 8 if category == 'proposed' else 5
            zorder = 10 if category == 'proposed' else 5

            ax.plot(method_data['Midpoint'], method_data['Tau'],
                   label=method, color=color, linewidth=linewidth,
                   alpha=alpha, linestyle=linestyle, marker=marker,
                   markersize=markersize, zorder=zorder)

    ax.set_xlabel('Document Length (pages)', fontweight='bold')
    ax.set_ylabel("Kendall's τ", fontweight='bold')
    ax.set_title('Performance vs Document Length', fontweight='bold', pad=15)

    # Set x-axis ticks
    unique_ranges = df[['Midpoint', 'Range']].drop_duplicates().sort_values('Midpoint')
    ax.set_xticks(unique_ranges['Midpoint'].values)
    ax.set_xticklabels(unique_ranges['Range'].values)

    ax.set_ylim(-0.1, 1.05)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
             frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    save_figure(fig, output_path)


def create_grouped_bar_chart(df, output_path):
    """Create grouped bar chart"""

    fig, ax = plt.subplots(figsize=(14, 6))

    # Prepare data
    ranges_df = df[['Range', 'Midpoint']].drop_duplicates().sort_values('Midpoint')
    ranges = ranges_df['Range'].tolist()
    methods = df['Method'].unique()

    x = np.arange(len(ranges))
    width = 0.08

    # Group methods by category
    method_order = []
    for cat in ['baseline', 'heuristic', 'neural_baseline', 'proposed']:
        for method in methods:
            if METHOD_CATEGORIES.get(method, 'other') == cat and method not in method_order:
                method_order.append(method)

    for i, method in enumerate(method_order):
        method_data = df[df['Method'] == method].set_index('Range')['Tau']
        values = [method_data.get(r, 0) for r in ranges]

        category = METHOD_CATEGORIES.get(method, 'other')
        color = METHOD_COLORS.get(method, '#000000')
        alpha = 1.0 if category == 'proposed' else 0.7

        ax.bar(x + i * width - (len(method_order) * width) / 2, values,
              width, label=method, color=color, alpha=alpha)

    ax.set_xlabel('Document Length Range (pages)', fontweight='bold')
    ax.set_ylabel("Kendall's τ", fontweight='bold')
    ax.set_title('Performance by Document Length (All Methods)', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(ranges, rotation=0)
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
             frameon=True, fancybox=True, shadow=True, ncol=1)

    plt.tight_layout()


    save_figure(fig, output_path)


def create_curriculum_comparison(df, output_path):
    """Compare direct vs curriculum training"""

    proposed_methods = ['Specialized PR (Direct)',
                       'Specialized PR (Curriculum)']

    df_proposed = df[df['Method'].isin(proposed_methods)]

    if df_proposed.empty:
        print("⚠ No proposed methods found for curriculum comparison")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in proposed_methods:
        method_data = df_proposed[df_proposed['Method'] == method].sort_values('Midpoint')

        if method == 'Specialized PR (Direct)':
            color = METHOD_COLORS.get(method)
            linestyle = '-'
            label = 'Specialized (Direct Training)'
        else:
            color = METHOD_COLORS.get(method)
            linestyle = '--'
            label = 'Specialized (Curriculum Learning)'

        ax.plot(method_data['Midpoint'], method_data['Tau'],
               label=label, color=color, linewidth=3,
               linestyle=linestyle, marker='o', markersize=8)

    ax.set_xlabel('Document Length (pages)', fontweight='bold')
    ax.set_ylabel("Kendall's τ", fontweight='bold')
    ax.set_title('Direct Training vs Curriculum Learning', fontweight='bold', pad=15)

    unique_ranges = df_proposed[['Midpoint', 'Range']].drop_duplicates().sort_values('Midpoint')
    ax.set_xticks(unique_ranges['Midpoint'].values)
    ax.set_xticklabels(unique_ranges['Range'].values)

    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()


    save_figure(fig, output_path)


def create_heatmap(df, output_path):
    """Create heatmap of all results"""

    # Pivot data
    pivot = df.pivot(index='Method', columns='Range', values='Tau')

    # Reorder columns by midpoint
    ranges_df = df[['Range', 'Midpoint']].drop_duplicates().sort_values('Midpoint')
    column_order = ranges_df['Range'].tolist()
    pivot = pivot[column_order]

    # Reorder rows by category
    method_order = []
    for cat in ['baseline', 'heuristic', 'neural_baseline', 'proposed']:
        for method in pivot.index:
            if METHOD_CATEGORIES.get(method, 'other') == cat:
                method_order.append(method)

    pivot = pivot.reindex(method_order)

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1.0, center=0.5,
                linewidths=0.5, cbar_kws={'label': "Kendall's τ"},
                ax=ax)

    ax.set_title('Performance Heatmap: All Methods × Document Lengths',
                fontweight='bold', pad=15)
    ax.set_xlabel('Document Length Range (pages)', fontweight='bold')
    ax.set_ylabel('Method', fontweight='bold')

    plt.tight_layout()


    save_figure(fig, output_path)


# ============================================================================
# NEW EXTENDED ANALYSIS PLOTS
# ============================================================================

def create_degradation_plot(df, output_path):
    """Show performance drop from shortest to longest documents"""

    fig, ax = plt.subplots(figsize=(10, 6))

    methods_to_compare = [
        'Pointer LSTM',
        'Seq2Seq Transformer',
        'BiLSTM Position',
        'Specialized PR (Direct)'
    ]

    # Add any available methods if they exist
    available_methods = [m for m in methods_to_compare if m in df['Method'].unique()]

    for method in available_methods:
        method_data = df[df['Method'] == method].sort_values('Midpoint')

        # Normalize to first value (percentage of initial performance)
        if len(method_data) > 0:
            initial_tau = method_data.iloc[0]['Tau']
            if initial_tau > 0:  # Avoid division by zero
                normalized = method_data['Tau'] / initial_tau * 100

                ax.plot(method_data['Midpoint'], normalized,
                       label=method, marker='o', linewidth=2,
                       color=METHOD_COLORS.get(method))

    ax.set_xlabel('Document Length (pages)', fontweight='bold')
    ax.set_ylabel('Performance Retention (%)', fontweight='bold')
    ax.set_title('Performance Degradation with Document Length', fontweight='bold', pad=15)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Initial Performance')

    # Set x-axis ticks
    unique_ranges = df[['Midpoint', 'Range']].drop_duplicates().sort_values('Midpoint')
    ax.set_xticks(unique_ranges['Midpoint'].values)
    ax.set_xticklabels(unique_ranges['Range'].values)

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_figure(fig, output_path)


def create_category_boxplot(df, output_path):
    """Box plot comparing method categories"""

    fig, ax = plt.subplots(figsize=(10, 6))

    category_names = {
        'baseline': 'Baseline',
        'heuristic': 'Heuristic',
        'neural_baseline': 'Neural Baseline',
        'proposed': 'Proposed'
    }

    # Filter out categories that don't exist
    existing_categories = [c for c in ['baseline', 'heuristic', 'neural_baseline', 'proposed']
                          if c in df['Category'].unique()]

    sns.boxplot(data=df, x='Category', y='Tau',
                order=existing_categories,
                palette=['#808080', '#88CCEE', '#AA4499', '#DD3333'][:len(existing_categories)],
                ax=ax)

    ax.set_xticklabels([category_names[c] for c in existing_categories])
    ax.set_ylabel("Kendall's τ", fontweight='bold')
    ax.set_xlabel('Method Category', fontweight='bold')
    ax.set_title('Performance Distribution by Method Category', fontweight='bold', pad=15)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()


    save_figure(fig, output_path)


def create_improvement_over_baseline(df, output_path):
    """Show improvement over random baseline"""

    # Check if Random baseline exists
    if 'Random' not in df['Method'].unique():
        print("⚠ Random baseline not found, skipping improvement plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Get random baseline performance
    baseline = df[df['Method'] == 'Random'].set_index('Range')['Tau']

    # Calculate improvement for each method
    ranges = sorted(df['Range'].unique(),
                   key=lambda r: (int(r.split('-')[0]), int(r.split('-')[1])))

    methods_to_plot = [m for m in df['Method'].unique() if m != 'Random']

    x = np.arange(len(ranges))
    width = 0.08

    for i, method in enumerate(methods_to_plot):
        improvements = []
        for r in ranges:
            method_tau = df[(df['Method'] == method) & (df['Range'] == r)]['Tau'].values
            base_tau = baseline.get(r, 0)

            if len(method_tau) > 0:
                # Improvement = (method - baseline) / (1 - baseline) * 100
                if base_tau < 1:
                    improvement = (method_tau[0] - base_tau) / (1 - base_tau) * 100
                else:
                    improvement = 0
                improvements.append(improvement)
            else:
                improvements.append(0)

        ax.bar(x + i * width, improvements, width,
              label=method, color=METHOD_COLORS.get(method), alpha=0.8)

    ax.set_xlabel('Document Length Range', fontweight='bold')
    ax.set_ylabel('Improvement over Random (%)', fontweight='bold')
    ax.set_title('Relative Improvement Over Random Baseline', fontweight='bold', pad=15)
    ax.set_xticks(x + width * len(methods_to_plot) / 2)
    ax.set_xticklabels(ranges)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()


    save_figure(fig, output_path)


def create_short_vs_long_scatter(df, output_path):
    """Scatter: performance on short docs (2-5) vs long docs (21-25)"""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Exclude ----transformer_pairwise_scoring from this plot
    methods = [m for m in df['Method'].unique() if m != '----transformer_pairwise_scoring']

    for method in methods:
        method_data = df[df['Method'] == method]

        short_tau = method_data[method_data['Range'] == '2-5']['Tau'].values
        long_tau = method_data[method_data['Range'] == '21-25']['Tau'].values

        if len(short_tau) > 0 and len(long_tau) > 0:
            category = METHOD_CATEGORIES.get(method, 'other')
            color = METHOD_COLORS.get(method)
            size = 150 if category == 'proposed' else 80
            marker = 'o' if category == 'proposed' else 's'

            # X-axis: short docs (2-5), Y-axis: long docs (21-25)
            ax.scatter(short_tau[0], long_tau[0],
                      color=color, s=size, marker=marker,
                      alpha=0.8, edgecolors='black', linewidth=1,
                      label=method)

    # Diagonal line (perfect scaling)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=2, label='Perfect Scaling')

    ax.set_xlabel('Performance on Short Docs (2-5 pages)', fontweight='bold')
    ax.set_ylabel('Performance on Long Docs (21-25 pages)', fontweight='bold')
    ax.set_title('Short vs Long Document Performance', fontweight='bold', pad=15)
    ax.set_xlim(-0.1, 1.05)
    ax.set_ylim(-0.1, 1.05)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()


    save_figure(fig, output_path)


def create_slope_analysis(df, output_path):
    """Analyze and compare degradation slopes"""

    slopes = []

    for method in df['Method'].unique():
        method_data = df[df['Method'] == method].sort_values('Midpoint')

        if len(method_data) >= 2:
            # Linear regression
            slope, intercept, r_value, _, _ = stats.linregress(
                method_data['Midpoint'], method_data['Tau']
            )

            slopes.append({
                'Method': method,
                'Slope': slope,
                'Category': METHOD_CATEGORIES.get(method, 'other'),
                'R²': r_value**2
            })

    slopes_df = pd.DataFrame(slopes).sort_values('Slope', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = [METHOD_COLORS.get(m, '#000000') for m in slopes_df['Method']]

    bars = ax.barh(slopes_df['Method'], slopes_df['Slope'], color=colors, alpha=0.8)

    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Performance Degradation Slope (τ per page)', fontweight='bold')
    ax.set_ylabel('Method', fontweight='bold')
    ax.set_title('Rate of Performance Degradation', fontweight='bold', pad=15)

    # Add value labels
    for i, (idx, row) in enumerate(slopes_df.iterrows()):
        ax.text(row['Slope'], i, f"  {row['Slope']:.4f}",
               va='center', fontsize=8)

    plt.tight_layout()


    save_figure(fig, output_path)


def create_radar_chart(df, output_path):
    """Radar chart comparing top methods across all ranges"""

    # Select top methods (those that exist in data)
    possible_methods = [
        'Specialized PR (Direct)',
        'Specialized PR (Curriculum)',
        'Pairwise Ranking',
        'Pointer LSTM',
        'Seq2Seq Transformer'
    ]

    top_methods = [m for m in possible_methods if m in df['Method'].unique()][:4]

    if len(top_methods) == 0:
        print("⚠ No methods found for radar chart")
        return

    ranges = sorted(df['Range'].unique(),
                   key=lambda r: (int(r.split('-')[0]), int(r.split('-')[1])))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    angles = [n / float(len(ranges)) * 2 * pi for n in range(len(ranges))]
    angles += angles[:1]  # Complete the circle

    for method in top_methods:
        values = []
        for r in ranges:
            tau = df[(df['Method'] == method) & (df['Range'] == r)]['Tau'].values
            values.append(tau[0] if len(tau) > 0 else 0)

        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2,
               label=method, color=METHOD_COLORS.get(method))
        ax.fill(angles, values, alpha=0.15, color=METHOD_COLORS.get(method))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(ranges)
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Range Performance Comparison',
                fontweight='bold', pad=20, size=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()


    save_figure(fig, output_path)


def create_win_rate_heatmap(df, output_path):
    """Heatmap showing head-to-head win rates"""

    methods = sorted(df['Method'].unique())
    win_matrix = pd.DataFrame(0, index=methods, columns=methods)

    ranges = df['Range'].unique()

    for r in ranges:
        range_data = df[df['Range'] == r]

        for m1 in methods:
            tau1 = range_data[range_data['Method'] == m1]['Tau'].values

            for m2 in methods:
                if m1 == m2:
                    continue

                tau2 = range_data[range_data['Method'] == m2]['Tau'].values

                if len(tau1) > 0 and len(tau2) > 0:
                    if tau1[0] > tau2[0]:
                        win_matrix.loc[m1, m2] += 1

    # Convert to percentage
    win_matrix = win_matrix / len(ranges) * 100

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(win_matrix, annot=True, fmt='.0f', cmap='RdYlGn',
                vmin=0, vmax=100, center=50,
                linewidths=0.5, cbar_kws={'label': 'Win Rate (%)'},
                ax=ax)

    ax.set_title('Head-to-Head Win Rate (%)', fontweight='bold', pad=15)
    ax.set_xlabel('Opponent Method', fontweight='bold')
    ax.set_ylabel('Method', fontweight='bold')

    plt.tight_layout()


    save_figure(fig, output_path)


def main():
    # Paths
    results_file = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "results" / "results_baseline.txt"
    output_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "figures"
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print("Creating ALL Publication-Quality Plots (PNG + PDF)")
    print("="*70 + "\n")

    # Parse results
    print("Parsing results...")
    results = parse_results(results_file)

    if not results:
        print("ERROR: No results parsed!")
        return

    print(f"Parsed {len(results)} methods")
    for method, ranges in results.items():
        print(f"  {method}: {len(ranges)} ranges")

    df = results_to_dataframe(results)

    if df.empty:
        print("ERROR: DataFrame is empty!")
        return

    print(f"\nFound {len(df['Method'].unique())} methods")
    print(f"Found {len(df['Range'].unique())} document length ranges\n")

    # Create all plots
    print("="*70)
    print("CREATING PLOTS...")
    print("="*70 + "\n")

    print("📊 ORIGINAL PLOTS:")
    print("-" * 70)
    create_line_plot(df, output_dir / "01_performance_vs_length.png")
    create_grouped_bar_chart(df, output_dir / "02_grouped_bars.png")
    create_curriculum_comparison(df, output_dir / "03_curriculum_comparison.png")
    create_heatmap(df, output_dir / "04_heatmap.png")

    print("\n📈 EXTENDED ANALYSIS PLOTS:")
    print("-" * 70)
    create_degradation_plot(df, output_dir / "05_degradation_analysis.png")
    create_category_boxplot(df, output_dir / "06_category_boxplot.png")
    create_improvement_over_baseline(df, output_dir / "07_improvement_over_baseline.png")
    create_short_vs_long_scatter(df, output_dir / "08_short_vs_long_scatter.png")
    create_slope_analysis(df, output_dir / "09_slope_analysis.png")
    create_radar_chart(df, output_dir / "10_radar_chart.png")
    create_win_rate_heatmap(df, output_dir / "11_win_rate_heatmap.png")

    print("\n" + "="*70)
    print("✅ ALL PLOTS COMPLETED!")
    print("="*70)
    print(f"\nAll figures saved to: {output_dir}")
    print(f"Total plots created: 11 (each in PNG + PDF format = 22 files)\n")

    print("📋 Plot Index:")
    print("  01 - Performance vs Length (line plot)")
    print("  02 - Grouped Bar Chart")
    print("  03 - Curriculum Comparison")
    print("  04 - Performance Heatmap")
    print("  05 - Degradation Analysis")
    print("  06 - Category Box Plot")
    print("  07 - Improvement over Baseline")
    print("  08 - Short vs Long Scatter")
    print("  09 - Slope Analysis")
    print("  10 - Radar Chart")
    print("  11 - Win Rate Heatmap")
    print("\n")


if __name__ == "__main__":
    main()
