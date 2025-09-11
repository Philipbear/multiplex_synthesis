import pandas as pd
import os
import matplotlib.pyplot as plt


def plot_masst_match_scatter():
    """
    Generate scatter plots for MASST match distribution by USI and structure
    using the same parameters as masst_matches_summary.py
    """
    print(" === Generating MASST Match Scatter Plots ===")
    
    # Load data files
    usi_match_counts = pd.read_csv("data/usi_match_counts.tsv", sep='\t')
    structure_match_counts = pd.read_csv("data/structure_match_counts.tsv", sep='\t')
    
    # Plot USI match distribution
    _create_match_scatter_plot(
        usi_match_counts,
        'USI',
        "plots/usi_masst_matches_scatter"
    )
    
    # Plot structure match distribution
    _create_match_scatter_plot(
        structure_match_counts,
        'structure',
        "plots/structure_masst_matches_scatter"
    )
    
    print("MASST match scatter plots generated successfully!")


def _create_match_scatter_plot(counts_df, data_type, output_prefix):
    """
    Create a scatter plot showing the number of matches for each item
    Using exact same parameters as masst_matches_summary.py
    """
    # Sort by match count in descending order
    counts_df = counts_df.sort_values('match_count', ascending=False).reset_index(drop=True)
    
    # filter
    counts_df = counts_df[counts_df['match_count'] > 0].reset_index(drop=True)
    
    # Set font to Arial
    plt.rcParams['font.family'] = 'Arial'
    
    # Set up the plot (same size as original)
    fig, ax = plt.subplots(figsize=(3.5, 1.7))
    
    # Plot scatter plot (same parameters as original)
    plt.scatter(range(len(counts_df)), counts_df['match_count'], 
               alpha=0.65, s=1.8, color='steelblue', edgecolor='none')
    
    # Add log scale for y-axis
    plt.yscale('log')
    
    plt.ylim(bottom=1)
    
    # Add labels (same font size as original)
    if data_type == 'USI':
        plt.xlabel('MS/MS library USI (sorted)', fontsize=8)
    else:
        plt.xlabel('Unique chemical structure (sorted)', fontsize=8)

    plt.ylabel('Number of MASST\nspectral matches', fontsize=8)

    # Add tick parameters (exact same as original)
    plt.tick_params(axis='x', which='major', length=1, width=0.8, pad=1,
                    colors='0.2', labelsize=6)
    plt.tick_params(axis='y', which='major', length=1, width=0.8, pad=1,
                    colors='0.2', labelsize=6)
    plt.tick_params(axis='y', which='minor', length=0, width=0.8, pad=1,
                    colors='0.2', labelsize=6)

    # For x axis, add commas as thousands separator
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    
    # Calculate statistics for annotation
    at_least_1_match = counts_df[counts_df['match_count'] >= 1].shape[0]
    at_least_3_matches = counts_df[counts_df['match_count'] >= 3].shape[0]
    at_least_5_matches = counts_df[counts_df['match_count'] >= 5].shape[0]

    # Add annotation with statistics (same positioning and font size)
    stats_text_1 = f"≥ 1 MASST matches: {at_least_1_match:,d} {data_type}s"
    stats_text_2 = f"≥ 3 MASST matches: {at_least_3_matches:,d} {data_type}s"
    stats_text_3 = f"≥ 5 MASST matches: {at_least_5_matches:,d} {data_type}s"

    x_pos = 0.30 if data_type == 'USI' else 0.24
    plt.annotate(stats_text_1, xy=(x_pos, 0.75), xycoords='axes fraction', fontsize=7)
    plt.annotate(stats_text_2, xy=(x_pos, 0.65), xycoords='axes fraction', fontsize=7)
    plt.annotate(stats_text_3, xy=(x_pos, 0.55), xycoords='axes fraction', fontsize=7)

    # Add grid (same parameters as original)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Style adjustments (same as original)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('0.2')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure (same formats as original)
    save_file_svg = f"{output_prefix}.svg"
    save_file_png = f"{output_prefix}.png"
    
    plt.savefig(save_file_svg, format='svg', bbox_inches='tight')
    plt.savefig(save_file_png, format='png', bbox_inches='tight', dpi=600)
    
    print(f"Scatter plot saved to: {save_file_svg}")
    print(f"Scatter plot saved to: {save_file_png}")
    
    plt.close()
    

if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("plots", exist_ok=True)
    
    plot_masst_match_scatter()
    