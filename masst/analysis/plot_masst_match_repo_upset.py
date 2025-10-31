import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import UpSet, from_indicators
import os


def create_upset_plot_by_usi(tsv_path, output_path, figsize=(8.6, 2.15), min_subset_size=10):
    """
    Create UpSet plot for repository distribution based on unique USIs
    
    Args:
        tsv_path: Path to TSV file with USI repository presence matrix
        output_path: Path to save the SVG plot
        figsize: Figure size tuple
        min_subset_size: Minimum subset size to show in plot
    """
    print(f'Loading data from {tsv_path}...')
    
    # Load the TSV file
    usi_presence = pd.read_csv(tsv_path, sep='\t', index_col=0)
    
    # Rename columns to full repository names
    repo_name_map = {
        'MS': 'GNPS/MassIVE',
        'MT': 'MetaboLights',
        'NO': 'NORMAN',
        'ST': 'Metabolomics Workbench'
    }
    
    usi_presence = usi_presence.rename(columns=repo_name_map)
    
    # Convert to boolean
    binary_df = (usi_presence > 0).astype(bool)
    
    print(f"Created binary matrix: {len(binary_df)} USIs across {len(binary_df.columns)} repositories")
    print(f"Repositories: {list(binary_df.columns)}")
    
    # Print some stats
    for repo in binary_df.columns:
        repo_only_count = binary_df[binary_df.sum(axis=1) == 1][repo].sum()
        print(f"USIs with matches only in '{repo}': {repo_only_count}")
    
    # Convert to format expected by upsetplot
    upset_data = from_indicators(binary_df)
    
    print('Creating UpSet plot...')
    
    # Set Arial font and styling
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = 7
    
    # Create UpSet plot
    upset = UpSet(
        upset_data,
        subset_size='count',
        intersection_plot_elements=5,
        totals_plot_elements=4,
        min_subset_size=min_subset_size,
        show_counts=True,
        sort_by='cardinality',
        sort_categories_by='-cardinality',
        element_size=8,
        facecolor='0.5',
        connecting_line_width=1.2
    )
    
    # Generate the plot
    axes_dict = upset.plot()
    
    # Get the figure and set size
    fig = list(axes_dict.values())[0].figure
    fig.set_size_inches(figsize)
    
    # Style the axes
    for ax_name, ax in axes_dict.items():
        if ax_name == 'intersections':
            ax.tick_params(axis='both', which='major', length=1.5, width=0.5, labelsize=6, color='0.3', pad=1)
            ax.tick_params(axis='both', which='minor', length=0, width=0)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.25, axis='y')
            
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color('1')
            
            for patch in ax.patches:
                patch.set_facecolor('0.15')
                patch.set_edgecolor('0.15')
            
            ax.set_ylabel('Intersection\nsize', fontsize=7, labelpad=4)
                
        elif ax_name == 'totals':
            ax.tick_params(axis='both', which='major', length=1.5, width=0.5, labelsize=6, color='0.3', pad=1)
            ax.tick_params(axis='both', which='minor', length=0, width=0)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.25, axis='x')
            
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color('0.4')
            
            for patch in ax.patches:
                patch.set_facecolor('0.15')
                patch.set_edgecolor('0.15')
    
    # Format labels with commas
    for ax_name, ax in axes_dict.items():
        if ax_name == 'intersections':
            max_y = max([patch.get_height() for patch in ax.patches]) if ax.patches else 100
            y_offset = max_y * 0.075
            
            for text in ax.texts:
                text.set_visible(False)
                    
        elif ax_name == 'totals':
            max_x = max([patch.get_width() for patch in ax.patches]) if ax.patches else 100
            x_offset = max_x * 0.05
            
            for text in ax.texts:
                x, y = text.get_position()
                text.set_position((x + x_offset, y))
                text.set_ha('right')
                text.set_va('center')
                text.set_fontsize(6)
                
                current_text = text.get_text()
                if current_text.isdigit():
                    text.set_text(f"{int(current_text):,}")
    
    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', transparent=True)
    
    print(f'UpSet plot saved to {output_path}')
    plt.close()


def create_repo_presence_matrix():
    """
    Create repository presence matrix from the original data if needed
    This is just a helper function - assumes you already have the matrix
    """
    # This function would create the matrix, but since you already have it
    # in data/usi_repo_presence_matrix.tsv, we'll use that directly
    pass


if __name__ == '__main__':
    # Make sure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    create_upset_plot_by_usi(
        tsv_path="data/usi_repo_presence_matrix.tsv",
        output_path="plots/upset_repo_distribution_USIs.svg",
        figsize=(3.85, 1.75),
        min_subset_size=1
    )