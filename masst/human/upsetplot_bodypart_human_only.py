import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import UpSet, from_indicators
import os


def create_upset_plot_by_usi(tsv_path, output_path, figsize=(8.6, 2.15), min_subset_size=10):
    """
    Create UpSet plot directly from TSV file containing bodypart data based on unique structures
    
    Args:
        tsv_path: Path to TSV file with columns ['lib_usi', 'UBERONBodyPartName', 'count']
        output_path: Path to save the SVG plot
        figsize: Figure size tuple
        min_subset_size: Minimum subset size to show in plot
    """
    print(f'Loading data from {tsv_path}...')
    
    # Load the TSV file
    df = pd.read_csv(tsv_path, sep='\t')
    
    # Create binary presence/absence matrix: structures as rows, body parts as columns
    binary_df = df.pivot_table(
        index='lib_usi',
        columns='UBERONBodyPartName', 
        values='count',
        fill_value=0
    )
    binary_df = (binary_df > 0).astype(bool)
    
    print(f"Created binary matrix: {len(binary_df)} structures across {len(binary_df.columns)} body parts")
    print(f"Body parts: {list(binary_df.columns)}")
    
    # Convert to format expected by upsetplot
    upset_data = from_indicators(binary_df)
    
    print('Creating UpSet plot...')
    
    # Set Arial font and styling
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = 5.5
    
    # Create UpSet plot
    upset = UpSet(
        upset_data,
        subset_size='count',
        intersection_plot_elements=5,
        totals_plot_elements=3,
        min_subset_size=min_subset_size,
        show_counts=True,
        sort_by='cardinality',
        sort_categories_by='-cardinality',
        element_size=4,
        facecolor='0.5',
        connecting_line_width=1  # adjusted `upsetplot` package, add this parameter
    )
    
    # Generate the plot
    axes_dict = upset.plot()
    
    # Get the figure and set size
    fig = list(axes_dict.values())[0].figure
    fig.set_size_inches(figsize)
    
    # Style the axes
    for ax_name, ax in axes_dict.items():
        if ax_name == 'intersections':
            ax.tick_params(axis='both', which='major', length=1, width=0.5, labelsize=5, color='0.3', pad=1)
            ax.tick_params(axis='both', which='minor', length=0, width=0)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.25, axis='y')
            
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color('1')
            
            for patch in ax.patches:
                patch.set_facecolor('0.15')
                patch.set_edgecolor('0.15')
            
            ax.set_ylabel('Intersection\nsize', fontsize=5.5, labelpad=2)
                
        elif ax_name == 'totals':
            ax.tick_params(axis='both', which='major', length=1, width=0.5, labelsize=4, color='0.3', pad=1)
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
                
                # x, y = text.get_position()
                # text.set_position((x, y + y_offset))
                # text.set_ha('center')
                # text.set_va('center')
                # text.set_fontsize(4.5)
                
                # current_text = text.get_text()
                # if current_text.isdigit():
                #     text.set_text(f"{int(current_text):,}")
                    
        elif ax_name == 'totals':
            max_x = max([patch.get_width() for patch in ax.patches]) if ax.patches else 100
            x_offset = max_x * 0.05
            
            for text in ax.texts:
                x, y = text.get_position()
                text.set_position((x + x_offset, y))
                text.set_ha('right')
                text.set_va('center')
                text.set_fontsize(4.5)
                
                current_text = text.get_text()
                if current_text.isdigit():
                    text.set_text(f"{int(current_text):,}")
    
    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', transparent=True)
    
    print(f'UpSet plot saved to {output_path}')
    plt.close()


if __name__ == '__main__':
    
    create_upset_plot_by_usi(
        tsv_path="masst/human/data/human_only_usis_raw_count.tsv",
        output_path="masst/human/plots/upset_human_only_bodypart_USIs.svg",
        figsize=(8.7, 3.5),
        min_subset_size=1
    )
    
    # create_upset_plot_by_usi(
    #     tsv_path="masst/human/data/human_only_usis_raw_count.tsv",
    #     output_path="masst/human/plots/upset_human_only_bodypart_USIs.svg",
    #     figsize=(18, 5),
    #     min_subset_size=1
    # )