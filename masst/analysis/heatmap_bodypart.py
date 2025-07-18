import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os


def prepare_heatmap_data(bodypart_data, top_n=50):
    """
    Prepare data for heatmap plotting
    
    Args:
        bodypart_data: DataFrame with columns ['UBERONBodyPartName', 'inchikey_2d', 'count', 'name']
        top_n: Number of top structures to include
    
    Returns:
        DataFrame ready for heatmap plotting
    """
    
    ####### filter #######
    # name contain exactly one '_'
    bodypart_data = bodypart_data[bodypart_data['name'].str.count('_') == 1].reset_index(drop=True)
    
    # Calculate total matches per structure across all body parts
    structure_totals = bodypart_data.groupby('inchikey_2d').agg({
        'count': 'sum',
        'name': 'first'  # Take first name for each structure
    }).reset_index()
    structure_totals = structure_totals.sort_values('count', ascending=False)
    
    # Get top N structures
    top_structures = structure_totals.head(top_n)
    
    # Filter data for top structures
    filtered_data = bodypart_data[bodypart_data['inchikey_2d'].isin(top_structures['inchikey_2d'])]
    
    # Create pivot table: structures as rows, body parts as columns
    heatmap_data = filtered_data.pivot_table(
        index='inchikey_2d', 
        columns='UBERONBodyPartName', 
        values='count',
        fill_value=0
    )
    
    # Create name mapping from top structures (ensuring uniqueness)
    name_mapping = top_structures.set_index('inchikey_2d')['name']
    
    # Map inchikey_2d to compound names
    heatmap_data.index = heatmap_data.index.map(name_mapping)
    
    # Sort by total matches (row sums) to maintain ranking
    row_totals = heatmap_data.sum(axis=1)
    heatmap_data = heatmap_data.loc[row_totals.sort_values(ascending=False).index]
    
    return heatmap_data


def plot_human_structure_heatmap(top_n=50):
    """
    Create heatmap for top structures in human body parts
    """
    print("=== Creating Human Structure Heatmap ===")
    
    # Load human data
    human_data = pd.read_csv('data/bodypart/human_compound_bodypart_counts.tsv', sep='\t')
    print(f"Loaded human data: {len(human_data)} structure-bodypart combinations")
    
    # Prepare heatmap data
    human_heatmap_data = prepare_heatmap_data(human_data, top_n=top_n)
    
    # Create heatmap
    output_prefix = "human_structures_bodyparts"
    
    # Apply log transformation
    data_log = np.log1p(human_heatmap_data)
        
    # Set Arial font
    plt.rcParams['font.family'] = 'Arial'
    
    # Create custom colormap: white -> blue -> red
    colors = [(1, 1, 1), (0.78, 0.84, 0.94), (0.94, 0.69, 0.65)]
    n_bins = 20
    cmap_wbr = LinearSegmentedColormap.from_list('white_blue_red', colors, N=n_bins)
    
    # Create clustermap
    clustermap = sns.clustermap(
        data_log,
        cmap=cmap_wbr,
        linewidths=0.5,
        annot=False,
        robust=True,
        figsize=(6.5, 4.7),
        row_cluster=True,     
        col_cluster=False,
        metric='euclidean',   
        cbar_pos=(-0.1, 1, 0.10, 0.023),  # x, y, width, height
        cbar_kws={'orientation': 'horizontal'},
        dendrogram_ratio=0.035,
        xticklabels=True,
        yticklabels=True
    )
    
    # # Adjust gap between dendrogram and heatmap
    # heatmap_pos = clustermap.ax_heatmap.get_position()
    # gap = 0.024
    # new_heatmap_left = heatmap_pos.x0 + gap
    # new_heatmap_width = heatmap_pos.width - gap
    
    # clustermap.ax_heatmap.set_position([new_heatmap_left, heatmap_pos.y0, 
    #                                    new_heatmap_width, heatmap_pos.height])
    
    # Move dendrogram to the right side
    # Get current positions
    heatmap_pos = clustermap.ax_heatmap.get_position()
    dendrogram_pos = clustermap.ax_row_dendrogram.get_position()
    
    # Calculate new positions
    gap = 0.005
    new_heatmap_width = heatmap_pos.width - gap
    new_dendrogram_left = heatmap_pos.x0 + new_heatmap_width + gap
    
    # Set new positions
    clustermap.ax_heatmap.set_position([heatmap_pos.x0, heatmap_pos.y0, 
                                        new_heatmap_width, heatmap_pos.height])
    clustermap.ax_row_dendrogram.set_position([new_dendrogram_left, dendrogram_pos.y0,
                                                dendrogram_pos.width, dendrogram_pos.height])
    
    # Flip the dendrogram to face left
    clustermap.ax_row_dendrogram.invert_xaxis()
    
    # Adjust tick parameters
    clustermap.ax_heatmap.tick_params(
        axis='x',
        which='major',
        length=0.75,
        width=0.35,
        pad=2,
        labelsize=4.2,
        colors='0.2',
        bottom=False,
        top=True,
        labelbottom=False,
        labeltop=True,
        labelrotation=90
    )
    
    clustermap.ax_heatmap.tick_params(
        axis='y',
        which='major',
        length=0.75,
        width=0.35,
        pad=2,
        labelsize=4.2,
        colors='0.2',
        left=True,
        right=False,
        labelleft=True,
        labelright=False
    )
    
    # Remove axis titles
    clustermap.ax_heatmap.set_xlabel('')
    clustermap.ax_heatmap.set_ylabel('')
    
    # Format colorbar
    clustermap.ax_cbar.set_xlabel('Log(count + 1)', fontsize=4.5, labelpad=0.5, color='0.2')
    clustermap.ax_cbar.tick_params(labelsize=4.5, length=0.75, width=0.35, pad=0.5, colors='0.2')
    
    # Save figures
    os.makedirs('plots/heatmaps', exist_ok=True)
    plt.savefig(f'plots/heatmaps/{output_prefix}_heatmap.svg', bbox_inches='tight', format='svg', transparent=True)
    plt.savefig(f'plots/heatmaps/{output_prefix}_heatmap.png', bbox_inches='tight', format='png', dpi=600, transparent=True)
    
    print(f"Heatmap saved: plots/heatmaps/{output_prefix}_heatmap.svg")
    plt.close()
    


def plot_rodent_structure_heatmap(top_n=50):
    """
    Create heatmap for top structures in rodent body parts
    """
    print("=== Creating Rodent Structure Heatmap ===")
    
    # Load rodent data
    rodent_data = pd.read_csv('data/bodypart/rodent_compound_bodypart_counts.tsv', sep='\t')
    print(f"Loaded rodent data: {len(rodent_data)} structure-bodypart combinations")
    
    # Prepare heatmap data
    rodent_heatmap_data = prepare_heatmap_data(rodent_data, top_n=top_n)
    
    # Create heatmap
    output_prefix = "rodent_structures_bodyparts"
    
    # Apply log transformation
    data_log = np.log1p(rodent_heatmap_data)
        
    # Set Arial font
    plt.rcParams['font.family'] = 'Arial'
    
    # Create custom colormap: white -> blue -> red
    colors = [(1, 1, 1), (0.78, 0.84, 0.94), (0.94, 0.69, 0.65)]
    n_bins = 20
    cmap_wbr = LinearSegmentedColormap.from_list('white_blue_red', colors, N=n_bins)
    
    # Create clustermap
    clustermap = sns.clustermap(
        data_log,
        cmap=cmap_wbr,
        linewidths=0.5,
        annot=False,
        robust=True,
        figsize=(6.5, 4.7),
        row_cluster=True,     
        col_cluster=False,
        metric='euclidean',   
        cbar_pos=(-0.1, 1, 0.10, 0.023),  # x, y, width, height
        cbar_kws={'orientation': 'horizontal'},
        dendrogram_ratio=0.035,
        xticklabels=True,
        yticklabels=True
    )
    
    # # Adjust gap between dendrogram and heatmap
    # heatmap_pos = clustermap.ax_heatmap.get_position()
    # gap = 0.024
    # new_heatmap_left = heatmap_pos.x0 + gap
    # new_heatmap_width = heatmap_pos.width - gap
    
    # clustermap.ax_heatmap.set_position([new_heatmap_left, heatmap_pos.y0, 
    #                                    new_heatmap_width, heatmap_pos.height])
    
    # Move dendrogram to the right side
    # Get current positions
    heatmap_pos = clustermap.ax_heatmap.get_position()
    dendrogram_pos = clustermap.ax_row_dendrogram.get_position()
    
    # Calculate new positions
    gap = 0.005
    new_heatmap_width = heatmap_pos.width - gap
    new_dendrogram_left = heatmap_pos.x0 + new_heatmap_width + gap
    
    # Set new positions
    clustermap.ax_heatmap.set_position([heatmap_pos.x0, heatmap_pos.y0, 
                                        new_heatmap_width, heatmap_pos.height])
    clustermap.ax_row_dendrogram.set_position([new_dendrogram_left, dendrogram_pos.y0,
                                                dendrogram_pos.width, dendrogram_pos.height])
    
    # Flip the dendrogram to face left
    clustermap.ax_row_dendrogram.invert_xaxis()
    
    # Adjust tick parameters
    clustermap.ax_heatmap.tick_params(
        axis='x',
        which='major',
        length=0.75,
        width=0.35,
        pad=2,
        labelsize=4.2,
        colors='0.2',
        bottom=False,
        top=True,
        labelbottom=False,
        labeltop=True,
        labelrotation=90
    )
    
    clustermap.ax_heatmap.tick_params(
        axis='y',
        which='major',
        length=0.75,
        width=0.35,
        pad=2,
        labelsize=4.2,
        colors='0.2',
        left=True,
        right=False,
        labelleft=True,
        labelright=False
    )
    
    # Remove axis titles
    clustermap.ax_heatmap.set_xlabel('')
    clustermap.ax_heatmap.set_ylabel('')
    
    # Format colorbar
    clustermap.ax_cbar.set_xlabel('Log(count + 1)', fontsize=4.5, labelpad=0.5, color='0.2')
    clustermap.ax_cbar.tick_params(labelsize=4.5, length=0.75, width=0.35, pad=0.5, colors='0.2')
    
    # Save figures
    os.makedirs('plots/heatmaps', exist_ok=True)
    plt.savefig(f'plots/heatmaps/{output_prefix}_heatmap.svg', bbox_inches='tight', format='svg', transparent=True)
    plt.savefig(f'plots/heatmaps/{output_prefix}_heatmap.png', bbox_inches='tight', format='png', dpi=600, transparent=True)
    
    print(f"Heatmap saved: plots/heatmaps/{output_prefix}_heatmap.svg")
    plt.close()


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    plot_human_structure_heatmap(top_n=50)
    plot_rodent_structure_heatmap(top_n=50)