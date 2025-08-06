import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


def prepare_data(masst_pkl_path):
    
    # MASST data
    # final cols: 'name', 'lib_usi', 'mri', 'mri_scan', 'NCBITaxonomy', 'UBERONBodyPartName', 'DOIDCommonName', 'HealthStatus'
    masst_df = pd.read_pickle(masst_pkl_path)

    # shown in humans
    masst_df = masst_df[(masst_df['NCBITaxonomy'].notna()) & (masst_df['NCBITaxonomy'] != '') & (masst_df['NCBITaxonomy'] != 'missing value')]
    
    masst_df = masst_df[masst_df['NCBITaxonomy'] == '9606|Homo sapiens']  # only human matches
    
    # fill empty UBERONBodyPartName with 'missing value'
    masst_df['DOIDCommonName'] = masst_df['DOIDCommonName'].fillna('missing value')
        
    # group by lib_usi and aggregate
    masst_df_grouped = masst_df.groupby(['name', 'DOIDCommonName']).agg({
        'mri': 'count'
    }).reset_index()
    masst_df_grouped.rename(columns={'mri': 'match_count'}, inplace=True)
        
    return masst_df_grouped


def plot_disease_heatmap(input_data, output_path):
    """
    Plot heatmap from compound-disease data.

    Args:
        input_data (pd.DataFrame): DataFrame with columns 'name', 'DOIDCommonName', 'match_count'
        output_path (str): Path to save the output SVG file
    """
    
    df = input_data.copy()
    
    # Create pivot table with compound names as rows (y-axis) and body parts as columns (x-axis)
    pivot_data = df.pivot_table(
        index='name', 
        columns='DOIDCommonName', 
        values='match_count', 
        aggfunc='sum'  # Sum match counts
    ).fillna(0)
    
    # Define the desired row order
    desired_row_order = ['Ibuprofen', 'Carnitine', 'Ibuprofen-carnitine', '5-ASA', 'Phenylpropionate', '5-ASA-phenylpropionate']
    
    # Filter to only include compounds that exist in the data and are in the desired order
    available_compounds = [compound for compound in desired_row_order if compound in pivot_data.index]
    
    # Reorder rows according to the specified order
    pivot_data = pivot_data.reindex(available_compounds)
    
    # Convert data to numeric, coercing errors to NaN
    for col in pivot_data.columns:
        pivot_data[col] = pd.to_numeric(pivot_data[col], errors='coerce')
        
    # Only keep rows with at least one non-zero value
    pivot_data = pivot_data.loc[(pivot_data != 0).any(axis=1)]
    
    # Apply log transformation to handle large value ranges (adding 1 to avoid log(0))
    data_log = np.log1p(pivot_data)
        
    # Set Arial font for all text elements
    plt.rcParams['font.family'] = 'Arial'
    
    # Create the custom colormap: white -> blue -> red
    colors = [(1, 1, 1), (0.78, 0.84, 0.94), (0.94, 0.69, 0.65)]  # White -> Blue -> Red
    n_bins = 20
    cmap_name = 'white_blue_red'
    cmap_wbr = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    
    # Create figure with proper tick positioning
    clustermap = sns.clustermap(data_log,
                               cmap=cmap_wbr,
                               linewidths=0.5,
                               annot=False,
                               robust=True,
                               figsize=(5.5, 2.8),
                               # Clustering parameters
                               row_cluster=False,
                               col_cluster=True,
                               metric='euclidean',   
                               cbar_pos=(-0.02, 0.4, 0.1, 0.03),  # x, y, width, height
                               cbar_kws={'orientation': 'horizontal'},
                               dendrogram_ratio=(0.15, 0.15),
                               # Show labels
                               xticklabels=True,
                               yticklabels=True)
    
    # Format tick labels
    plt.setp(clustermap.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=6)
    plt.setp(clustermap.ax_heatmap.get_yticklabels(), rotation=0, va='center', fontsize=6)
    
    # Move dendrogram to the right side
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
    
    # Adjust tick parameters for both axes
    clustermap.ax_heatmap.tick_params(
        axis='x',
        which='major',
        length=0.75,
        width=0.35,
        pad=2,
        labelsize=6,
        colors='0.2'
    )
    
    clustermap.ax_heatmap.tick_params(
        axis='y',
        which='major',
        length=0.75,
        width=0.35,
        pad=2,
        labelsize=6,
        colors='0.2',
        left=True,
        right=False,
        labelleft=True,
        labelright=False
    )
    
    # Set axis titles
    clustermap.ax_heatmap.set_xlabel('Disease', fontsize=6, labelpad=5)
    clustermap.ax_heatmap.set_ylabel('Compounds', fontsize=6, labelpad=9, rotation=270)

    # Add colorbar label
    clustermap.ax_cbar.set_xlabel('Log(Count + 1)', fontsize=6, labelpad=2, color='0.2')
    clustermap.ax_cbar.tick_params(labelsize=6, length=0.75, width=0.35, pad=2, colors='0.2')
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', format='svg', transparent=True)
    # plt.savefig(output_path.replace('.svg', '.png'), bbox_inches='tight', dpi=600, format='png')
    print(f"Heatmap saved to: {output_path}")

    return clustermap


if __name__ == '__main__':
    
    # Prepare the data
    masst_pkl_path = 'target_drugs/all/data/all_masst_matches_with_metadata_0.7_4.pkl'
    compound_disease_data = prepare_data(masst_pkl_path)
    
    # Save the processed data
    compound_disease_data.to_csv('target_drugs/all/data/compound_disease_data.tsv', sep='\t', index=False)

    # Generate and save the heatmap
    plot_disease_heatmap(
        compound_disease_data, 
        'target_drugs/all/plots/disease_heatmap.svg'
    )