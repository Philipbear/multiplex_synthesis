import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


def prepare_data(masst_pkl_path):
    
    # MASST data
    # final cols: 'name', 'lib_usi', 'mri', 'mri_scan', 'NCBITaxonomy', 'UBERONBodyPartName', 'DOIDCommonName', 'HealthStatus'
    masst_df = pd.read_pickle(masst_pkl_path)

    # remove lib_usis with < 3 matches
    lib_usi_counts = masst_df['lib_usi'].value_counts()
    valid_usis = lib_usi_counts[lib_usi_counts >= 3].index
    masst_df = masst_df[masst_df['lib_usi'].isin(valid_usis)].reset_index(drop=True)

    # shown in humans
    masst_df = masst_df[(masst_df['NCBITaxonomy'].notna()) & (masst_df['NCBITaxonomy'] != '') & (masst_df['NCBITaxonomy'] != 'missing value')]
    
    # remove lib_usis with matches to non-human species
    # first group by lib_usi and get unique NCBITaxonomy
    print('unique lib_usi count:', masst_df['lib_usi'].nunique())
    # len is 1 and it should be ['9606|Homo sapiens']
    masst_df = masst_df.groupby('lib_usi').filter(lambda x: len(x['NCBITaxonomy'].unique()) == 1 and x['NCBITaxonomy'].unique()[0] == '9606|Homo sapiens')
    print('unique lib_usi count after filtering:', masst_df['lib_usi'].nunique())
    
    masst_df = masst_df[masst_df['NCBITaxonomy'] == '9606|Homo sapiens']  # only human matches
    
    # fill empty DOIDCommonName with 'missing value'
    masst_df['DOIDCommonName'] = masst_df['DOIDCommonName'].fillna('missing value')
    # remove missing values in DOIDCommonName
    masst_df = masst_df[masst_df['DOIDCommonName'] != 'missing value'].reset_index(drop=True)
        
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

    # only conjugates
    df = df[df['name'].str.contains('_', case=False)].reset_index(drop=True)
    df['name'] = df['name'].apply(lambda x: x.split(' (known')[0])  # clean names
    df['name'] = df['name'].apply(lambda x: x.replace('5-Aminosalicylic acid', '5-ASA'))  # specific name replacement
    
    # Create pivot table with compound names as rows (y-axis) and body parts as columns (x-axis)
    pivot_data = df.pivot_table(
        index='DOIDCommonName', 
        columns='name', 
        values='match_count', 
        aggfunc='max'  # max match counts
    ).fillna(0)
    
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
                               figsize=(3.7, 1),
                               # Clustering parameters
                               row_cluster=True,
                               col_cluster=True,
                               metric='euclidean',
                               cbar_pos=(-0.05, 0.0, 0.1, 0.05),  # x, y, width, height
                               cbar_kws={'orientation': 'horizontal'},
                               dendrogram_ratio=(0.02, 0.07),
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
    gap = 0.00
    new_heatmap_width = heatmap_pos.width - gap
    new_dendrogram_left = heatmap_pos.x0 + new_heatmap_width + gap
    
    # Set new positions
    clustermap.ax_heatmap.set_position([heatmap_pos.x0, heatmap_pos.y0, 
                                        new_heatmap_width, heatmap_pos.height])
    clustermap.ax_row_dendrogram.set_position([new_dendrogram_left, dendrogram_pos.y0,
                                                dendrogram_pos.width, dendrogram_pos.height])
    
    # Flip the dendrogram to face left
    clustermap.ax_row_dendrogram.invert_xaxis()
    
    # Adjust the position of the column dendrogram
    col_dendrogram_pos = clustermap.ax_col_dendrogram.get_position()
    clustermap.ax_col_dendrogram.set_position([heatmap_pos.x0, col_dendrogram_pos.y0 - 0.06,
                                                col_dendrogram_pos.width, col_dendrogram_pos.height])

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
    clustermap.ax_heatmap.set_xlabel('', fontsize=6, labelpad=2)
    clustermap.ax_heatmap.set_ylabel('', fontsize=6, labelpad=9, rotation=270)

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
    masst_pkl_path = 'target_drugs/5ASA_conjugates/data/5-Aminosalicylic acid_masst_matches.pkl'
    compound_disease_data = prepare_data(masst_pkl_path)
    
    # Save the processed data
    compound_disease_data.to_csv('target_drugs/5ASA_conjugates/data/compound_disease_data.tsv', sep='\t', index=False)

    # Generate and save the heatmap
    plot_disease_heatmap(
        compound_disease_data,
        'target_drugs/5ASA_conjugates/plots/disease_heatmap.svg'
    )