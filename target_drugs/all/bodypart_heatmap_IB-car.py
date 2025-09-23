import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


def prepare_data(masst_pkl_path):
    
    # MASST data
    # final cols: 'name', 'lib_usi', 'mri', 'mri_scan', 'SampleType', 'NCBITaxonomy', 'NCBIDivision', 'UBERONBodyPartName', 'HealthStatus'
    masst_df = pd.read_pickle(masst_pkl_path)

    # shown in humans
    masst_df = masst_df[(masst_df['NCBITaxonomy'].notna()) & (masst_df['NCBITaxonomy'] != '') & (masst_df['NCBITaxonomy'] != 'missing value')]
    
    masst_df = masst_df[masst_df['NCBITaxonomy'] == '9606|Homo sapiens']  # only human matches
    
    # fill empty UBERONBodyPartName with 'missing value'
    masst_df['UBERONBodyPartName'] = masst_df['UBERONBodyPartName'].fillna('missing value')
        
    # group by lib_usi and aggregate
    masst_df_grouped = masst_df.groupby(['name', 'UBERONBodyPartName']).agg({
        'mri': 'count'
    }).reset_index()
    masst_df_grouped.rename(columns={'mri': 'match_count'}, inplace=True)
        
    return masst_df_grouped

def plot_bodypart_heatmap(input_data, output_path):
    """
    Plot heatmap from compound-bodypart data.

    Args:
        input_data (pd.DataFrame): DataFrame with columns 'name', 'UBERONBodyPartName', 'match_count'
        output_path (str): Path to save the output SVG file
    """
    
    df = input_data.copy()
    
    # Create pivot table with body parts as rows (y-axis) and compound names as columns (x-axis)
    pivot_data = df.pivot_table(
        index='UBERONBodyPartName',  # Body parts as rows
        columns='name',              # Compounds as columns
        values='match_count', 
        aggfunc='sum'  # Sum match counts
    ).fillna(0)
    
    # Define the desired column order (compounds)
    desired_column_order = ['Ibuprofen', 'Carnitine', 'Ibuprofen-carnitine']
    
    # Filter to only include compounds that exist in the data and are in the desired order
    available_compounds = [compound for compound in desired_column_order if compound in pivot_data.columns]
    
    # Reorder columns according to the specified order
    pivot_data = pivot_data[available_compounds]
    
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
                               figsize=(2.3, 5),  # Adjusted figure size for transposed layout
                               # Clustering parameters
                               row_cluster=True,   # Now cluster body parts (rows)
                               col_cluster=False,  # Don't cluster compounds (columns)
                               metric='euclidean',   
                               cbar_pos=(-0.15, 0.8, 0.03, 0.1),  # Adjusted colorbar position
                               cbar_kws={'orientation': 'vertical'},  # Changed to vertical
                               dendrogram_ratio=(0.15, 0.15),
                               # Show labels
                               xticklabels=True,
                               yticklabels=True)
    
    # Format tick labels
    plt.setp(clustermap.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    plt.setp(clustermap.ax_heatmap.get_yticklabels(), rotation=0, va='center', fontsize=7)
    
    # Adjust tick parameters for both axes
    clustermap.ax_heatmap.tick_params(
        axis='x',
        which='major',
        length=0.75,
        width=0.35,
        pad=2,
        labelsize=7,
        colors='0.2'
    )
    
    clustermap.ax_heatmap.tick_params(
        axis='y',
        which='major',
        length=0.75,
        width=0.35,
        pad=2,
        labelsize=7,
        colors='0.2',
        left=False,
        right=True,
        labelleft=False,
        labelright=True
    )
    
    # Set axis titles
    clustermap.ax_heatmap.set_xlabel('', fontsize=0, labelpad=0)
    clustermap.ax_heatmap.set_ylabel('', fontsize=0, labelpad=0, rotation=90)

    # Add colorbar label
    clustermap.ax_cbar.set_ylabel('Log(Count + 1)', fontsize=7, labelpad=2, color='0.2')
    clustermap.ax_cbar.tick_params(labelsize=7, length=0.75, width=0.35, pad=2, colors='0.2')
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', format='svg', transparent=True)
    print(f"Heatmap saved to: {output_path}")

    return clustermap


if __name__ == '__main__':
    
    # Save the processed data
    compound_bodypart_data = pd.read_csv('target_drugs/all/data/compound_bodypart_data.tsv', sep='\t')

    compound_bodypart_data = compound_bodypart_data[compound_bodypart_data['name'].isin(['Ibuprofen', 'Carnitine', 'Ibuprofen-carnitine'])].reset_index(drop=True)

    # Generate and save the heatmap
    plot_bodypart_heatmap(
        compound_bodypart_data, 
        'target_drugs/all/plots/bodypart_heatmap_IB-car.svg'
    )