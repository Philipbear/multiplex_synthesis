import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial import distance


def load_masst_data(masst_pkl_path):
    """
    Load the generated MASST data from pickle file
    """
    print(f"Loading MASST data from {masst_pkl_path}...")
    df = pd.read_pickle(masst_pkl_path)
    
    df['name'] = df['name'].apply(lambda x: x.split('(known')[0])
    print(f"Loaded dataset with {len(df):,} rows")
    return df


def get_microbemasst_datasets(file_path='masst/analysis/data/microbe_masst_table.csv'):
    """
    Read the microbemasst datasets from a tsv file.
    """
    df = pd.read_csv(file_path)
    return df['MassIVE'].unique().tolist()


def create_output_directory(base_dir):
    """
    Create directory structure for output plots
    """
    dirs = {
        'main': base_dir,
        'bodypart': f"{base_dir}/bodypart",
        'disease': f"{base_dir}/disease",
        'health': f"{base_dir}/health",
        'microbe': f"{base_dir}/microbe"
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def analyze_microbe_distribution(masst_df, output_dirs, 
                               microbe_table_path='masst/analysis/data/microbe_masst_table.csv'):
    """
    Analyze microbeMASST matches distribution and save the data files for later plotting
    """
    print("\n=== Analyzing MicrobeMASST Distribution ===")
    
    # Create a copy of the dataframe
    df = masst_df.copy()
    
    # Get microbeMASST dataset IDs
    microbe_dataset_ids = get_microbemasst_datasets(microbe_table_path)
    print(f"Found {len(microbe_dataset_ids)} microbeMASST datasets")
    
    # Extract dataset IDs from MRI
    df['dataset_id'] = df['mri'].apply(lambda x: x.split(':')[0])
    
    # Flag microbe matches
    df['is_microbe_match'] = df['dataset_id'].isin(microbe_dataset_ids)
    
    print(f"Total matches: {len(df)}")
    print(f"Microbe matches: {df['is_microbe_match'].sum()}")
    
    # Count microbe matches per lib_usi
    usi_microbe_counts = df[df['is_microbe_match']].groupby('lib_usi').size().reset_index()
    usi_microbe_counts.columns = ['lib_usi', 'microbe_match_count']

    # Count microbe matches per structure (inchikey_2d)
    structure_microbe_counts = df[(df['is_microbe_match']) & (df['inchikey_2d'].notna())].groupby('inchikey_2d').size().reset_index()
    structure_microbe_counts.columns = ['inchikey_2d', 'microbe_match_count']
    
    # Get USI names for reference
    usi_names = df[['lib_usi', 'name']].drop_duplicates().set_index('lib_usi')['name'].to_dict()
    usi_microbe_counts['name'] = usi_microbe_counts['lib_usi'].map(usi_names)

    # Get structure names for reference
    structure_names = df[['inchikey_2d', 'name']].drop_duplicates().set_index('inchikey_2d')['name'].to_dict()
    structure_microbe_counts['name'] = structure_microbe_counts['inchikey_2d'].map(structure_names)

    # Calculate total counts of USIs and structures
    total_usis = df['lib_usi'].nunique()
    total_structures = df['inchikey_2d'].nunique()

    # Calculate percentage of USIs and structures with microbe matches
    usis_with_microbe = usi_microbe_counts['lib_usi'].nunique()
    structures_with_microbe = structure_microbe_counts['inchikey_2d'].nunique()

    print(f"Library USIs with microbe matches: {usis_with_microbe:,}/{total_usis:,} ({usis_with_microbe/total_usis*100:.1f}%)")
    print(f"Unique structures with microbe matches: {structures_with_microbe:,}/{total_structures:,} ({structures_with_microbe/total_structures*100:.1f}%)")
    
    # Save match counts and metadata to files for later plotting
    usi_microbe_counts.to_csv(f"{output_dirs['microbe']}/usi_microbe_match_counts.tsv", sep='\t', index=False)
    structure_microbe_counts.to_csv(f"{output_dirs['microbe']}/structure_microbe_match_counts.tsv", sep='\t', index=False)
    
    # Save metadata about totals for plotting
    microbe_metadata = {
        'total_usis': total_usis,
        'total_structures': total_structures,
        'usis_with_microbe': usis_with_microbe,
        'structures_with_microbe': structures_with_microbe
    }
    
    pd.DataFrame([microbe_metadata]).to_csv(f"{output_dirs['microbe']}/microbe_analysis_metadata.tsv", sep='\t', index=False)
    
    return {
        'usi_microbe_counts': usi_microbe_counts,
        'structure_microbe_counts': structure_microbe_counts,
        'total_usis': total_usis,
        'total_structures': total_structures,
        'usis_with_microbe': usis_with_microbe,
        'structures_with_microbe': structures_with_microbe
    }


def create_microbe_match_distribution(counts_df, total_items, items_with_matches, title_suffix, output_prefix):
    """
    Create distribution plots showing how many microbeMASST matches each item has
    Using log scale on y-axis to better visualize the distribution
    """
    
    # Create histogram with log scale y-axis
    plt.figure(figsize=(10, 6))
    
    # Plot histogram with a logarithmic y-scale for better visualization
    ax = sns.histplot(counts_df['microbe_match_count'], bins=50, kde=False, color='tomato')
    ax.set_yscale('log')
    
    plt.title(f'Distribution of MicrobeMASST Matches per {title_suffix}', fontsize=16)
    plt.xlabel('Number of MicrobeMASST Matches', fontsize=14)
    plt.ylabel('Frequency (log scale)', fontsize=14)
    
    # Add text annotation for items with no matches
    items_without_matches = total_items - items_with_matches
    plt.figtext(0.7, 0.8, f"Items with no matches: {items_without_matches:,}\n({items_without_matches/total_items*100:.1f}% of total)", 
               bbox=dict(facecolor='white', alpha=0.8))
    
    # Add statistics
    plt.figtext(0.7, 0.7, 
               f"Mean: {counts_df['microbe_match_count'].mean():.1f}\n"
               f"Median: {counts_df['microbe_match_count'].median():.0f}\n"
               f"Max: {counts_df['microbe_match_count'].max():,}",
               bbox=dict(facecolor='white', alpha=0.8))
    
    # Add grid lines for better readability with log scale
    plt.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_prefix}.svg", format='svg', bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved to {output_prefix}.png")


def create_microbe_presence_pie(with_microbe, without_microbe, title_suffix, output_file):
    """
    Create a pie chart showing the proportion of items with/without microbe matches
    """
    plt.figure(figsize=(10, 8))
    
    # Create labels
    total = with_microbe + without_microbe
    
    labels = [
        f'With Microbe Matches\n{with_microbe:,} ({with_microbe/total*100:.1f}%)',
        f'Without Microbe Matches\n{without_microbe:,} ({without_microbe/total*100:.1f}%)'
    ]
    
    # Create the pie chart
    plt.pie(
        [with_microbe, without_microbe], 
        labels=labels,
        colors=['tomato', '#FFC107'],
        autopct='%1.1f%%',
        startangle=90,
        explode=[0.05, 0],
        shadow=False,
        textprops={'fontsize': 14}
    )
    
    plt.title(f'Distribution of {title_suffix} with Microbe Matches', fontsize=16)
    plt.tight_layout()
    
    # Save figures
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print(f"Pie chart saved to {output_file}")


def create_top_items_plot(data_df, name_col, count_col, title, output_file, x_label='Count'):
    """
    Create a horizontal bar plot for top items by microbe match count
    """
    if data_df.empty:
        print(f"Warning: No data available for plot {title}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 15))
    
    # Create the bar chart
    bars = plt.barh(data_df[name_col], data_df[count_col], color='tomato')
    
    plt.title(title, fontsize=10)
    plt.xlabel(x_label, fontsize=8)
    plt.ylabel('Compound Name', fontsize=8)
    
    ax.tick_params(axis='y', labelsize=6)
    ax.tick_params(axis='x', labelsize=6)
    ax.margins(y=0.01)
    
    # Add count labels to bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{int(width):,}', ha='left', va='center', fontsize=6)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Bar plot saved to {output_file}")


def generate_heatmap(data_df, category_col, count_col, entity_col, title, output_file, 
                    top_n_entities=150, top_n_categories=30, color_palette='viridis'):
    """
    Generate a heatmap showing top compounds across different categories
    
    Parameters:
    -----------
    data_df: DataFrame with the raw data
    category_col: Column name for categories (e.g., 'UBERONBodyPartName', 'DOIDCommonName')
    count_col: Column name for count values
    entity_col: Column name for entities (e.g., 'inchikey_2d' or 'name')
    title: Title for the plot
    output_file: Output file path
    top_n_entities: Number of top entities to include
    top_n_categories: Number of top categories to include
    color_palette: Color palette for the heatmap
    """
    if data_df.empty:
        print(f"Warning: No data available for heatmap {title}")
        return
    
    # Get total counts by entity across all categories to find top entities
    entity_totals = data_df.groupby(entity_col)[count_col].sum().sort_values(ascending=False)
    top_entities = entity_totals.head(top_n_entities).index.tolist()
    
    # Filter for top entities
    filtered_df = data_df[data_df[entity_col].isin(top_entities)]
    
    # Get top categories by total count
    category_totals = filtered_df.groupby(category_col)[count_col].sum().sort_values(ascending=False)
    top_categories = category_totals.head(top_n_categories).index.tolist()
    
    # Filter for top categories
    filtered_df = filtered_df[filtered_df[category_col].isin(top_categories)]
    
    # Pivot data for heatmap
    pivot_df = filtered_df.pivot_table(
        index=category_col, 
        columns=entity_col, 
        values=count_col, 
        fill_value=0
    )
    
    # Sort rows and columns by total values
    pivot_df = pivot_df.loc[pivot_df.sum(axis=1).sort_values(ascending=False).index]
    pivot_df = pivot_df[pivot_df.sum().sort_values(ascending=False).index]
    
    # Apply log transformation to better visualize differences (log(x+1) to handle zeros)
    log_data = np.log1p(pivot_df)
    
    # Determine figure size based on number of categories and entities
    fig_width = 20
    fig_height = 16
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # Create the heatmap
    ax = sns.heatmap(
        log_data,
        cmap=color_palette,
        linewidth=0.5,
        cbar_kws={'label': 'log(count + 1)'}
    )
    
    plt.title(title, fontsize=8)
    plt.tight_layout()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(fontsize=5)
    
    # Save the heatmap
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {output_file}")
    
    # Create clustered version
    try:
        plt.figure(figsize=(fig_width, fig_height))
        
        # Calculate linkage for clustering
        row_linkage = hierarchy.linkage(distance.pdist(log_data.values), method='average')
        col_linkage = hierarchy.linkage(distance.pdist(log_data.values.T), method='average')
        
        # Create clustered heatmap
        cluster_grid = sns.clustermap(
            log_data,
            cmap=color_palette,
            figsize=(fig_width, fig_height),
            row_linkage=row_linkage,
            col_linkage=col_linkage,
            cbar_kws={'label': 'log(count + 1)'},
            xticklabels=True,
            yticklabels=True
        )
        
        # Adjust labels
        cluster_grid.ax_heatmap.set_xticklabels(
            cluster_grid.ax_heatmap.get_xticklabels(),
            rotation=90,
            fontsize=5
        )
        cluster_grid.ax_heatmap.set_yticklabels(
            cluster_grid.ax_heatmap.get_yticklabels(),
            fontsize=5
        )
        
        # Add title
        plt.suptitle(f"Clustered {title}", fontsize=8, y=0.95)
        
        # Save clustered heatmap
        clustered_output = output_file.replace('.png', '_clustered.png')
        cluster_grid.savefig(clustered_output, dpi=300, bbox_inches='tight')
        cluster_grid.savefig(clustered_output.replace('.png', '.svg'), format='svg', bbox_inches='tight')
        plt.close()
        print(f"Clustered heatmap saved to {clustered_output}")
        
    except Exception as e:
        print(f"Error creating clustered heatmap: {e}")
    
    return pivot_df


def analyze_bodypart_distribution(masst_df, output_dirs):
    """
    Analyze bodypart distribution for human and rodent samples
    """
    print("\n=== Analyzing Body Part Distribution ===")
    
    # Create a copy of the dataframe
    df = masst_df.copy()
    
    # Filter out rows without bodypart information
    df = df[df['UBERONBodyPartName'].notna() & df['NCBITaxonomy'].notna() & (df['UBERONBodyPartName'] != 'missing value')]
    
    # Separate human and rodent data
    human_df = df[df['NCBITaxonomy'] == '9606|Homo sapiens']
    rodent_df = df[df['NCBITaxonomy'].isin(["10088|Mus", "10090|Mus musculus", "10105|Mus minutoides", "10114|Rattus", "10116|Rattus norvegicus"])]
    
    print(f"Found {len(human_df):,} human matches and {len(rodent_df):,} rodent matches with body part information")
    
    # Analysis by USI
    print("Analyzing USI distribution by body part...")
    human_usi_counts = human_df.groupby('UBERONBodyPartName')['lib_usi'].nunique().reset_index()
    human_usi_counts = human_usi_counts.rename(columns={'lib_usi': 'count'}).sort_values('count', ascending=False)

    rodent_usi_counts = rodent_df.groupby('UBERONBodyPartName')['lib_usi'].nunique().reset_index()
    rodent_usi_counts = rodent_usi_counts.rename(columns={'lib_usi': 'count'}).sort_values('count', ascending=False)

    # Analysis by unique structures
    print("Analyzing unique structure distribution by body part...")
    human_structure_counts = human_df[human_df['inchikey_2d'].notna()].groupby('UBERONBodyPartName')['inchikey_2d'].nunique().reset_index()
    human_structure_counts = human_structure_counts.rename(columns={'inchikey_2d': 'count'}).sort_values('count', ascending=False)
    
    rodent_structure_counts = rodent_df[rodent_df['inchikey_2d'].notna()].groupby('UBERONBodyPartName')['inchikey_2d'].nunique().reset_index()
    rodent_structure_counts = rodent_structure_counts.rename(columns={'inchikey_2d': 'count'}).sort_values('count', ascending=False)
    
    # Save data files
    human_usi_counts.to_csv(f"{output_dirs['bodypart']}/human_bodypart_usi_counts.tsv", sep='\t', index=False)
    rodent_usi_counts.to_csv(f"{output_dirs['bodypart']}/rodent_bodypart_usi_counts.tsv", sep='\t', index=False)
    human_structure_counts.to_csv(f"{output_dirs['bodypart']}/human_bodypart_structure_counts.tsv", sep='\t', index=False)
    rodent_structure_counts.to_csv(f"{output_dirs['bodypart']}/rodent_bodypart_structure_counts.tsv", sep='\t', index=False)
    
    # Create raw dataframes for heatmap generation
    # Count number of MASST matches for each compound-bodypart pair
    human_compound_bodypart_counts = human_df.groupby(['UBERONBodyPartName', 'name']).size().reset_index(name='count')
    human_compound_bodypart_counts.to_csv(f"{output_dirs['bodypart']}/human_compound_bodypart_counts.tsv", sep='\t', index=False)
    
    # Do the same for rodent data
    rodent_compound_bodypart_counts = rodent_df.groupby(['UBERONBodyPartName', 'name']).size().reset_index(name='count')
    rodent_compound_bodypart_counts.to_csv(f"{output_dirs['bodypart']}/rodent_compound_bodypart_counts.tsv", sep='\t', index=False)
    
    return {
        'human_usi_counts': human_usi_counts,
        'rodent_usi_counts': rodent_usi_counts,
        'human_structure_counts': human_structure_counts,
        'rodent_structure_counts': rodent_structure_counts
    }


def analyze_disease_distribution(masst_df, output_dirs):
    """
    Analyze disease distribution in MASST matches
    """
    print("\n=== Analyzing Disease Distribution ===")
    
    # Create a copy of the dataframe
    df = masst_df.copy()
    
    # Filter for valid disease information
    df = df[df['DOIDCommonName'].notna() & 
           (df['DOIDCommonName'] != 'missing value')]
    
    print(f"Found {len(df):,} matches with disease information")
    
    # Analysis by USI
    print("Analyzing USI distribution by disease...")
    disease_usi_counts = df.groupby('DOIDCommonName')['lib_usi'].nunique().reset_index()
    disease_usi_counts = disease_usi_counts.rename(columns={'lib_usi': 'count'}).sort_values('count', ascending=False)

    # Analysis by unique structures
    print("Analyzing unique structure distribution by disease...")
    disease_structure_counts = df[df['inchikey_2d'].notna()].groupby('DOIDCommonName')['inchikey_2d'].nunique().reset_index()
    disease_structure_counts = disease_structure_counts.rename(columns={'inchikey_2d': 'count'}).sort_values('count', ascending=False)
     
    # Save data files
    disease_usi_counts.to_csv(f"{output_dirs['disease']}/disease_usi_counts.tsv", sep='\t', index=False)
    disease_structure_counts.to_csv(f"{output_dirs['disease']}/disease_structure_counts.tsv", sep='\t', index=False)
       
    # Create raw data for heatmap - count MASST matches per compound-disease pair
    compound_disease_counts = df.groupby(['DOIDCommonName', 'name']).size().reset_index(name='count')
    compound_disease_counts.to_csv(f"{output_dirs['disease']}/compound_disease_counts.tsv", sep='\t', index=False)
    
    return {
        'disease_usi_counts': disease_usi_counts,
        'disease_structure_counts': disease_structure_counts
    }


def analyze_health_status(masst_df, output_dirs):
    """
    Analyze health status distribution in MASST matches
    """
    print("\n=== Analyzing Health Status Distribution ===")
    
    # Create a copy of the dataframe
    df = masst_df.copy()
    
    # Filter for valid health status information
    df = df[df['HealthStatus'].notna() & (df['HealthStatus'] != 'missing value')]
    
    print(f"Found {len(df):,} matches with health status information")
    
    # Analysis by USI
    print("Analyzing USI distribution by health status...")
    health_usi_counts = df.groupby('HealthStatus')['lib_usi'].nunique().reset_index()
    health_usi_counts = health_usi_counts.rename(columns={'lib_usi': 'count'}).sort_values('count', ascending=False)
    
    # Analysis by unique structures
    print("Analyzing unique structure distribution by health status...")
    health_structure_counts = df[df['inchikey_2d'].notna()].groupby('HealthStatus')['inchikey_2d'].nunique().reset_index()
    health_structure_counts = health_structure_counts.rename(columns={'inchikey_2d': 'count'}).sort_values('count', ascending=False)
    
    # Save data files
    health_usi_counts.to_csv(f"{output_dirs['health']}/health_status_usi_counts.tsv", sep='\t', index=False)
    health_structure_counts.to_csv(f"{output_dirs['health']}/health_status_structure_counts.tsv", sep='\t', index=False)
    
    # Create raw data for heatmap - count MASST matches per compound-health status pair
    compound_health_counts = df.groupby(['HealthStatus', 'name']).size().reset_index(name='count')
    compound_health_counts.to_csv(f"{output_dirs['health']}/compound_health_counts.tsv", sep='\t', index=False)
    
    return {
        'health_usi_counts': health_usi_counts,
        'health_structure_counts': health_structure_counts
    }
    

def generate_top_bar_plot(data_df, category_col, count_col, title, output_file, x_label='Count', top_n=None, color='tomato'):
    """
    Generate a horizontal bar plot for top categories
    """
    if data_df.empty:
        print(f"Warning: No data available for plot {title}")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Sort and get top N if specified
    plot_df = data_df.sort_values(count_col, ascending=False)
    if top_n:
        plot_df = plot_df.head(top_n)
    
    # Create the bar chart
    bars = plt.barh(plot_df[category_col], plot_df[count_col], color=color)
    
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(category_col.replace('UBERON', '').replace('DOID', '').replace('Name', ''), fontsize=14)
    
    # Add count labels to bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{int(width):,}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Bar plot saved to {output_file}")


def run_comprehensive_analysis(masst_pkl_path, output_base_dir="plots", 
                               microbe_table_path='masst/analysis/data/microbe_masst_table.csv'):
    """
    Run comprehensive analysis on MASST data
    """
    # Load the MASST data
    masst_df = load_masst_data(masst_pkl_path)
    
    # some prefiltering
    # only consider conjugated products, remove starting materials
    masst_df = masst_df[masst_df['name'].str.contains('_', na=False)]
    
    # Create output directories
    output_dirs = create_output_directory(output_base_dir)
    
    # Run analyses
    bodypart_results = analyze_bodypart_distribution(masst_df, output_dirs)
    disease_results = analyze_disease_distribution(masst_df, output_dirs)
    health_results = analyze_health_status(masst_df, output_dirs)    
    microbe_results = analyze_microbe_distribution(masst_df, output_dirs, microbe_table_path)
    
    print("\n=== Comprehensive Analysis Complete ===")
    
    return {
        'bodypart': bodypart_results,
        'disease': disease_results,
        'health': health_results,
        'microbe': microbe_results
    }


def make_all_plots():
    """
    Generate all plots for the analysis using saved data files
    """
    print("Generating all plots...")
    # load all files needed for the plots
    output_dirs = create_output_directory('plots')
    
    print(" === Body Part Plots ===")
    # Load data files
    human_usi_counts = pd.read_csv(f"{output_dirs['bodypart']}/human_bodypart_usi_counts.tsv", sep='\t')
    rodent_usi_counts = pd.read_csv(f"{output_dirs['bodypart']}/rodent_bodypart_usi_counts.tsv", sep='\t')
    human_structure_counts = pd.read_csv(f"{output_dirs['bodypart']}/human_bodypart_structure_counts.tsv", sep='\t')
    rodent_structure_counts = pd.read_csv(f"{output_dirs['bodypart']}/rodent_bodypart_structure_counts.tsv", sep='\t')
    human_compound_bodypart_counts = pd.read_csv(f"{output_dirs['bodypart']}/human_compound_bodypart_counts.tsv", sep='\t')
    rodent_compound_bodypart_counts = pd.read_csv(f"{output_dirs['bodypart']}/rodent_compound_bodypart_counts.tsv", sep='\t')
        
    # Generate plots
    generate_top_bar_plot(
        human_usi_counts.head(20),
        'UBERONBodyPartName',
        'count',
        'Top 20 Human Body Parts by MS/MS Spectra',
        f"{output_dirs['bodypart']}/human_bodypart_scan_distribution.png",
        x_label='Number of MS/MS Spectra'
    )

    generate_top_bar_plot(
        rodent_usi_counts.head(20),
        'UBERONBodyPartName',
        'count',
        'Top 20 Rodent Body Parts by MS/MS Spectra',
        f"{output_dirs['bodypart']}/rodent_bodypart_scan_distribution.png",
        x_label='Number of MS/MS Spectra'
    )
    
    generate_top_bar_plot(
        human_structure_counts.head(20), 
        'UBERONBodyPartName', 
        'count',
        'Top 20 Human Body Parts by Unique Structures',
        f"{output_dirs['bodypart']}/human_bodypart_structure_distribution.png",
        x_label='Number of Unique Structures' 
    )
    
    generate_top_bar_plot(
        rodent_structure_counts.head(20), 
        'UBERONBodyPartName', 
        'count',
        'Top 20 Rodent Body Parts by Unique Structures',
        f"{output_dirs['bodypart']}/rodent_bodypart_structure_distribution.png",
        x_label='Number of Unique Structures'
    )
    
    generate_heatmap(
        human_compound_bodypart_counts,
        'UBERONBodyPartName', 'count', 'name',
        'Human Body Parts vs Top Compounds',
        f"{output_dirs['bodypart']}/human_bodypart_compound_heatmap.png"
    )
    
    generate_heatmap(
        rodent_compound_bodypart_counts,
        'UBERONBodyPartName', 'count', 'name',
        'Rodent Body Parts vs Top Compounds',
        f"{output_dirs['bodypart']}/rodent_bodypart_compound_heatmap.png",
    )
    
    print(" === Disease Plots ===")
    # Load disease data files
    disease_usi_counts = pd.read_csv(f"{output_dirs['disease']}/disease_usi_counts.tsv", sep='\t')
    disease_structure_counts = pd.read_csv(f"{output_dirs['disease']}/disease_structure_counts.tsv", sep='\t')
    compound_disease_counts = pd.read_csv(f"{output_dirs['disease']}/compound_disease_counts.tsv", sep='\t')
    
    # Generate disease plots
    generate_top_bar_plot(
        disease_usi_counts.head(20),
        'DOIDCommonName',
        'count',
        'Top 20 Diseases by MS/MS Spectra',
        f"{output_dirs['disease']}/disease_scan_distribution.png",
        x_label='Number of MS/MS Spectra'
    )
    
    generate_top_bar_plot(
        disease_structure_counts.head(20),
        'DOIDCommonName',
        'count',
        'Top 20 Diseases by Unique Structures',
        f"{output_dirs['disease']}/disease_structure_distribution.png",
        x_label='Number of Unique Structures'
    )
    
    generate_heatmap(
        compound_disease_counts,
        'DOIDCommonName', 'count', 'name',
        'Diseases vs Top Compounds',
        f"{output_dirs['disease']}/disease_compound_heatmap.png"
    )
    
    print(" === Health Status Plots ===")
    # Load health status data files
    health_usi_counts = pd.read_csv(f"{output_dirs['health']}/health_status_usi_counts.tsv", sep='\t')
    health_structure_counts = pd.read_csv(f"{output_dirs['health']}/health_status_structure_counts.tsv", sep='\t')
    compound_health_counts = pd.read_csv(f"{output_dirs['health']}/compound_health_counts.tsv", sep='\t')
    
    # Generate health status plots
    generate_top_bar_plot(
        health_usi_counts,
        'HealthStatus',
        'count',
        'Health Status Distribution by MS/MS Spectra',
        f"{output_dirs['health']}/health_status_usi_distribution.png",
        x_label='Number of MS/MS Spectra'
    )
    
    generate_top_bar_plot(
        health_structure_counts,
        'HealthStatus',
        'count',
        'Health Status Distribution by Unique Structures',
        f"{output_dirs['health']}/health_status_structure_distribution.png",
        x_label='Number of Unique Structures'
    )
    
    generate_heatmap(
        compound_health_counts,
        'HealthStatus', 'count', 'name',
        'Health Status vs Top Compounds',
        f"{output_dirs['health']}/health_status_compound_heatmap.png"
    )
    
    print(" === Generating Microbe Plots ===")    
    # Load microbe data files and metadata
    usi_microbe_counts = pd.read_csv(f"{output_dirs['microbe']}/usi_microbe_match_counts.tsv", sep='\t')
    structure_microbe_counts = pd.read_csv(f"{output_dirs['microbe']}/structure_microbe_match_counts.tsv", sep='\t')
    metadata = pd.read_csv(f"{output_dirs['microbe']}/microbe_analysis_metadata.tsv", sep='\t').iloc[0].to_dict()
    total_usis = metadata.get('total_usis')
    total_structures = metadata.get('total_structures')
    usis_with_microbe = metadata.get('usis_with_microbe')
    structures_with_microbe = metadata.get('structures_with_microbe')
    
    # Generate bar plots for top items
    # create_top_items_plot(
    #     usi_microbe_counts.sort_values('microbe_match_count', ascending=False).head(20),
    #     'name', 'microbe_match_count',
    #     'Top 20 MS/MS Spectra by Microbe Matches',
    #     f"{output_dirs['microbe']}/top_scan_microbe_matches.png",
    #     x_label='Number of MicrobeMASST Matches'
    # )
    
    create_top_items_plot(
        structure_microbe_counts.sort_values('microbe_match_count', ascending=False).head(100),
        'name', 'microbe_match_count',
        'Top Structures by Microbe Matches',
        f"{output_dirs['microbe']}/top_structure_microbe_matches.png",
        x_label='Number of MicrobeMASST Matches'
    )
    
    # Create distribution plots
    create_microbe_match_distribution(
        usi_microbe_counts,
        total_usis,
        usis_with_microbe,
        'MS/MS Library USIs',
        f"{output_dirs['microbe']}/usi_microbe_match_distribution"
    )
    
    create_microbe_match_distribution(
        structure_microbe_counts,
        total_structures,
        structures_with_microbe,
        'Unique Molecular Structures',
        f"{output_dirs['microbe']}/structure_microbe_match_distribution"
    )
    
    # Create pie charts
    create_microbe_presence_pie(
        usis_with_microbe,
        total_usis - usis_with_microbe,
        'MS/MS Library USIs',
        f"{output_dirs['microbe']}/usi_microbe_presence_pie.png"
    )
    
    create_microbe_presence_pie(
        structures_with_microbe, 
        total_structures - structures_with_microbe,
        'Unique Molecular Structures', 
        f"{output_dirs['microbe']}/structure_microbe_presence_pie.png"
    )
    
    print("All plots generated successfully!")



if __name__ == "__main__":
    # File paths
    masst_pkl_path = '/home/shipei/projects/synlib/masst/data/all_masst_matches_with_metadata.pkl'
    output_dir = 'plots'
    microbe_table_path = '/home/shipei/projects/microbe_masst/sql/microbe_masst_table.csv'
    
    # Run the analysis
    results = run_comprehensive_analysis(
        masst_pkl_path, 
        output_dir,
        microbe_table_path
    )
    
    make_all_plots()
    
    print("Analysis complete.")