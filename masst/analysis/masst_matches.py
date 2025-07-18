import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles


'''
'lib_usi', 'mri', 'mri_scan', 'lib_scan', 'name', 'inchikey_2d', 'NCBITaxonomy', 'UBERONBodyPartName', 'DOIDCommonName', 'HealthStatus'
'''

def load_masst_data(masst_pkl_path):
    """
    Load the generated MASST data from pickle file
    """
    print(f"Loading MASST data from {masst_pkl_path}...")
    df = pd.read_pickle(masst_pkl_path)
    
    df['name'] = df['name'].apply(lambda x: x.split(' (known')[0])
    print(f"Loaded dataset with {len(df):,} rows")
    
    return df


def analyze_masst_match_distribution(masst_df):
    """
    Analyze MASST match distribution by USI and by structure, saving data files for plotting
    """
    
    print("\n=== Analyzing MASST Match Distribution ===")
    
    # Analyze by USI (lib_usi)
    print("Analyzing matches by USI...")
    usi_match_counts = masst_df.groupby('lib_usi').size().reset_index()
    usi_match_counts.columns = ['lib_usi', 'match_count']
    
    # Get USI names for reference
    usi_names = masst_df[['lib_usi', 'name']].drop_duplicates().set_index('lib_usi')['name'].to_dict()
    usi_match_counts['name'] = usi_match_counts['lib_usi'].map(usi_names)
    
    # Analyze by structure (inchikey_2d)
    print("Analyzing matches by structure...")
    structure_data = masst_df[masst_df['inchikey_2d'].notna()].copy()
    structure_match_counts = structure_data.groupby('inchikey_2d').size().reset_index()
    structure_match_counts.columns = ['inchikey_2d', 'match_count']
    
    # Get structure names for reference
    structure_names = structure_data[['inchikey_2d', 'name']].drop_duplicates().set_index('inchikey_2d')['name'].to_dict()
    structure_match_counts['name'] = structure_match_counts['inchikey_2d'].map(structure_names)
    
    # Calculate statistics
    total_usis = masst_df['lib_usi'].nunique()
    total_structures = masst_df['inchikey_2d'].nunique()
    total_matches = len(masst_df)
    
    # Print statistics
    print(f"Total unique USIs with matches: {total_usis:,}")
    print(f"Total unique structures with matches: {total_structures:,}")
    print(f"Total MASST matches: {total_matches:,}")
    print(f"Average matches per USI: {usi_match_counts['match_count'].mean():.1f}")
    print(f"Median matches per USI: {usi_match_counts['match_count'].median():.1f}")
    print(f"Average matches per structure: {structure_match_counts['match_count'].mean():.1f}")
    print(f"Median matches per structure: {structure_match_counts['match_count'].median():.1f}")
    
    # Save data files for plotting
    usi_match_counts.to_csv("data/usi_match_counts.tsv", sep='\t', index=False)
    structure_match_counts.to_csv("data/structure_match_counts.tsv", sep='\t', index=False)

    return {
        'usi_match_counts': usi_match_counts,
        'structure_match_counts': structure_match_counts
    }


def analyze_masst_repo_distribution(masst_df):
    """
    Analyze the distribution of MASST matches across different repositories
    Generate data for Venn plot analysis by USI and structure
    """

    masst_df_copy = masst_df.copy()
    masst_df_copy['repo'] = masst_df_copy['mri'].apply(lambda x: x.split(':')[0][:2] if ':' in x else 'unknown')
    
    print("\n=== Analyzing MASST Match Distribution by Repository ===")
    
    # === Analysis by USI ===
    # Create presence/absence matrix for USIs (for Venn plots)
    usi_repo_presence = masst_df_copy.groupby(['lib_usi', 'repo']).size().unstack(fill_value=0)
    usi_repo_presence = (usi_repo_presence > 0).astype(int)
    
    # === Analysis by Structure ===
    # Filter out rows without structure information
    structure_data = masst_df_copy[masst_df_copy['inchikey_2d'].notna()].copy()
    
    # Create presence/absence matrix for structures (for Venn plots)
    structure_repo_presence = structure_data.groupby(['inchikey_2d', 'repo']).size().unstack(fill_value=0)
    structure_repo_presence = (structure_repo_presence > 0).astype(int)
    
    # === Save data files for Venn plot analysis ===
    usi_repo_presence.to_csv("data/usi_repo_presence_matrix.tsv", sep='\t', index=True)
    structure_repo_presence.to_csv("data/structure_repo_presence_matrix.tsv", sep='\t', index=True)
    
    return


def perform_analysis_main(masst_pkl_path):
    """
    Main function to perform MASST match distribution analysis
    """
    print("=== Starting MASST Match Distribution Analysis ===")
    
    masst_df = load_masst_data(masst_pkl_path)
    
    # 1. Analyze MASST match distribution
    if os.path.exists("data/usi_match_counts.tsv") and os.path.exists("data/structure_match_counts.tsv"):
        print("Data files already exist, skipping analysis.")
    else:
        analyze_masst_match_distribution(masst_df)
    
    # 2. Analyze repo distribution of all MASST matches
    if os.path.exists("data/usi_repo_presence_matrix.tsv") and os.path.exists("data/structure_repo_presence_matrix.tsv"):
        print("Repo distribution data files already exist, skipping analysis.")
    else:
        analyze_masst_repo_distribution(masst_df)

    print("Analysis completed successfully!")
    
    return


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
    fig, ax = plt.subplots(figsize=(2.5, 1.35))
    
    # Plot scatter plot (same parameters as original)
    plt.scatter(range(len(counts_df)), counts_df['match_count'], 
               alpha=0.5, s=0.8, color='steelblue', edgecolor='none')
    
    # Add log scale for y-axis
    plt.yscale('log')
    
    plt.ylim(bottom=1)
    
    # Add labels (same font size as original)
    if data_type == 'USI':
        plt.xlabel('MS/MS library USI (sorted)', fontsize=6)
    else:
        plt.xlabel('Unique chemical structure (sorted)', fontsize=6)

    plt.ylabel('Number of MASST\nspectral matches', fontsize=6)
    
    # Add tick parameters (exact same as original)
    plt.tick_params(axis='x', which='major', length=1, width=0.8, pad=1,
                    colors='0.2', labelsize=5)
    plt.tick_params(axis='y', which='major', length=1, width=0.8, pad=1,
                    colors='0.2', labelsize=5.5)
    plt.tick_params(axis='y', which='minor', length=0, width=0.8, pad=1,
                    colors='0.2', labelsize=5.5)
    
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
    plt.annotate(stats_text_1, xy=(x_pos, 0.75), xycoords='axes fraction', fontsize=5)
    plt.annotate(stats_text_2, xy=(x_pos, 0.65), xycoords='axes fraction', fontsize=5)
    plt.annotate(stats_text_3, xy=(x_pos, 0.55), xycoords='axes fraction', fontsize=5)

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
    

def print_venn_plot_data():
    """
    Load the generated TSV files and print data needed for Venn plots
    """
    print("\n=== Venn Plot Data ===")
    
    # Load USI repo presence matrix
    print("Loading USI repo presence data...")
    usi_presence = pd.read_csv("data/usi_repo_presence_matrix.tsv", sep='\t', index_col=0)
    
    # Load structure repo presence matrix
    print("Loading structure repo presence data...")
    structure_presence = pd.read_csv("data/structure_repo_presence_matrix.tsv", sep='\t', index_col=0)
    
    # Get repository names
    usi_repos = list(usi_presence.columns)
    structure_repos = list(structure_presence.columns)
    
    print(f"\nRepositories found in USI data: {usi_repos}")
    print(f"Repositories found in structure data: {structure_repos}")
    
    # === USI Venn Plot Data ===
    print("\n=== USI Venn Plot Data ===")
    
    # For each repository, get the set of USIs
    usi_sets = {}
    for repo in usi_repos:
        usi_sets[repo] = set(usi_presence[usi_presence[repo] == 1].index)
        print(f"Repository {repo}: {len(usi_sets[repo]):,} USIs")
    
    # Calculate overlaps for common Venn plot combinations
    if len(usi_repos) >= 2:
        print(f"\nUSI overlaps:")
        for i, repo1 in enumerate(usi_repos):
            for repo2 in usi_repos[i+1:]:
                overlap = len(usi_sets[repo1] & usi_sets[repo2])
                print(f"{repo1} ∩ {repo2}: {overlap:,} USIs")
        
        # If 3 or more repos, show 3-way overlap
        if len(usi_repos) >= 3:
            triple_overlap = len(usi_sets[usi_repos[0]] & usi_sets[usi_repos[1]] & usi_sets[usi_repos[2]])
            print(f"{usi_repos[0]} ∩ {usi_repos[1]} ∩ {usi_repos[2]}: {triple_overlap:,} USIs")
    
    # === Structure Venn Plot Data ===
    print("\n=== Structure Venn Plot Data ===")
    
    # For each repository, get the set of structures
    structure_sets = {}
    for repo in structure_repos:
        structure_sets[repo] = set(structure_presence[structure_presence[repo] == 1].index)
        print(f"Repository {repo}: {len(structure_sets[repo]):,} structures")
    
    # Calculate overlaps for common Venn plot combinations
    if len(structure_repos) >= 2:
        print(f"\nStructure overlaps:")
        for i, repo1 in enumerate(structure_repos):
            for repo2 in structure_repos[i+1:]:
                overlap = len(structure_sets[repo1] & structure_sets[repo2])
                print(f"{repo1} ∩ {repo2}: {overlap:,} structures")
        
        # If 3 or more repos, show 3-way overlap
        if len(structure_repos) >= 3:
            triple_overlap = len(structure_sets[structure_repos[0]] & structure_sets[structure_repos[1]] & structure_sets[structure_repos[2]])
            print(f"{structure_repos[0]} ∩ {structure_repos[1]} ∩ {structure_repos[2]}: {triple_overlap:,} structures")
    
    # === Return data for programmatic use ===
    return {
        'usi_sets': usi_sets,
        'structure_sets': structure_sets,
        'usi_repos': usi_repos,
        'structure_repos': structure_repos
    }
    
    
def plot_venn_plots():
    """
    Generate Venn plots for USI and structure distribution across repositories
    """    
    
    print("\n=== Generating Venn Plots ===")
    
    # Load data
    venn_data = print_venn_plot_data()
    usi_sets = venn_data['usi_sets']
    structure_sets = venn_data['structure_sets']
    usi_repos = venn_data['usi_repos']
    structure_repos = venn_data['structure_repos']
    
    # Set font to Arial
    plt.rcParams['font.family'] = 'Arial'
    
    # === USI Venn Plots ===
    print("\nGenerating USI Venn plots...")
    _create_venn_plot(usi_sets, usi_repos, 'USI', 'plots/usi_venn')
    
    # === Structure Venn Plots ===
    print("\nGenerating structure Venn plots...")
    _create_venn_plot(structure_sets, structure_repos, 'structure', 'plots/structure_venn')
    
    print("Venn plots generated successfully!")


def _create_venn_plot(data_sets, repos, data_type, output_prefix):
    """
    Create Venn plot for 3 repositories
    """
    
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    
    repo1, repo2, repo3 = repos[0], repos[1], repos[2]
    set1, set2, set3 = data_sets[repo1], data_sets[repo2], data_sets[repo3]
    
    # Create Venn diagram
    venn = venn3([set1, set2, set3], set_labels=(repo1, repo2, repo3))
    venn_circles = venn3_circles([set1, set2, set3])
    
    # Customize appearance
    if venn:
        for patch in venn.patches:
            if patch:
                patch.set_alpha(0.7)
        
        # Set colors
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        for i, patch in enumerate(venn.patches):
            if patch:
                patch.set_facecolor(colors[i % 3])
    
    # Customize circles
    for circle in venn_circles:
        circle.set_linewidth(0.8)
        circle.set_edgecolor('0.2')
    
    # # Add title
    # plt.title(f'{data_type.capitalize()} Distribution Across Repositories', fontsize=12, pad=20)
    
    # # Add statistics text
    # total1 = len(set1)
    # total2 = len(set2)
    # total3 = len(set3)
    # overlap_12 = len(set1 & set2)
    # overlap_13 = len(set1 & set3)
    # overlap_23 = len(set2 & set3)
    # overlap_123 = len(set1 & set2 & set3)
    
    # stats_text = (f"Total {data_type}s:\n"
    #              f"{repo1}: {total1:,}\n"
    #              f"{repo2}: {total2:,}\n"
    #              f"{repo3}: {total3:,}\n"
    #              f"3-way overlap: {overlap_123:,}")
    
    # plt.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
    #          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    save_file_svg = f"{output_prefix}_3way.svg"
    
    plt.savefig(save_file_svg, format='svg', bbox_inches='tight', transparent=True)
    
    print(f"3-way Venn plot saved: {save_file_svg}")
    plt.close()

if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    #####
    # on server
    # masst_pkl_path = "/home/shipei/projects/synlib/masst/data/all_masst_matches_with_metadata.pkl"    
    # perform_analysis_main(masst_pkl_path)
    
    #####
    # local, plot
    # plot_masst_match_scatter()
    
    plot_venn_plots()
