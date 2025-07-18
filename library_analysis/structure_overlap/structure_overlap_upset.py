import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import UpSet, from_indicators
import numpy as np
import os


def load_data():
    """
    Load structure overlap data
    """
    print('Loading structure overlap data...')
    df = pd.read_csv('library_analysis/structure_overlap/structure_overlap.tsv', sep='\t', low_memory=False)
    
    df = df[['dbid_SYNTHETIC-COMBINED-LIBRARY_bool', 'dbid_ALL_GNPS_bool', 'dbid_PubChem_bool', 'dbid_PubChem Lite_bool', 'dbid_FoodDB_bool', 'dbid_HMDB_bool', 'dbid_NORMAN_bool']]
    
    df = df.rename(columns={
        'dbid_SYNTHETIC-COMBINED-LIBRARY_bool': 'Synthetic library',
        'dbid_ALL_GNPS_bool': 'GNPS',
        'dbid_PubChem_bool': 'PubChem',
        'dbid_PubChem Lite_bool': 'PubChemLite',
        'dbid_FoodDB_bool': 'FoodDB',
        'dbid_HMDB_bool': 'HMDB',
        'dbid_NORMAN_bool': 'NORMAN'
    })
    
    # Convert to boolean if not already
    for col in df.columns:
        df[col] = df[col].astype(bool)
    
    print(f"Loaded {len(df)} structures across {len(df.columns)} databases")
    print(f"Databases: {list(df.columns)}")
    
    return df


def prepare_upset_data(df):
    """
    Convert structure overlap data to format suitable for UpSet plot
    
    Args:
        df: DataFrame with boolean columns for each database
    
    Returns:
        pandas Series suitable for UpSet plotting
    """
    print('Preparing data for UpSet plot...')
    
    # Remove rows where all values are False (no database contains the structure)
    df_filtered = df[df.any(axis=1)]
    
    # Convert to format expected by upsetplot
    upset_data = from_indicators(df_filtered)
    
    print(f"Found {len(df_filtered)} structures with at least one database match")
    
    return upset_data, df_filtered


def create_upset_plot(upset_data, binary_df, figsize=(6, 1.85), min_subset_size=1):
    """
    Create and save UpSet plot
    """
    print('Creating UpSet plot...')
    
    # Set Arial font and figure size
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = 7
    
    # Create UpSet plot with visible elements and custom styling
    upset = UpSet(
        upset_data,
        subset_size='count',
        intersection_plot_elements=5,  # Show intersection bars
        totals_plot_elements=5,        # Show totals bars
        min_subset_size=min_subset_size,
        show_counts=True,
        sort_by='cardinality',         # Sort by number of sets in intersection
        sort_categories_by='-cardinality',  # Sort categories by total count
        element_size=9,               # Size of dots in matrix
        facecolor='0.4',               # Dot color
    )
    
    # Generate the plot - this returns a dict of axes, not a figure
    axes_dict = upset.plot()
    
    # Get the figure from one of the axes
    fig = list(axes_dict.values())[0].figure
    
    # Now set the figure size
    fig.set_size_inches(figsize)
    
    # Adjust tick parameters and spine line width for bar plots
    for ax_name, ax in axes_dict.items():
        if ax_name == 'intersections':            
            # Intersection bar plot (top)
            ax.tick_params(
                axis='both',
                which='major',
                length=2,      # Tick length
                width=0.8,     # Tick width
                labelsize=6,   # Label size
                color='0.3',   # Tick color
                pad=2          # Distance between tick and label
            )
            # Remove minor ticks if present
            ax.tick_params(axis='both', which='minor', length=0, width=0)
            
            # Add grid
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
            
            # Adjust spine line width for intersection bars
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color('0.4')
            
            # Color bars
            for patch in ax.patches:
                patch.set_facecolor('0.4')      # Bar fill color
                patch.set_edgecolor('0.4')      # Bar edge color
            
            ax.set_ylabel('Intersection\nsize', fontsize=7, labelpad=2)
            
        elif ax_name == 'totals':
            # Totals bar plot (right side)
            ax.tick_params(
                axis='both',
                which='major',
                length=2,
                width=0.8,
                labelsize=6,
                color='0.3',
                pad=2
            )
            ax.tick_params(axis='both', which='minor', length=0, width=0)
            
            # Add grid
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='x')
            
            # Adjust spine line width for totals bars
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color('0.4')
            
            # Color bars
            for patch in ax.patches:
                patch.set_facecolor('#476f95')      # Bar fill color
                patch.set_edgecolor('#476f95')      # Bar edge color
    
    # Format numbers with commas for intersection bars
    if 'intersections' in axes_dict:
        ax = axes_dict['intersections']
        for text in ax.texts:
            text.set_visible(False)
            # current_text = text.get_text()
            # if current_text.isdigit():
            #     formatted_text = f"{int(current_text):,}"
            #     text.set_text(formatted_text)
            #     text.set_fontsize(5)
    
    # Format numbers with commas for totals bars
    if 'totals' in axes_dict:
        ax = axes_dict['totals']
        for text in ax.texts:
            current_text = text.get_text()
            if current_text.isdigit():
                formatted_text = f"{int(current_text):,}"
                text.set_text(formatted_text)
                text.set_fontsize(6.5)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('library_analysis/structure_overlap/plots/structure_overlap_upset.svg', 
                format='svg', bbox_inches='tight', transparent=True)
    
    print('UpSet plot saved to library_analysis/structure_overlap/plots/')
    
    return fig


def analyze_intersections(binary_df, top_n=20):
    """
    Analyze and print the top intersections
    
    Args:
        binary_df: Binary DataFrame
        top_n: Number of top intersections to analyze
    """
    print(f'\nAnalyzing top {top_n} intersections...')
    
    # Count structures per database
    database_counts = binary_df.sum(axis=0).sort_values(ascending=False)
    print(f'\nStructures per database:')
    for db, count in database_counts.items():
        print(f'  {db}: {count:,} structures')
    
    # Find intersections
    upset_data = from_indicators(binary_df)
    
    # Get top intersections
    top_intersections = upset_data.sort_values(ascending=False).head(top_n)
    
    print(f'\nTop {top_n} intersections:')
    
    # Get database names from the binary_df columns
    database_names = list(binary_df.columns)
    
    for intersection, count in top_intersections.items():
        # intersection is a tuple of boolean values corresponding to each database
        databases = [database_names[i] for i, present in enumerate(intersection) if present]
        
        if len(databases) == 1:
            print(f'  {databases[0]} only: {count:,} structures')
        else:
            databases_str = ' ∩ '.join(databases)
            print(f'  {databases_str}: {count:,} structures')
    
    return top_intersections


def create_summary_statistics(binary_df):
    """
    Create summary statistics for the dataset
    """
    print('\n=== Summary Statistics ===')
    
    total_structures = len(binary_df)
    total_databases = len(binary_df.columns)
    
    print(f'Total structures: {total_structures:,}')
    print(f'Total databases: {total_databases}')
    
    # Structures found in multiple databases
    structures_per_database = binary_df.sum(axis=1)
    multi_database_structures = (structures_per_database > 1).sum()
    single_database_structures = (structures_per_database == 1).sum()
    
    print(f'Structures found in multiple databases: {multi_database_structures:,} ({multi_database_structures/total_structures*100:.1f}%)')
    print(f'Structures found in single database: {single_database_structures:,} ({single_database_structures/total_structures*100:.1f}%)')
    
    # Most ubiquitous structures
    max_databases = structures_per_database.max()
    most_ubiquitous = structures_per_database[structures_per_database == max_databases].index
    print(f'Most ubiquitous structure(s) found in {max_databases} databases: {len(most_ubiquitous)} structure(s)')


def main():
    """
    Main function to run the complete analysis
    """
    # Create output directory
    os.makedirs('library_analysis/structure_overlap/plots', exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Prepare data for UpSet plot
    upset_data, binary_df = prepare_upset_data(df)
    
    # Create summary statistics
    create_summary_statistics(binary_df)
    
    # Analyze intersections
    top_intersections = analyze_intersections(binary_df)
    
    # Create UpSet plot
    fig = create_upset_plot(upset_data, binary_df)
    
    print('\nAnalysis complete! All plots saved to library_analysis/structure_overlap/plots/')
    
    return df, binary_df, upset_data, top_intersections


if __name__ == '__main__':
    # Run the complete analysis
    df, binary_df, upset_data, top_intersections = main()