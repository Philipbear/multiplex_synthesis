import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from tqdm import tqdm


def generate_match_counts(input_file='masst/analysis/data/all_masst_matches.tsv', 
                          output_file='masst/analysis/data/match_counts.csv'):
    """
    Generate and save counts of matches per query scan.
    
    Args:
        input_file: Path to the TSV file with all MASST matches
        output_file: Path to save the match count data
    
    Returns:
        pd.Series: Match counts per scan
    """
    print(f"Loading data from {input_file}...")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the data in chunks to handle large files
    chunk_size = 500000
    reader = pd.read_csv(input_file, sep='\t', chunksize=chunk_size)
    
    # Use a dictionary to count matches per scan
    scan_counts = {}
    
    for chunk in tqdm(reader, desc="Processing chunks"):
        chunk = chunk[chunk['dataset'] != 'MSV000094559']
            
        # Count matches per scan in this chunk
        for scan, count in chunk['scan'].value_counts().items():
            if scan in scan_counts:
                scan_counts[scan] += count
            else:
                scan_counts[scan] = count
    
    # Convert counts to a Series for easier saving
    counts_series = pd.Series(scan_counts, name="match_count")
    counts_df = counts_series.reset_index()
    counts_df.columns = ['scan', 'match_count']
    
    # Save the counts to CSV
    counts_df.to_csv(output_file, index=False)
    print(f"Match counts saved to {output_file}")
    
    # Print summary statistics
    print(f"Found {len(counts_series)} scans with matches")
    print(f"Average matches per scan: {counts_series.mean():.1f}")
    print(f"Median matches per scan: {counts_series.median():.1f}")
    print(f"Maximum matches per scan: {counts_series.max()}")
    
    return counts_series


def plot_match_distribution(input_file='masst/analysis/data/match_counts.csv', 
                            save_path='masst/analysis/plot', total_success_masst=128515):
    """
    Create a histogram of the number of matches per query scan from precomputed data.
    
    Args:
        input_file: Path to the CSV file with match counts
        save_path: Directory to save the output plots
    
    Returns:
        pd.Series: Match counts per scan
    """
    print(f"Loading match counts from {input_file}...")
    
    # Load precomputed match counts
    counts_df = pd.read_csv(input_file)
    counts_series = pd.Series(counts_df['match_count'].values, index=counts_df['scan'])
    
    # Create output directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Set up the plot
    plt.figure(figsize=(8, 5))
    
    # Calculate statistics
    total_scans = len(counts_series)
    
    # Calculate spectra with 0 matches
    spectra_with_zero_matches = total_success_masst - total_scans
    
    # Add zero matches to the counts_series for plotting
    extended_counts = pd.Series([0] * spectra_with_zero_matches + list(counts_series.values))
    
    # Plot histogram with all data including zero matches
    ax = sns.histplot(extended_counts, bins=50, kde=False)
    
    # y axis log scale
    ax.set_yscale('log')
    ax.set_ylim(1, ax.get_ylim()[1])  # Set y-axis limits to avoid log(0)
    
    # Add labels and title
    plt.xlabel('Number of MASST spectral matches')
    plt.ylabel('Unique USI count')
    title = 'Distribution of MASST Matches per USI'
    plt.title(title)
    
    # Count spectra with various match counts
    count_more_eq_1 = total_scans  # All scans in counts_series have at least 1 match
    count_more_eq_5 = (counts_series >= 5).sum()
    count_more_eq_10 = (counts_series >= 10).sum()
    count_more_eq_20 = (counts_series >= 20).sum()
    count_more_eq_50 = (counts_series >= 50).sum()
    
    # Calculate percentages based on total attempted MASST queries
    pct_more_eq_1 = count_more_eq_1 / total_success_masst * 100
    pct_more_eq_5 = count_more_eq_5 / total_success_masst * 100
    pct_more_eq_10 = count_more_eq_10 / total_success_masst * 100
    pct_more_eq_20 = count_more_eq_20 / total_success_masst * 100
    pct_more_eq_50 = count_more_eq_50 / total_success_masst * 100
    
    # Add annotation with statistics including exact counts
    stats_text = (f"≥1 matches: {count_more_eq_1:,d} MS/MS ({pct_more_eq_1:.1f}%)\n"
                 f"≥5 matches: {count_more_eq_5:,d} MS/MS ({pct_more_eq_5:.1f}%)\n"
                 f"≥10 matches: {count_more_eq_10:,d} MS/MS ({pct_more_eq_10:.1f}%)\n"
                 f"≥20 matches: {count_more_eq_20:,d} MS/MS ({pct_more_eq_20:.1f}%)\n"
                 f"≥50 matches: {count_more_eq_50:,d} MS/MS ({pct_more_eq_50:.1f}%)")
    
    plt.annotate(stats_text, xy=(0.6, 0.8), xycoords='axes fraction')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    filename = "masst_matches_distribution.svg"
    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file, format='svg', bbox_inches='tight')
    print(f"Figure saved to: {save_file}")
    
    # Save as PNG
    save_file_png = save_file.replace('.svg', '.png')
    plt.savefig(save_file_png, format='png', bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {save_file_png}")
    
    # Show plot
    plt.show()
    
    return counts_series


def plot_match_scatter(input_file='masst/analysis/data/match_counts.csv', 
                       save_path='masst/analysis/plot'):
    """
    Create a scatter plot showing the number of matches for each USI.
    
    Args:
        input_file: Path to the CSV file with match counts
        save_path: Directory to save the output plots
    """
    print(f"Loading match counts from {input_file}...")
    
    # Load precomputed match counts
    counts_df = pd.read_csv(input_file)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Sort by match count in descending order
    counts_df = counts_df.sort_values('match_count', ascending=False).reset_index(drop=True)
    
    # Set up the plot (make it wider for better visibility)
    plt.figure(figsize=(8, 5))
    
    # Plot scatter plot
    plt.scatter(range(len(counts_df)), counts_df['match_count'], 
               alpha=0.5, s=10, color='steelblue', edgecolor='none')
    
    # Add log scale for y-axis
    plt.yscale('log')
    
    # Set minimum y value to avoid log(0)
    plt.ylim(bottom=0.9)
    
    # Add labels and title
    plt.xlabel('USI index (sorted by match count)')
    plt.ylabel('Number of MASST spectral matches')
    plt.title('MASST Spectral Matches per USI')
    
    # Calculate statistics for annotation
    total_usis = len(counts_df)
    median_matches = counts_df['match_count'].median()
    percentile_25 = counts_df['match_count'].quantile(0.25)
    percentile_75 = counts_df['match_count'].quantile(0.75)
    max_matches = counts_df['match_count'].max()
    
    # Add annotation with statistics
    stats_text = (f"Total USIs with MASST matches: {total_usis:,d}\n"
                  f"25th percentile: {percentile_25:.0f} matches\n"
                  f"Median: {median_matches:.0f} matches\n"
                  f"75th percentile: {percentile_75:.0f} matches\n"
                 f"Maximum: {max_matches:,d} matches")
    
    plt.annotate(stats_text, xy=(0.60, 0.80), xycoords='axes fraction')
    
    # Add grid to help with readability
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    filename = "masst_matches_scatter.svg"
    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file, format='svg', bbox_inches='tight')
    print(f"Figure saved to: {save_file}")
    
    # Save as PNG
    save_file_png = save_file.replace('.svg', '.png')
    plt.savefig(save_file_png, format='png', bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {save_file_png}")
    
    # Show plot
    plt.show()
    
    return counts_df


def merge_masst_summary_with_spec_info():
    df1 = pd.read_csv('masst/analysis/data/match_counts.csv')    
    df2 = pd.read_csv('masst/data_prepare/ms2_all_unique_usi.tsv', sep='\t', low_memory=False)
    
    # rename df2 columns
    df2.columns = ['usi', 'scan', 'spec_cnt_sharing_usi']
    
    # merge the two dataframes on the 'scan' column
    df = df2.merge(df1, on='scan', how='left')
    # fillna with 0
    df['match_count'] = df['match_count'].fillna(0)
    # convert to int
    df['match_count'] = df['match_count'].astype(int)
    
    # drop col of scan
    df = df.drop(columns=['scan']).reset_index(drop=True)
    
    # load the library dataframe
    lib = pd.read_pickle('data_cleaning/cleaned_data/ms2_all_df.pkl')
    '''
        ['name', 'exact_mass', 'scan', 'smiles', 'inchi', 'usi', 'mz', 'adduct',
       'formula', 'inchikey', '2d_inchikey']
    '''
    lib = lib[['scan', 'name', 'smiles', 'inchikey', '2d_inchikey', 'usi', 'mz', 'adduct']]
    # Merge the two dataframes on the 'usi' column
    df = lib.merge(df, on='usi', how='left')
    
    # sort by match_count
    df = df.sort_values('match_count', ascending=False).reset_index(drop=True)
    
    # Save the merged dataframe to a new CSV file
    df.to_csv('masst/analysis/data/match_counts_with_spec_info.tsv', index=False, sep='\t')

    # group by 2d_inchikey
    # calculate total match count, reserve scans as a list, for others, reserve the first row
    df = df.groupby('2d_inchikey').agg({
        'match_count': 'sum',
        'name': lambda x: ';'.join(list(set(x))),
        'smiles': 'first',
        'inchikey': 'first',
        'usi': lambda x: list(x),
        'scan': lambda x: list(x),
    }).reset_index()
    # rename the columns
    df.columns = ['2d_inchikey', 'match_count', 'name', 'smiles', 'inchikey', 'usi', 'scan']
    # sort by match_count
    df = df.sort_values('match_count', ascending=False).reset_index(drop=True)
    
    # Save the grouped dataframe to a new CSV file
    df.to_csv('masst/analysis/data/match_counts_grouped_by_2d_inchikey.tsv', index=False, sep='\t')

    # filter the name col, contains "_"
    df = df[df['name'].str.contains('_')]
    df.to_csv('masst/analysis/data/match_counts_grouped_by_2d_inchikey_filtered.tsv', index=False, sep='\t')


def plot_ref_spec_match_distribution(input_file='masst/analysis/data/match_counts_with_spec_info.tsv', 
                                     save_path='masst/analysis/plot',
                                     group_by_2d_inchikey=False):
    """
    Create a histogram of the number of MASST matches per reference spectrum or unique structure.
    
    Args:
        input_file: Path to the TSV file with match counts and spectral info
        save_path: Directory to save the output plots
        group_by_2d_inchikey: If True, group matches by 2D InChIKey (unique structure)
    
    Returns:
        pd.DataFrame: The loaded data
    """
    print(f"Loading match counts from {input_file}...")
    
    # Load match counts with spectral info
    df = pd.read_csv(input_file, sep='\t')
    
    # Group by 2D InChIKey if requested
    if group_by_2d_inchikey:
        print("Grouping by 2D InChIKey (unique structure)...")
        df = df.groupby('2d_inchikey').agg({
            'match_count': 'sum',
            'name': lambda x: ';'.join(list(set(x))),
            'smiles': 'first',
            'inchikey': 'first'
        }).reset_index()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Set up the plot
    plt.figure(figsize=(8, 5))
    
    # Plot histogram
    ax = sns.histplot(df['match_count'], bins=50, kde=False)
    
    # y axis log scale
    ax.set_yscale('log')
    ax.set_ylim(0.9, ax.get_ylim()[1])  # Set y-axis limits to avoid log(0)
    
    # Add labels and title
    plt.xlabel('Number of MASST spectral matches')
    
    if group_by_2d_inchikey:
        entity_type = "unique structure"
        y_label = 'Unique structure count (log scale)'
        title = 'Distribution of MASST Matches per Unique Structure'
    else:
        entity_type = "reference spectrum"
        y_label = 'Reference spectra count (log scale)'
        title = 'Distribution of MASST Matches per Reference Spectrum'
    
    plt.ylabel(y_label)
    plt.title(title)
    
    # Calculate total reference spectra or structures
    total_entities = len(df)
    
    # Count spectra with various match counts
    count_more_eq_1 = (df['match_count'] >= 1).sum()
    count_more_eq_5 = (df['match_count'] >= 5).sum()
    count_more_eq_10 = (df['match_count'] >= 10).sum()
    count_more_eq_20 = (df['match_count'] >= 20).sum()
    count_more_eq_50 = (df['match_count'] >= 50).sum()
    
    # Calculate percentages
    pct_more_eq_1 = count_more_eq_1 / total_entities * 100
    pct_more_eq_5 = count_more_eq_5 / total_entities * 100
    pct_more_eq_10 = count_more_eq_10 / total_entities * 100
    pct_more_eq_20 = count_more_eq_20 / total_entities * 100
    pct_more_eq_50 = count_more_eq_50 / total_entities * 100
    
    # Add annotation with statistics
    if group_by_2d_inchikey:
        stats_prefix = f"Total unique structures: {total_entities:,d}\n"
    else:
        stats_prefix = f"Total reference spectra: {total_entities:,d}\n"
    
    stats_text = (stats_prefix +
                 f"≥1 matches: {count_more_eq_1:,d} ({pct_more_eq_1:.1f}%)\n"
                 f"≥5 matches: {count_more_eq_5:,d} ({pct_more_eq_5:.1f}%)\n"
                 f"≥10 matches: {count_more_eq_10:,d} ({pct_more_eq_10:.1f}%)\n"
                 f"≥20 matches: {count_more_eq_20:,d} ({pct_more_eq_20:.1f}%)\n"
                 f"≥50 matches: {count_more_eq_50:,d} ({pct_more_eq_50:.1f}%)")
    
    plt.annotate(stats_text, xy=(0.6, 0.7), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.7))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    suffix = "_by_structure" if group_by_2d_inchikey else ""
    filename = f"masst_matches_distribution{suffix}.svg"
    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file, format='svg', bbox_inches='tight')
    print(f"Figure saved to: {save_file}")
    
    # Save as PNG
    save_file_png = save_file.replace('.svg', '.png')
    plt.savefig(save_file_png, format='png', bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {save_file_png}")
    
    # Show plot
    plt.show()
    
    return df


def plot_ref_spec_match_scatter(input_file='masst/analysis/data/match_counts_with_spec_info.tsv', 
                               save_path='masst/analysis/plot',
                               group_by_2d_inchikey=False):
    """
    Create a scatter plot showing the number of matches for each reference spectrum or unique structure.
    
    Args:
        input_file: Path to the TSV file with match counts and spectral info
        save_path: Directory to save the output plots
        group_by_2d_inchikey: If True, group matches by 2D InChIKey (unique structure)
    
    Returns:
        pd.DataFrame: The sorted data
    """
    print(f"Loading match counts from {input_file}...")
    
    # Load match counts with spectral info
    df = pd.read_csv(input_file, sep='\t')
    
    # Group by 2D InChIKey if requested
    if group_by_2d_inchikey:
        print("Grouping by 2D InChIKey (unique structure)...")
        df = df.groupby('2d_inchikey').agg({
            'match_count': 'sum',
            'name': lambda x: ';'.join(list(set(x))),
            'smiles': 'first',
            'inchikey': 'first'
        }).reset_index()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Sort by match count in descending order
    df = df.sort_values('match_count', ascending=False).reset_index(drop=True)
    df = df[df['match_count'] > 0].reset_index(drop=True)
    
    # Set up the plot
    plt.figure(figsize=(8, 5))
    
    # Plot scatter plot
    plt.scatter(range(len(df)), df['match_count'], 
               alpha=0.5, s=10, color='steelblue', edgecolor='none')
    
    # Add log scale for y-axis
    plt.yscale('log')
    
    # Set minimum y value to avoid log(0)
    plt.ylim(bottom=0.9)
    
    # Add labels and title
    if group_by_2d_inchikey:
        x_label = 'Unique structure index (sorted by match count)'
        title = 'MASST Spectral Matches per Unique Structure'
    else:
        x_label = 'Reference spectrum index (sorted by match count)'
        title = 'MASST Spectral Matches per Reference Spectrum'
    
    plt.xlabel(x_label)
    plt.ylabel('Number of MASST spectral matches')
    plt.title(title)
    
    # Calculate statistics
    total_entities = len(df)
    median_matches = df['match_count'].median()
    percentile_25 = df['match_count'].quantile(0.25)
    percentile_75 = df['match_count'].quantile(0.75)
    max_matches = df['match_count'].max()
    
    # Get the top matched reference or structure
    top_match = df.iloc[0]
    top_match_name = top_match['name']
    if isinstance(top_match_name, str) and len(top_match_name) > 30:
        top_match_name = top_match_name[:27] + "..."
    
    # Add annotation with statistics
    if group_by_2d_inchikey:
        stats_prefix = f"Total unique structures with matches: {total_entities:,d}\n"
    else:
        stats_prefix = f"Total reference spectra with matches: {total_entities:,d}\n"
    
    stats_text = (stats_prefix +
                  f"25th percentile: {percentile_25:.0f} matches\n"
                  f"Median: {median_matches:.0f} matches\n"
                  f"75th percentile: {percentile_75:.0f} matches\n"
                  f"Max: {max_matches:,d} matches")
    
    plt.annotate(stats_text, xy=(0.5, 0.7), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.7))
    
    # Add grid for readability
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    suffix = "_by_structure" if group_by_2d_inchikey else ""
    filename = f"masst_matches_scatter{suffix}.svg"
    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file, format='svg', bbox_inches='tight')
    print(f"Figure saved to: {save_file}")
    
    # Save as PNG
    save_file_png = save_file.replace('.svg', '.png')
    plt.savefig(save_file_png, format='png', bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {save_file_png}")
    
    # Show plot
    plt.show()
    
    return df


if __name__ == "__main__":
    
    # Define file paths
    raw_data_path = 'masst/analysis/data/all_masst_matches.tsv'
    counts_file_path = 'masst/analysis/data/match_counts.csv'
    plot_path = 'masst/analysis/plot'
    
    # # Check if we need to generate the counts data
    # if not os.path.exists(counts_file_path):
    #     print("Generating match counts data...")
    #     generate_match_counts(input_file=raw_data_path, output_file=counts_file_path)
    # else:
    #     print(f"Match counts data already exists at {counts_file_path}")
    
    # # Create the histogram plot
    # print("Creating match distribution histogram...")
    # plot_match_distribution(input_file=counts_file_path, save_path=plot_path)
    
    # # Create the scatter plot
    # print("\nCreating match distribution scatter plot...")
    # plot_match_scatter(input_file=counts_file_path, save_path=plot_path)
    
    # #################
    # merge_masst_summary_with_spec_info()    
    # #################
    # Create the histogram plot for reference spectra
    print("\nCreating reference spectra match distribution histogram...")
    plot_ref_spec_match_distribution(
        input_file='masst/analysis/data/match_counts_with_spec_info.tsv', 
        save_path=plot_path,
        group_by_2d_inchikey=False
    )
    plot_ref_spec_match_scatter(
        input_file='masst/analysis/data/match_counts_with_spec_info.tsv', 
        save_path=plot_path,
        group_by_2d_inchikey=False
    )
    
    # Create the histogram and scatter plots for unique structures
    print("\nCreating unique structure match distribution plots...")
    plot_ref_spec_match_distribution(
        input_file='masst/analysis/data/match_counts_with_spec_info.tsv', 
        save_path=plot_path,
        group_by_2d_inchikey=True
    )
    plot_ref_spec_match_scatter(
        input_file='masst/analysis/data/match_counts_with_spec_info.tsv', 
        save_path=plot_path,
        group_by_2d_inchikey=True
    )
