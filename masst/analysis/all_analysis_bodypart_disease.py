import pandas as pd
import os


def load_masst_data(masst_pkl_path):
    """
    Load the generated MASST data from pickle file
    """
    print(f"Loading MASST data from {masst_pkl_path}...")
    df = pd.read_pickle(masst_pkl_path)
    
    df['name'] = df['name'].apply(lambda x: x.split(' (known')[0])
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
        'microbe': f"{base_dir}/microbe"
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def analyze_microbe_distribution(masst_df, output_dirs, 
                                 microbe_table_path='masst/analysis/data/microbe_masst_table.csv'):
    """
    Analyze microbeMASST matches distribution by USI and structure and save the data files for later plotting
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
    
    # === Analysis by USI ===
    # Count microbe matches per lib_usi
    usi_microbe_counts = df[df['is_microbe_match']].groupby('lib_usi').size().reset_index()
    usi_microbe_counts.columns = ['lib_usi', 'microbe_match_count']

    # Get USI names for reference
    usi_names = df[['lib_usi', 'name']].drop_duplicates().set_index('lib_usi')['name'].to_dict()
    usi_microbe_counts['name'] = usi_microbe_counts['lib_usi'].map(usi_names)

    # === Analysis by Structure ===
    # Count microbe matches per structure (inchikey_2d), taking first name for each structure
    structure_microbe_data = df[
        (df['is_microbe_match']) & 
        (df['inchikey_2d'].notna())
    ].groupby('inchikey_2d').agg({
        'mri': 'size',  # Count matches
        'name': 'first'  # Take first name for each structure
    }).reset_index()
    
    structure_microbe_data.columns = ['inchikey_2d', 'microbe_match_count', 'name']

    # Calculate total counts
    total_usis = df['lib_usi'].nunique()
    total_structures = df['inchikey_2d'].nunique()

    # Calculate percentage of USIs and structures with microbe matches
    usis_with_microbe = usi_microbe_counts['lib_usi'].nunique()
    structures_with_microbe = structure_microbe_data['inchikey_2d'].nunique()

    print(f"Library USIs with microbe matches: {usis_with_microbe:,}/{total_usis:,} ({usis_with_microbe/total_usis*100:.1f}%)")
    print(f"Unique structures with microbe matches: {structures_with_microbe:,}/{total_structures:,} ({structures_with_microbe/total_structures*100:.1f}%)")
    
    # Save match counts and metadata to files for later plotting
    usi_microbe_counts.to_csv(f"{output_dirs['microbe']}/usi_microbe_match_counts.tsv", sep='\t', index=False)
    structure_microbe_data.to_csv(f"{output_dirs['microbe']}/structure_microbe_match_counts.tsv", sep='\t', index=False)
    
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
        'structure_microbe_counts': structure_microbe_data,
        'total_usis': total_usis,
        'total_structures': total_structures,
        'usis_with_microbe': usis_with_microbe,
        'structures_with_microbe': structures_with_microbe
    }


def analyze_bodypart_distribution(masst_df, output_dirs):
    """
    Analyze bodypart distribution for human and rodent samples by USI and structure
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
    
    # === Analysis by USI ===
    print("Analyzing USI distribution by body part...")
    
    # Count unique USIs per body part
    human_usi_counts = human_df.groupby('UBERONBodyPartName')['lib_usi'].nunique().reset_index()
    human_usi_counts = human_usi_counts.rename(columns={'lib_usi': 'count'}).sort_values('count', ascending=False)

    rodent_usi_counts = rodent_df.groupby('UBERONBodyPartName')['lib_usi'].nunique().reset_index()
    rodent_usi_counts = rodent_usi_counts.rename(columns={'lib_usi': 'count'}).sort_values('count', ascending=False)

    # Raw USI-bodypart match counts for upset plots
    human_bodypart_usi_counts = human_df.groupby(['UBERONBodyPartName', 'lib_usi']).size().reset_index(name='count')
    rodent_bodypart_usi_counts = rodent_df.groupby(['UBERONBodyPartName', 'lib_usi']).size().reset_index(name='count')

    # === Analysis by Structure ===
    print("Analyzing unique structure distribution by body part...")
    
    # Count unique structures per body part
    human_structure_counts = human_df[human_df['inchikey_2d'].notna()].groupby('UBERONBodyPartName')['inchikey_2d'].nunique().reset_index()
    human_structure_counts = human_structure_counts.rename(columns={'inchikey_2d': 'count'}).sort_values('count', ascending=False)
    
    rodent_structure_counts = rodent_df[rodent_df['inchikey_2d'].notna()].groupby('UBERONBodyPartName')['inchikey_2d'].nunique().reset_index()
    rodent_structure_counts = rodent_structure_counts.rename(columns={'inchikey_2d': 'count'}).sort_values('count', ascending=False)
    
    # Raw structure-bodypart match counts for upset plots (grouped by inchikey_2d, taking first name)
    human_compound_bodypart_counts = human_df[human_df['inchikey_2d'].notna()].groupby(['UBERONBodyPartName', 'inchikey_2d']).agg({
        'mri': 'size',  # Count matches
        'name': 'first'  # Take first name for each structure
    }).reset_index()
    human_compound_bodypart_counts.columns = ['UBERONBodyPartName', 'inchikey_2d', 'count', 'name']
    
    rodent_compound_bodypart_counts = rodent_df[rodent_df['inchikey_2d'].notna()].groupby(['UBERONBodyPartName', 'inchikey_2d']).agg({
        'mri': 'size',  # Count matches
        'name': 'first'  # Take first name for each structure
    }).reset_index()
    rodent_compound_bodypart_counts.columns = ['UBERONBodyPartName', 'inchikey_2d', 'count', 'name']
    
    # === Prepare data for UpSet plots ===
    print("Preparing data for UpSet plots...")
    
    # For USI upset plots - we need the format: UBERONBodyPartName, lib_usi, count
    # Reformat the data to match upsetplot_bodypart.py expectations
    human_bodypart_usi_upset = human_bodypart_usi_counts.copy()
    rodent_bodypart_usi_upset = rodent_bodypart_usi_counts.copy()
    
    # For structure upset plots - we need the format: UBERONBodyPartName, inchikey_2d, count
    # The compound_bodypart_counts already have the right format, just reorder columns
    human_compound_bodypart_upset = human_compound_bodypart_counts[['UBERONBodyPartName', 'inchikey_2d', 'count', 'name']].copy()
    rodent_compound_bodypart_upset = rodent_compound_bodypart_counts[['UBERONBodyPartName', 'inchikey_2d', 'count', 'name']].copy()
    
    # Save data files for summary statistics
    human_usi_counts.to_csv(f"{output_dirs['bodypart']}/human_bodypart_usi_counts.tsv", sep='\t', index=False)
    rodent_usi_counts.to_csv(f"{output_dirs['bodypart']}/rodent_bodypart_usi_counts.tsv", sep='\t', index=False)
    human_structure_counts.to_csv(f"{output_dirs['bodypart']}/human_bodypart_structure_counts.tsv", sep='\t', index=False)
    rodent_structure_counts.to_csv(f"{output_dirs['bodypart']}/rodent_bodypart_structure_counts.tsv", sep='\t', index=False)
    
    # Save raw counts for upset plots (matching upsetplot_bodypart.py format)
    human_bodypart_usi_upset.to_csv(f"{output_dirs['bodypart']}/human_bodypart_usi_raw_counts.tsv", sep='\t', index=False)
    rodent_bodypart_usi_upset.to_csv(f"{output_dirs['bodypart']}/rodent_bodypart_usi_raw_counts.tsv", sep='\t', index=False)
    human_compound_bodypart_upset.to_csv(f"{output_dirs['bodypart']}/human_compound_bodypart_counts.tsv", sep='\t', index=False)
    rodent_compound_bodypart_upset.to_csv(f"{output_dirs['bodypart']}/rodent_compound_bodypart_counts.tsv", sep='\t', index=False)
    
    # Print statistics for upset plots
    print(f"\nUpSet plot data prepared:")
    print(f"Human USI-bodypart combinations: {len(human_bodypart_usi_upset):,}")
    print(f"Human structure-bodypart combinations: {len(human_compound_bodypart_upset):,}")
    print(f"Rodent USI-bodypart combinations: {len(rodent_bodypart_usi_upset):,}")
    print(f"Rodent structure-bodypart combinations: {len(rodent_compound_bodypart_upset):,}")
    
    print(f"\nHuman body parts: {human_bodypart_usi_upset['UBERONBodyPartName'].nunique()}")
    print(f"Rodent body parts: {rodent_bodypart_usi_upset['UBERONBodyPartName'].nunique()}")
    print(f"Human unique USIs: {human_bodypart_usi_upset['lib_usi'].nunique():,}")
    print(f"Human unique structures: {human_compound_bodypart_upset['inchikey_2d'].nunique():,}")
    print(f"Rodent unique USIs: {rodent_bodypart_usi_upset['lib_usi'].nunique():,}")
    print(f"Rodent unique structures: {rodent_compound_bodypart_upset['inchikey_2d'].nunique():,}")
    
    return {
        'human_usi_counts': human_usi_counts,
        'rodent_usi_counts': rodent_usi_counts,
        'human_structure_counts': human_structure_counts,
        'rodent_structure_counts': rodent_structure_counts,
        'human_upset_usi_data': human_bodypart_usi_upset,
        'rodent_upset_usi_data': rodent_bodypart_usi_upset,
        'human_upset_structure_data': human_compound_bodypart_upset,
        'rodent_upset_structure_data': rodent_compound_bodypart_upset
    }
    

def analyze_disease_distribution(masst_df, output_dirs):
    """
    Analyze disease distribution in MASST matches by USI and structure
    """
    print("\n=== Analyzing Disease Distribution ===")
    
    # Create a copy of the dataframe
    df = masst_df.copy()
    
    # Filter for valid disease information
    df = df[df['DOIDCommonName'].notna() & 
           (df['DOIDCommonName'] != 'missing value')]
    
    print(f"Found {len(df):,} matches with disease information")
    
    # === Analysis by USI ===
    print("Analyzing USI distribution by disease...")
    
    # Count unique USIs per disease
    disease_usi_counts = df.groupby('DOIDCommonName')['lib_usi'].nunique().reset_index()
    disease_usi_counts = disease_usi_counts.rename(columns={'lib_usi': 'count'}).sort_values('count', ascending=False)

    # Raw USI-disease match counts
    disease_usi_raw_counts = df.groupby(['DOIDCommonName', 'lib_usi']).size().reset_index(name='count')

    # === Analysis by Structure ===
    print("Analyzing unique structure distribution by disease...")
    
    # Count unique structures per disease
    disease_structure_counts = df[df['inchikey_2d'].notna()].groupby('DOIDCommonName')['inchikey_2d'].nunique().reset_index()
    disease_structure_counts = disease_structure_counts.rename(columns={'inchikey_2d': 'count'}).sort_values('count', ascending=False)
     
    # Raw structure-disease match counts (grouped by inchikey_2d, taking first name)
    compound_disease_counts = df[df['inchikey_2d'].notna()].groupby(['DOIDCommonName', 'inchikey_2d']).agg({
        'mri': 'size',  # Count matches
        'name': 'first'  # Take first name for each structure
    }).reset_index()
    compound_disease_counts.columns = ['DOIDCommonName', 'inchikey_2d', 'count', 'name']
    
    # Save data files
    disease_usi_counts.to_csv(f"{output_dirs['disease']}/disease_usi_counts.tsv", sep='\t', index=False)
    disease_structure_counts.to_csv(f"{output_dirs['disease']}/disease_structure_counts.tsv", sep='\t', index=False)
    disease_usi_raw_counts.to_csv(f"{output_dirs['disease']}/disease_usi_raw_counts.tsv", sep='\t', index=False)
    compound_disease_counts.to_csv(f"{output_dirs['disease']}/compound_disease_counts.tsv", sep='\t', index=False)
    
    return {
        'disease_usi_counts': disease_usi_counts,
        'disease_structure_counts': disease_structure_counts
    }


def analyze_general_structure_distribution(masst_df, output_dirs):
    """
    General analysis: summarize MASST matches per InChIKey 2D (unique molecular structure)
    """
    print("\n=== Analyzing General Structure Distribution ===")
    
    # Create a copy of the dataframe
    df = masst_df.copy()
    
    # Filter out rows without InChIKey 2D information
    df = df[df['inchikey_2d'].notna()]
    
    print(f"Found {len(df):,} MASST matches with InChIKey 2D information")
    print(f"Found {df['inchikey_2d'].nunique():,} unique molecular structures")
    
    # Count total MASST matches per InChIKey 2D
    structure_match_counts = df.groupby('inchikey_2d').size().reset_index(name='masst_match_count')
    
    # Get additional statistics per structure
    structure_stats = df.groupby('inchikey_2d').agg({
        'name': 'first',  # Use first name for each InChIKey 2D
        'lib_scan': 'nunique',  # Number of unique library spectra matched
        'mri': 'nunique',       # Number of unique datasets where found
        'NCBITaxonomy': lambda x: x.dropna().nunique(),  # Number of unique taxa
        'UBERONBodyPartName': lambda x: x[x != 'missing value'].dropna().nunique(),  # Number of unique body parts
        'DOIDCommonName': lambda x: x[x != 'missing value'].dropna().nunique(),      # Number of unique diseases
        'HealthStatus': lambda x: x[x != 'missing value'].dropna().nunique()         # Number of unique health statuses
    }).reset_index()
    
    # Rename columns for clarity
    structure_stats = structure_stats.rename(columns={
        'lib_scan': 'unique_library_spectra',
        'mri': 'unique_matched_files', 
        'NCBITaxonomy': 'unique_taxa',
        'UBERONBodyPartName': 'unique_body_parts',
        'DOIDCommonName': 'unique_diseases',
        'HealthStatus': 'unique_health_statuses'
    })
    
    # Merge match counts with statistics
    structure_summary = structure_match_counts.merge(structure_stats, on='inchikey_2d')
    
    # Sort by MASST match count (descending)
    structure_summary = structure_summary.sort_values('masst_match_count', ascending=False)
    
    # Add rank column
    structure_summary['rank'] = range(1, len(structure_summary) + 1)
    
    # Reorder columns for better readability
    column_order = [
        'rank', 'inchikey_2d', 'name', 'masst_match_count', 
        'unique_library_spectra', 'unique_matched_files', 'unique_taxa',
        'unique_body_parts', 'unique_diseases', 'unique_health_statuses'
    ]
    structure_summary = structure_summary[column_order]
    
    # Save the comprehensive summary
    output_file = f"{output_dirs['main']}/structure_masst_summary.tsv"
    structure_summary.to_csv(output_file, sep='\t', index=False)
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total unique structures analyzed: {len(structure_summary):,}")
    print(f"Total MASST matches: {structure_summary['masst_match_count'].sum():,}")
    print(f"Average MASST matches per structure: {structure_summary['masst_match_count'].mean():.1f}")
    print(f"Median MASST matches per structure: {structure_summary['masst_match_count'].median():.0f}")
    print(f"Max MASST matches for a single structure: {structure_summary['masst_match_count'].max():,}")
    print(f"Min MASST matches for a single structure: {structure_summary['masst_match_count'].min()}")
    
    # Print top 10 structures
    print(f"\nTop 10 structures by MASST match count:")
    for _, row in structure_summary.head(10).iterrows():
        print(f"  {row['rank']:3d}. {row['name'][:60]:<60} - {row['masst_match_count']:,} matches")
    
    print(f"\nDetailed results saved to: {output_file}")
    
    return {
        'structure_summary': structure_summary,
        'total_structures': len(structure_summary),
        'total_matches': structure_summary['masst_match_count'].sum()
    }


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
    print("\n=== Starting Comprehensive Analysis ===")
    
    # General structure distribution analysis
    general_structure_results = analyze_general_structure_distribution(masst_df, output_dirs)
    bodypart_results = analyze_bodypart_distribution(masst_df, output_dirs)
    disease_results = analyze_disease_distribution(masst_df, output_dirs)
    microbe_results = analyze_microbe_distribution(masst_df, output_dirs, microbe_table_path)
    
    print("\n=== Comprehensive Analysis Complete ===")
    
    return


if __name__ == "__main__":
    
    masst_pkl_path = '/home/shipei/projects/synlib/masst/data/all_masst_matches_with_metadata.pkl'
    output_dir = 'data'
    microbe_table_path = '/home/shipei/projects/microbe_masst/sql/microbe_masst_table.csv'
    
    # Run the analysis
    results = run_comprehensive_analysis(
        masst_pkl_path, 
        output_dir,
        microbe_table_path
    )    
    
    print("Analysis complete.")