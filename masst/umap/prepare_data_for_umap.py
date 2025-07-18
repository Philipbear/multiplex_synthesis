import pandas as pd
import numpy as np
import os
import pickle


def load_and_preprocess_masst_data(merged_all_masst_path, min_matches_per_usi=3):
    """
    Load and preprocess MASST data once for all subsequent processing
    """
    print("Loading MASST data (this may take a while)...")
    df = pd.read_pickle(merged_all_masst_path)
    
    print(f"Initial data: {len(df):,} matches, {df['lib_usi'].nunique():,} unique lib_usi")
    
    # Clean compound names
    df['name'] = df['name'].apply(lambda x: x.split(' (known')[0] if pd.notnull(x) else x)
    
    # Filter out USIs with fewer than min_matches_per_usi matches
    print(f"Filtering USIs with at least {min_matches_per_usi} MASST matches...")
    usi_counts = df['lib_usi'].value_counts()
    valid_usis = usi_counts[usi_counts >= min_matches_per_usi].index
    df = df[df['lib_usi'].isin(valid_usis)].reset_index(drop=True)
    
    print(f"After USI filtering: {len(df):,} matches, {df['lib_usi'].nunique():,} unique lib_usi")
    
    # Remove rows without NCBITaxonomy information
    print("Filtering data with NCBITaxonomy information...")
    df = df[(df['NCBITaxonomy'].notna()) & (df['NCBITaxonomy'] != '') & (df['NCBITaxonomy'] != 'missing value')]
    
    df['ncbi_ids'] = df['NCBITaxonomy'].apply(lambda x: x.split('|')[0])
    df['ncbi_ids'] = df['ncbi_ids'].astype(int)
    
    print(f"After NCBI filtering: {len(df):,} matches, {df['lib_usi'].nunique():,} unique lib_usi")
    print(f"Unique NCBI IDs: {df['ncbi_ids'].nunique():,}")
    
    return df


def prepare_data_for_rank(df, ncbi_dict_path, output_dir, rank_name):
    """
    Prepare UMAP data for a specific taxonomic rank using pre-loaded data
    """
    print(f"\nProcessing {rank_name} rank...")
    
    # Map NCBITaxonomy to shown rank using the provided dictionary
    print(f"Mapping NCBITaxonomy to {rank_name} shown rank...")
    with open(ncbi_dict_path, 'rb') as f:
        ncbi_to_shown_rank = pickle.load(f)
    
    # Create a copy for this rank to avoid modifying the original
    df_rank = df.copy()
    df_rank['shown_rank'] = df_rank['ncbi_ids'].map(ncbi_to_shown_rank)
    
    # Remove rows without shown_rank mapping
    df_rank = df_rank[df_rank['shown_rank'].notna()]
    
    print(f"Total MASST matches with {rank_name} shown rank info: {len(df_rank):,}")
    print(f"Unique lib_usi: {df_rank['lib_usi'].nunique():,}")
    print(f"Unique {rank_name} shown ranks: {df_rank['shown_rank'].nunique():,}")
    
    # Create match count matrix: lib_usi × shown_rank
    print("Creating match count matrix...")
    match_counts = df_rank.groupby(['lib_usi', 'shown_rank']).size().reset_index(name='match_count')

    # Pivot to create the matrix: rows=lib_usi, columns=shown_rank, values=match_count
    umap_matrix = match_counts.pivot(index='lib_usi', columns='shown_rank', values='match_count')
    umap_matrix = umap_matrix.fillna(0).astype(int)
    
    print(f"UMAP matrix shape: {umap_matrix.shape[0]} lib_usi × {umap_matrix.shape[1]} {rank_name} shown ranks")
    
    # Get metadata for each lib_usi
    print("Collecting metadata for each lib_usi...")
    metadata_cols = ['lib_usi', 'name', 'inchikey_2d']
    
    # Get first occurrence of each lib_usi with its metadata
    usi_metadata = df_rank[metadata_cols].drop_duplicates(subset=['lib_usi'], keep='first').set_index('lib_usi')
    
    # Ensure the index order matches the matrix
    usi_metadata = usi_metadata.reindex(umap_matrix.index)
    
    # Add total match count and shown rank count as additional features
    usi_metadata['total_matches'] = umap_matrix.sum(axis=1)
    usi_metadata['num_shown_ranks'] = (umap_matrix > 0).sum(axis=1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the matrix and metadata
    print(f"Saving {rank_name} data to {output_dir}...")
    
    # Save as pickle for fast loading
    umap_matrix.to_pickle(f"{output_dir}/umap_match_matrix.pkl")
    usi_metadata.to_pickle(f"{output_dir}/usi_metadata.pkl")
    
    # Save as TSV for inspection
    umap_matrix.to_csv(f"{output_dir}/umap_match_matrix.tsv", sep='\t')
    usi_metadata.to_csv(f"{output_dir}/usi_metadata.tsv", sep='\t')
    
    # Save shown rank information
    shown_rank_info = df_rank.groupby('shown_rank').agg({
        'mri': 'count',  # Total matches per shown rank
        'lib_usi': 'nunique'  # Unique USIs per shown rank
    }).rename(columns={'mri': 'total_matches', 'lib_usi': 'unique_usis'})
    shown_rank_info = shown_rank_info.sort_values('total_matches', ascending=False)
    shown_rank_info.to_csv(f"{output_dir}/shown_rank_info.tsv", sep='\t')
    
    print(f"{rank_name.capitalize()} data preparation complete!")
    print(f"Files saved:")
    print(f"  - umap_match_matrix.pkl/.tsv: {umap_matrix.shape[0]} × {umap_matrix.shape[1]} matrix")
    print(f"  - usi_metadata.pkl/.tsv: metadata for {len(usi_metadata)} USIs")
    print(f"  - shown_rank_info.tsv: information for {len(shown_rank_info)} shown ranks")
    
    return umap_matrix.index.tolist()  # Return list of USIs


def prepare_usi_labeling_data(df, base_umap_dir, output_dir):
    """
    Prepare USI to label mapping for UMAP plot labeling using show_name categories
    Output: table with columns ['usi', 'ncbi_list', 'num_ncbi', 'label', 
    'kingdom_id_count', 'phylum_id_count', 'class_id_count', 
    'order_id_count', 'family_id_count', 'genus_id_count', 'species_id_count']
    """
    print("\nPreparing USI labeling data...")
    
    # Load show_name mapping
    ncbi_to_show_name_path = f'{base_umap_dir}/data/ncbi_to_show_name.pkl'
    print("Mapping NCBITaxonomy to show_name...")
    with open(ncbi_to_show_name_path, 'rb') as f:
        ncbi_to_show_name = pickle.load(f)
    
    # Load all rank mappings
    ranks = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    rank_mappings = {}
    
    for rank in ranks:
        rank_dict_path = f'{base_umap_dir}/data/ncbi_to_{rank}.pkl'
        if os.path.exists(rank_dict_path):
            print(f"Loading {rank} mapping...")
            with open(rank_dict_path, 'rb') as f:
                rank_mappings[rank] = pickle.load(f)
        else:
            print(f"Warning: {rank_dict_path} not found. Skipping {rank} labeling.")
            rank_mappings[rank] = {}
    
    # Create a copy for labeling to avoid modifying the original
    df_label = df.copy()
    df_label['show_name'] = df_label['ncbi_ids'].map(ncbi_to_show_name)
    
    # Add rank mappings to dataframe
    for rank in ranks:
        df_label[f'{rank}_shown_rank'] = df_label['ncbi_ids'].map(rank_mappings[rank])
    
    # Remove rows without show_name mapping
    df_label = df_label[df_label['show_name'].notna()]
    
    print(f"Total MASST matches with show_name info: {len(df_label):,}")
    print(f"Unique lib_usi: {df_label['lib_usi'].nunique():,}")
    print(f"Unique show_names: {df_label['show_name'].nunique():,}")
    
    # Group by USI to get all categories for each USI
    agg_dict = {
        'ncbi_ids': lambda x: list(set(x)),  # Unique NCBI IDs for this USI
        'show_name': lambda x: list(set(x))  # Unique show_names for this USI
    }
    
    # Add aggregation for each rank
    for rank in ranks:
        agg_dict[f'{rank}_shown_rank'] = lambda x: list(set([y for y in x if pd.notna(y)]))
    
    usi_groups = df_label.groupby('lib_usi').agg(agg_dict).reset_index()
    
    # Create the output table
    output_data = []
    
    for _, row in usi_groups.iterrows():
        usi = row['lib_usi']
        ncbi_list = sorted(row['ncbi_ids'])  # Sort for consistency
        num_ncbi = len(ncbi_list)
        show_names = set(row['show_name'])
        
        num_target_categories = len(show_names)
        
        # Determine main label based on show_names
        if num_target_categories == 1:
            label = list(show_names)[0]
        elif num_target_categories == 2:
            # Sort alphabetically for consistent naming
            sorted_cats = sorted(list(show_names))
            label = ' + '.join(sorted_cats)
        else:
            # 0, 3, 4, 5, or 6 categories
            label = 'Others'
        
        # Create the base record
        record = {
            'usi': usi,
            'ncbi_list': ncbi_list,
            'num_ncbi': num_ncbi,
            'label': label
        }
        
        # Add rank-specific ID counts
        for rank in ranks:
            rank_categories = set(row[f'{rank}_shown_rank']) if row[f'{rank}_shown_rank'] else set()
            # Remove empty strings and None values
            rank_categories = {cat for cat in rank_categories if cat and cat != ''}
            
            # Store the count of unique IDs for this rank
            record[f'{rank}_id_count'] = len(rank_categories)
        
        output_data.append(record)
    
    # Create DataFrame
    output_df = pd.DataFrame(output_data)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the table
    output_df.to_pickle(f"{output_dir}/usi_labeling_table.pkl")
    output_df.to_csv(f"{output_dir}/usi_labeling_table.tsv", sep='\t', index=False)
    
    # Print summary statistics for each rank
    print(f"\nUSI labeling data saved to {output_dir}:")
    print(f"  - usi_labeling_table.pkl/.tsv: table with {len(output_df)} USIs")
    print(f"\nTable shape: {output_df.shape}")
    print(f"Columns: {list(output_df.columns)}")    
    
    return output_df


def process_all_ranks(merged_all_masst_path, base_umap_dir, min_matches_per_usi=3):
    """
    Main function to process all taxonomic ranks efficiently
    """
    print("="*80)
    print("EFFICIENT MASST DATA PROCESSING FOR ALL RANKS")
    print("="*80)
    
    # Load and preprocess data once
    df = load_and_preprocess_masst_data(merged_all_masst_path, min_matches_per_usi)
    
    # Process each taxonomic rank
    ranks = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

    for i, rank in enumerate(ranks, 1):
        print(f"\n{'='*60}")
        print(f"Processing rank {i}/{len(ranks)}: {rank.upper()}")
        print('='*60)
        
        ncbi_dict_path = f'{base_umap_dir}/data/ncbi_to_{rank}.pkl'
        output_dir = f'{base_umap_dir}/{rank}_based/data'
        
        # Check if the NCBI dictionary exists
        if not os.path.exists(ncbi_dict_path):
            print(f"Warning: {ncbi_dict_path} not found. Skipping {rank}.")
            continue
            
        prepare_data_for_rank(df, ncbi_dict_path, output_dir, rank)
    
    # Prepare USI labeling data (now includes all ranks)
    print(f"\n{'='*60}")
    print("Preparing USI labeling data with all ranks")
    print('='*60)
    
    prepare_usi_labeling_data(df, base_umap_dir, f'{base_umap_dir}/data')
    
    print(f"\n{'='*80}")
    print("ALL DATA PROCESSING COMPLETE!")
    print(f"Processed data for {len(ranks)} taxonomic ranks")
    print(f"Total unique USIs processed: {df['lib_usi'].nunique():,}")
    print(f"Total matches processed: {len(df):,}")
    print('='*80)


if __name__ == '__main__':
    # Define the data paths (adjust as needed)
    base_data_path = '/home/shipei/projects/synlib/masst/data/all_masst_matches_with_metadata.pkl'
    base_umap_dir = '/home/shipei/projects/synlib/masst/umap'
    
    # Use the efficient processing function
    process_all_ranks(
        merged_all_masst_path=base_data_path,
        base_umap_dir=base_umap_dir,
        min_matches_per_usi=3
    )