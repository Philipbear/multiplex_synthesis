'''
get all unique USIs that have matches to humans, and prepare mgf, for mol networking

only for conjugates (have '_' in the name)

# final cols: 'lib_usi', 'mri', 'mri_scan', 'lib_scan', 'name', 'inchikey_2d', 'NCBITaxonomy', 'UBERONBodyPartName', 'DOIDCommonName', 'HealthStatus'
'''

import pandas as pd
import os
from tqdm import tqdm


def get_human_usi(merged_all_masst_path, out_dir, min_matches_per_usi=3):
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
    df['ncbi_ids'] = df['ncbi_ids'].astype(str)
    
    print(f"After NCBI filtering: {len(df):,} matches, {df['lib_usi'].nunique():,} unique lib_usi")
    print(f"Unique NCBI IDs: {df['ncbi_ids'].nunique():,}")
    
    # group by lib_usi and aggregate
    print("Aggregating data by lib_usi...")
    df_grouped = df.groupby('lib_usi').agg({
        'name': 'first',
        'inchikey_2d': 'first',
        'ncbi_ids': lambda x: list(set(x)),
        'mri': 'count'
    }).reset_index()
    df_grouped.rename(columns={'mri': 'match_count'}, inplace=True)
    
    # only human matches
    df_human_only = df_grouped[df_grouped['ncbi_ids'].apply(lambda x: len(x) == 1 and x[0] == '9606')].reset_index(drop=True)    
    print(f"Final human-only matches: {len(df_human_only):,} matches, {df_human_only['lib_usi'].nunique():,} unique lib_usi")
    
    human_only_usis = df_human_only['lib_usi'].unique().tolist()
    df_human_only = df[df['lib_usi'].isin(human_only_usis)].reset_index(drop=True)
    
    # group by UBERONBodyPartName and lib_usi
    print("Grouping by UBERONBodyPartName and lib_usi...")
    df_human_only_grouped = df_human_only.groupby(['UBERONBodyPartName', 'lib_usi']).agg({
        'mri': 'count',
    }).reset_index()
    df_human_only_grouped.rename(columns={'mri': 'count'}, inplace=True)

    # Save the grouped DataFrame
    df_human_only_grouped.to_pickle(os.path.join(out_dir, 'human_only_usis_raw_count.pkl'))
    df_human_only_grouped.to_csv(os.path.join(out_dir, 'human_only_usis_raw_count.tsv'), sep='\t', index=False)

    print("Data processing complete!")


def analysis(df_path):
    df = pd.read_csv(df_path, sep='\t')
    print(f"Loaded data from {df_path}: {len(df):,} rows")

    # unique USIs
    unique_usis = df['lib_usi'].unique().tolist()
    print(f"Found {len(unique_usis):,} unique USIs")
    
    db = pd.read_pickle('data_cleaning/cleaned_data/ms2_all_df.pkl')
    db = db[db['usi'].isin(unique_usis)].reset_index(drop=True)
    print(f"Filtered database to {len(db):,} rows")
    
    # unique inchikeys
    unique_inchis = db['2d_inchikey'].unique().tolist()
    print(f"Found {len(unique_inchis):,} unique InChIs in the database")
    
    

if __name__ == '__main__':
    
    # on server
    # get_human_usi(
    #     merged_all_masst_path='/home/shipei/projects/synlib/masst/data/all_masst_matches_with_metadata.pkl',
    #     out_dir='/home/shipei/projects/synlib/masst/human',
    #     min_matches_per_usi=3
    # )
    
    # on local
    analysis('masst/human/data/human_only_usis_raw_count.tsv')
    