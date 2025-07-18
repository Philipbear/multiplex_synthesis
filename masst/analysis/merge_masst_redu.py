import pandas as pd
import os


'''
NCBITaxonomy
    human: '9606|Homo sapiens'
    rodents: (["10088|Mus", "10090|Mus musculus", "10105|Mus minutoides", "10114|Rattus", "10116|Rattus norvegicus"])
'''


def merge_all(masst_path='masst/analysis/data/all_masst_matches.tsv',
              lib_path='data_cleaning/cleaned_data/ms2_all_df.pkl', 
              redu_path='masst/analysis/data/redu.tsv',
              output_path='masst/analysis/data/all_masst_matches_redu.pkl',
              min_freq=3):
    """
    Merge all MASST results with the library and redu dataframes.
    Ensures all MASST matches are mapped to all associated library scans.
    """
    print('Merging all data sources...')
    
    # Prepare the library dataframe
    lib = prepare_lib(lib_path)
    ######### for lib, keep unique USIs
    lib = lib.drop_duplicates(subset=['lib_usi']).reset_index(drop=True)
    print(f"Library entries: {len(lib)} rows")
    
    # Prepare the MASST results
    masst = prepare_all_masst_results(masst_path, lib_path, min_freq)
    print(f"MASST matches: {len(masst)} rows")
    
    # Prepare the redu dataframe
    redu = prepare_redu(redu_path)
    print(f"ReDU entries: {len(redu)} rows")
    
    # merge: match each lib_scan with all masst matches having the same lib_usi
    print('Merging library scans to all matching MASST results...')
    df = pd.merge(
        masst,               # masst matches: 'lib_usi', 'mri', 'mri_scan'        
        lib,                 # lib: 'lib_scan', 'lib_usi', 'name', 'inchikey_2d'
        on='lib_usi',
        how='left',        # left join to keep all masst matches
    )
    
    print(f"MASST matches: {len(df)} rows")
    print(f"Unique library USIs with matches: {df['lib_usi'].nunique()}")
    print(f"Unique MRIs (matched datasets): {df['mri'].nunique()}")
    
    # Merge with redu dataframe to add metadata
    print('Adding ReDU metadata...')
    final_df = df.merge(redu, on='mri', how='left')

    # final cols: 'lib_usi', 'mri', 'mri_scan', 'lib_scan', 'name', 'inchikey_2d', 'NCBITaxonomy', 'UBERONBodyPartName', 'DOIDCommonName', 'HealthStatus'

    # save
    final_df.to_pickle(output_path)
    print(f"Final merged dataset saved to: {output_path}")
    
    # Print merge statistics
    print(f"Final merged dataset: {len(final_df)} rows")
    with_metadata = final_df[final_df['NCBITaxonomy'].notna()].copy()
    print(f"Matches with ReDU metadata: {len(with_metadata)} ({len(with_metadata)/len(final_df)*100:.1f}%)")
    
    return final_df


def prepare_all_masst_results(masst_path='masst/analysis/data/all_masst_matches.tsv',
                              lib_path='data_cleaning/cleaned_data/ms2_all_df.pkl', min_freq=1):
    """
    Prepare all MASST results for analysis.
    """
    print('Preparing all MASST results...')
    # Load the MASST results
    df = pd.read_csv(masst_path, sep='\t')
    df = df[~df['dataset'].isin(['MSV000094559', 'MSV000094447'])].reset_index(drop=True)
    df = df[['scan', 'USI']]
    
    # Filter out low frequency matches
    if min_freq > 1:
        print(f"Before filtering, total scans: {df['scan'].nunique()}")
        freq = df['scan'].value_counts()
        valid_scans = freq[freq >= min_freq].index
        df = df[df['scan'].isin(valid_scans)].reset_index(drop=True)
        print(f"After filtering, total scans: {df['scan'].nunique()} (min_freq={min_freq})")

    # Load the library dataframe
    lib = pd.read_pickle(lib_path)
    
    # Merge the library dataframe with the MASST dataframe
    df = df.merge(lib[['scan', 'usi']], on='scan', how='left')
    
    df = df[['usi', 'USI']]
    df.columns = ['lib_usi', 'matched_usi']
    
    # Add the mri column
    df['mri'] = df['matched_usi'].apply(lambda x: x.split(':scan')[0].split('mzspec:')[1])
    df['mri_scan'] = df['matched_usi'].apply(lambda x: x.split(':scan:')[1])
    df['mri_scan'] = df['mri_scan'].astype(float)
    df['mri_scan'] = df['mri_scan'].astype(int)
    
    df = df.drop(columns=['matched_usi'])

    return df   # masst_matches: 'lib_usi', 'mri', 'mri_scan'


def prepare_redu(redu_path='masst/analysis/data/redu.tsv'):
    """
    Prepare the redu dataframe for analysis.
    """
    print('Preparing redu dataframe...')
    # Load the redu dataframe
    redu = pd.read_csv(redu_path, sep='\t', low_memory=False)
    
    redu = redu[['USI', 'NCBITaxonomy', 'UBERONBodyPartName', 'DOIDCommonName', 'HealthStatus']]
    
    # rename USI to mri
    redu = redu.rename(columns={'USI': 'mri'})
    
    redu['mri'] = redu['mri'].apply(lambda x: x.split(':scan')[0].split('mzspec:')[1])

    return redu   # redu: 'mri', 'NCBITaxonomy', 'UBERONBodyPartName', 'DOIDCommonName', 'HealthStatus'


def prepare_lib(lib_path='data_cleaning/cleaned_data/ms2_all_df.pkl'):
    """
    Prepare the library dataframe for analysis.
    """
    print('Preparing library dataframe...')
    
    # load the library dataframe
    lib = pd.read_pickle(lib_path)
    '''
        ['name', 'exact_mass', 'scan', 'smiles', 'inchi', 'usi', 'mz', 'adduct',
       'formula', 'inchikey', '2d_inchikey']
    '''
    lib = lib[['scan', 'usi', 'name', 'exact_mass', 'adduct', 'mz', '2d_inchikey']]
    lib = lib.rename(columns={'usi': 'lib_usi', 
                              'exact_mass': 'prec_mz',
                              'mz': 'mono_mass',
                              '2d_inchikey': 'inchikey_2d', 
                              'scan': 'lib_scan'})
    
    return lib   # lib: 'lib_scan', 'lib_usi', 'name', 'prec_mz', 'adduct', 'mono_mass', 'inchikey_2d'



if __name__ == '__main__':        
    
    # Define paths
    # masst_path = 'masst/analysis/data/all_masst_matches.tsv'
    # lib_path = 'data_cleaning/cleaned_data/ms2_all_df.pkl'
    # redu_path = 'masst/analysis/data/redu.tsv'
    # output_path = 'masst/analysis/data/all_masst_matches_with_metadata.pkl'
    
    masst_path = '/home/shipei/projects/microbe_masst/all_masst_matches.tsv'
    lib_path = '/home/shipei/projects/microbe_masst/sql/ms2_all_df.pkl'
    redu_path = '/home/shipei/projects/microbe_masst/sql/redu.tsv'    
    output_path = '/home/shipei/projects/synlib/masst/data/all_masst_matches_with_metadata.pkl'
    
    # Run the merge function
    merged_df = merge_all(masst_path, lib_path, redu_path, output_path, 1)
    
    # Print the first few rows of the merged dataframe
    print(merged_df.head())
    
    '''
    Merging all data sources...
    Preparing library dataframe...
    Library entries: 129013 rows
    Preparing all MASST results...
    Before filtering, total scans: 57378
    After filtering, total scans: 47156 (min_freq=3)
    MASST matches: 292842443 rows
    Preparing redu dataframe...
    ReDU entries: 749984 rows
    Merging library scans to all matching MASST results...
    MASST matches: 292842443 rows
    Unique library USIs with matches: 47156
    Unique MRIs (matched datasets): 350087
    Adding ReDU metadata...
    Final merged dataset saved to: /home/shipei/projects/synlib/masst/data/all_masst_matches_with_metadata.pkl
    Final merged dataset: 292842443 rows
    Matches with ReDU metadata: 78948189 (27.0%)
    '''
    