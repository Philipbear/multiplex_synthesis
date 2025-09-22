import pandas as pd
import os


'''
NCBITaxonomy
    human: '9606|Homo sapiens'
    rodents: (["10088|Mus", "10090|Mus musculus", "10105|Mus minutoides", "10114|Rattus", "10116|Rattus norvegicus"])
'''

SYN_DATASETS = ['MSV000097885', 'MSV000097874', 'MSV000097869', 'MSV000094559', 'MSV000094447', 'MSV000094393', 'MSV000094391', 
                'MSV000094382', 'MSV000094337', 'MSV000094300', 'MSV000098637', 'MSV000098628', 'MSV000098639', 'MSV000098640']

def merge_all(masst_path='masst/analysis/data/all_masst_matches.tsv',
              lib_path='target_drugs/all/target.tsv',
              redu_path='masst/analysis/data/redu.tsv',
              output_path='masst/analysis/data/all_masst_matches_redu.pkl'):
    """
    Merge all MASST results with the library and redu dataframes.
    Ensures all MASST matches are mapped to all associated library scans.
    """
    print('Merging all data sources...')
    
    # Prepare the MASST results
    masst = prepare_all_masst_results(masst_path, lib_path)
    print(f"MASST matches: {len(masst)} rows")
    
    # Prepare the redu dataframe
    redu = prepare_redu(redu_path)
    print(f"ReDU entries: {len(redu)} rows")
    
    # Merge with redu dataframe to add metadata
    print('Adding ReDU metadata...')
    final_df = masst.merge(redu, on='mri', how='left')

    # final cols: 'name', 'lib_usi', 'mri', 'mri_scan', 'NCBITaxonomy', 'UBERONBodyPartName', 'DOIDCommonName', 'HealthStatus'

    # save
    final_df.to_pickle(output_path)
    print(f"Final merged dataset saved to: {output_path}")
    
    final_df.to_csv(output_path.replace('.pkl', '.tsv'), sep='\t', index=False)
    
    return final_df


def prepare_all_masst_results(masst_path='masst/analysis/data/all_masst_matches.tsv',
                              lib_path='target_drugs/all/target.tsv'):
    """
    Prepare all MASST results for analysis.
    """
    print('Preparing all MASST results...')
    # Load the MASST results
    df = pd.read_csv(masst_path, sep='\t')
    df = df[~df['dataset'].isin(SYN_DATASETS)].reset_index(drop=True)
    df = df[['name', 'USI']]
    
    # Load the library dataframe
    lib = pd.read_csv(lib_path, sep='\t', low_memory=False)

    # Merge the library dataframe with the MASST dataframe
    df = df.merge(lib, on='name', how='left')
    
    df = df[['name', 'usi', 'USI']]
    df.columns = ['name', 'lib_usi', 'matched_usi']
    
    # Add the mri column
    df['mri'] = df['matched_usi'].apply(lambda x: x.split(':scan')[0].split('mzspec:')[1])
    df['mri_scan'] = df['matched_usi'].apply(lambda x: x.split(':scan:')[1])
    df['mri_scan'] = df['mri_scan'].astype(float)
    df['mri_scan'] = df['mri_scan'].astype(int)
    
    df = df.drop(columns=['matched_usi'])

    return df   # masst_matches: name, 'lib_usi', 'mri', 'mri_scan'


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
    masst_path = 'target_drugs/all/data/all_masst_matches_0.7_3.tsv'
    lib_path = 'target_drugs/all/target.tsv'
    redu_path = 'masst/analysis/data/redu.tsv'
    output_path = 'target_drugs/all/data/all_masst_matches_with_metadata_0.7_3.pkl'
    
    # Run the merge function
    merged_df = merge_all(masst_path, lib_path, redu_path, output_path)
    
    # Print the first few rows of the merged dataframe
    print(merged_df.head())
    