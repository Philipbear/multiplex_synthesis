import pandas as pd
import os
import multiprocessing as mp
from functools import partial
from tqdm import tqdm


'''
'spec_id', 'lib_usi', 'name', 'inchikey_2d', 'mri', 'mri_scan', 'repo', 'NCBITaxonomy', 'UBERONBodyPartName', 'DOIDCommonName', 'HealthStatus'
'''

n_cores = max(mp.cpu_count() - 1, 1)

def process_file(file, result_dir):
        
    file_path = os.path.join(result_dir, file)

    df = pd.read_pickle(file_path)
    
    df = df[['lib_usi', 'mri', 'mri_scan', 'inchikey_2d', 'NCBITaxonomy', 'UBERONBodyPartName']]
    
    df = df[(df['NCBITaxonomy'].notna()) & (df['NCBITaxonomy'] != '') & (df['NCBITaxonomy'] != 'missing value')]
    
    df['ncbi_ids'] = df['NCBITaxonomy'].apply(lambda x: x.split('|')[0])
    df['ncbi_ids'] = df['ncbi_ids'].astype(str)
    
    df['matched_usi'] = df['mri'].astype(str) + ':' + df['mri_scan'].astype(str)
    
    # drop cols
    df = df[['lib_usi', 'matched_usi', 'inchikey_2d', 'ncbi_ids', 'UBERONBodyPartName']]    
    
    # group by lib_usi and aggregate
    df_grouped = df.groupby('lib_usi').agg({
        'inchikey_2d': 'first',
        'ncbi_ids': lambda x: list(set(x)),
        'matched_usi': 'count'
    }).reset_index()
    df_grouped.rename(columns={'matched_usi': 'match_count'}, inplace=True)
    
    # only human matches
    df_human_only = df_grouped[df_grouped['ncbi_ids'].apply(lambda x: len(x) == 1 and x[0] == '9606')].reset_index(drop=True)
    human_only_usis = df_human_only['lib_usi'].unique().tolist()
    df_human_only = df[df['lib_usi'].isin(human_only_usis)].reset_index(drop=True)
    
    # group by UBERONBodyPartName and lib_usi
    df_human_only_grouped = df_human_only.groupby(['UBERONBodyPartName', 'lib_usi']).agg({
        'matched_usi': 'count',
        'inchikey_2d': 'first'
    }).reset_index()
    df_human_only_grouped.rename(columns={'matched_usi': 'count'}, inplace=True)

    return df_human_only_grouped


def perform_analysis_main(processed_output_path, out_dir):
    """
    Analyze MASST match distribution by USI and by structure, saving data files for plotting
    """
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("\n=== Analyzing human only Masst matches ===")
    
    files = [f for f in os.listdir(processed_output_path) if f.endswith('.pkl')]
    
    pool = mp.Pool(n_cores)
    process_func = partial(process_file, result_dir=processed_output_path)
    results = list(tqdm(pool.imap(process_func, files), total=len(files), desc="Processing MASST files"))
    pool.close()
    pool.join()
    
    # Combine results from all files
    all_dfs = pd.concat(results, ignore_index=True)

    # Save combined DataFrame to a pickle file
    all_dfs.to_pickle(os.path.join(out_dir, "human_only_usis_raw_count.pkl"))
    all_dfs.to_csv(os.path.join(out_dir, "human_only_usis_raw_count.tsv"), sep='\t', index=False)
    
    # unique USIs
    unique_usis = all_dfs['lib_usi'].unique().tolist()
    print(f"{len(unique_usis):,} unique USIs")
    
    # unique inchikeys
    unique_inchis = all_dfs['inchikey_2d'].unique().tolist()
    print(f"{len(unique_inchis):,} unique InChikeys")
    

if __name__ == '__main__':
    
    # on server
    processed_output_path = "/home/shipei/projects/synlib/masst/processed_output"
    out_dir = "/home/shipei/projects/synlib/masst/human/data"
    
    perform_analysis_main(processed_output_path, out_dir)
