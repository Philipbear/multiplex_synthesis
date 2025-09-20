import os
import pandas as pd
import multiprocessing as mp
from functools import partial
from tqdm import tqdm


SYN_DATASETS = ['MSV000097885', 'MSV000097874', 'MSV000097869', 'MSV000094559', 'MSV000094447', 'MSV000094393', 'MSV000094391', 
                'MSV000094382', 'MSV000094337', 'MSV000094300', 'MSV000098637', 'MSV000098628', 'MSV000098639', 'MSV000098640']


def process_file(file, result_dir):
    
    this_spec_id = file.split('.tsv')[0]
    
    file_path = os.path.join(result_dir, file)
    
    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    if df.empty or df.isna().all().all():
        return None
    
    df = df[~df['dataset'].isin(SYN_DATASETS)].reset_index(drop=True)
    df['mri'] = df['dataset'].astype(str) + ':' + df['file'].astype(str) + ':scan:' + df['scan'].astype(str)
    
    # rename
    df = df.rename(columns={'scan': 'mri_scan'})
    
    # drop cols
    df = df.drop(columns=['delta_mass', 'cosine', 'matching_peaks', 'file'], errors='ignore')
    
    if df.empty:
        return None

    df['spec_id'] = this_spec_id
    
    # add repo column
    df['repo'] = df['mri'].apply(lambda x: x.split(':')[0][:2])

    return df


def prepare_lib(lib_path='data_cleaning/cleaned_data/ms2_all_df.pkl'):
    """
    Prepare the library dataframe for analysis.
    """
    print('Preparing library dataframe...')
    
    '''
    name	mz	smiles	inchi	usi	spec_id	adduct	formula	exact_mass	inchikey	2d_inchikey
    '''
    
    # load the library dataframe
    lib = pd.read_pickle(lib_path)
    
    # sort by spec_id
    lib = lib.sort_values(by='spec_id').reset_index(drop=True)
    
    lib = lib[['spec_id', 'usi', 'name', '2d_inchikey']]
    lib = lib.rename(columns={'usi': 'lib_usi', '2d_inchikey': 'inchikey_2d'})

    # dereplicate by usi, keep the first one (smallest spec_id)
    lib = lib.drop_duplicates(subset='lib_usi', keep='first').reset_index(drop=True)
    print(f"Number of unique USIs: {len(lib)}")

    return lib   # lib: spec_id, lib_usi, name, inchikey_2d


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
    
    redu['mri'] = redu['mri'].apply(lambda x: x.split('mzspec:')[1])
    redu['dataset'] = redu['mri'].apply(lambda x: x.split(':')[0])

    return redu   # redu: 'mri', dataset, 'NCBITaxonomy', 'UBERONBodyPartName', 'DOIDCommonName', 'HealthStatus'


def process_all_matches_on_server(result_dir, lib_path, redu_path, out_dir, n_cores=None, batch_size=1000):
    
    os.makedirs(out_dir, exist_ok=True)
    
    lib_df = prepare_lib(lib_path)
    redu_df = prepare_redu(redu_path)

    # Get all TSV files
    files = [f for f in os.listdir(result_dir) if f.endswith('.tsv') and f.startswith('CCMS')]
    
    if n_cores is None:
        # Use all available cores except one
        n_cores = max(mp.cpu_count() - 1, 1)

    print(f"Processing {len(files)} files using {n_cores} cores in batches of {batch_size}...")

    num_batches = (len(files) + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        batch_files = files[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        print(f"Processing batch {batch_idx+1}/{num_batches} with {len(batch_files)} files...")

        pool = mp.Pool(n_cores)
        process_func = partial(process_file, result_dir=result_dir)
        results = list(tqdm(pool.imap(process_func, batch_files), total=len(batch_files), desc=f"Batch {batch_idx+1}"))
        pool.close()
        pool.join()

        valid_dfs = [df for df in results if df is not None]
        print(f"Concatenating {len(valid_dfs)} valid dataframes in batch {batch_idx+1}...")

        if not valid_dfs:
            print(f"No valid dataframes in batch {batch_idx+1}, skipping.")
            continue

        all_df = pd.concat(valid_dfs, ignore_index=True, copy=False)

        # merge with lib_df
        all_df = all_df.merge(lib_df, on='spec_id', how='left') # spec_id, lib_usi, name, inchikey_2d

        # merge with redu
        redu_df_batch = redu_df[redu_df['dataset'].isin(all_df['dataset'].unique())].reset_index(drop=True)
        redu_df_batch = redu_df_batch.drop(columns=['dataset'], errors='ignore')
        all_df = all_df.merge(redu_df_batch, on='mri', how='left') # 'NCBITaxonomy', 'UBERONBodyPartName', 'DOIDCommonName', 'HealthStatus'

        # cols
        all_df = all_df.rename(columns={'scan_id': 'mri_scan'})
        all_df = all_df[['spec_id', 'lib_usi', 'name', 'inchikey_2d',
                         'mri', 'mri_scan', 'repo', 'NCBITaxonomy', 'UBERONBodyPartName', 'DOIDCommonName', 'HealthStatus']]

        # print some info
        print(f"Batch {batch_idx+1}: Total MASST matches: {len(all_df)}")

        # Save batch
        batch_path = os.path.join(out_dir, f'all_masst_matches_batch{batch_idx+1}.pkl')
        all_df.to_pickle(batch_path)
        print(f"Saved batch {batch_idx+1} with {len(all_df)} rows to {batch_path}")

    print(f"All batches saved to {out_dir}")


if __name__ == "__main__":
    
    masst_raw_dir = '/home/shipei/projects/synlib/masst/main/data/masst_results'
    lib_path = '/home/shipei/projects/synlib/masst/ms2_all_df.pkl'
    redu_path = '/home/shipei/projects/synlib/masst/redu.tsv'
    out_dir = '/home/shipei/projects/synlib/masst/processed_output'

    process_all_matches_on_server(masst_raw_dir, lib_path, redu_path, out_dir)
    