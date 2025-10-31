import pandas as pd
import os
import multiprocessing as mp
from functools import partial
from tqdm import tqdm


'''
'spec_id', 'lib_usi', 'name', 'inchikey_2d', 'mri', 'mri_scan', 'repo', 'NCBITaxonomy', 'UBERONBodyPartName', 'DOIDCommonName', 'HealthStatus'
'''

n_cores = max(mp.cpu_count() - 1, 1)


def get_short_mri(mri):
    dataset = mri.split(':')[0]
    file_path = mri.split(':')[1]
    
    file_path = file_path.split('/')[-1].split('.mz')[0]

    return f"{dataset}:{file_path}"


def process_file(file, result_dir, out_dir, microbemasst_df):
        
    file_path = os.path.join(result_dir, file)

    df = pd.read_pickle(file_path)
    
    df = df[['lib_usi', 'inchikey_2d', 'mri', 'mri_scan']]
    df['short_mri'] = df['mri'].apply(lambda x: get_short_mri(x))
    
    # remove col
    df = df[['lib_usi', 'inchikey_2d', 'short_mri', 'mri_scan']]
    
    df = df.merge(microbemasst_df, on='short_mri', how='left')

    # remove rows with NaN in Taxa_NCBI
    df = df[~df['Taxa_NCBI'].isna()].reset_index(drop=True)
    
    if df.empty:
        return
    
    out_file_path = os.path.join(out_dir, file)
    df.to_pickle(out_file_path)
    print(f"Saved {len(df)} entries with microbe info to {out_file_path}")


def microbemasst_main(processed_output_path, out_dir, microbemasst_table_path):
    """
    Analyze MASST match distribution by USI and by structure, saving data files for plotting
    """
    
    print("\n=== Merge with microbe MASST table ===")
    
    os.makedirs(out_dir, exist_ok=True)
    
    microbemasst_df = pd.read_csv(microbemasst_table_path)
    microbemasst_df = microbemasst_df[(microbemasst_df['QC'] == 'No') & (microbemasst_df['Blank'] == 'No')].reset_index(drop=True)
    # keep columns
    microbemasst_df = microbemasst_df[['file_usi', 'Taxaname_file', 'Taxaname_alternative', 'Taxa_NCBI']]
    microbemasst_df['file_usi'] = microbemasst_df['file_usi'].apply(lambda x: x.split('mzspec:')[1])
    # rename
    microbemasst_df = microbemasst_df.rename(columns={'file_usi': 'short_mri'})
    
    print(f"Total {len(microbemasst_df)} entries in microbe MASST table after QC and blank filtering.")
    
    files = [f for f in os.listdir(processed_output_path) if f.endswith('.pkl')]
    
    pool = mp.Pool(n_cores)
    process_func = partial(process_file, result_dir=processed_output_path, out_dir=out_dir, microbemasst_df=microbemasst_df)
    results = list(tqdm(pool.imap(process_func, files), total=len(files), desc="Processing MASST files"))
    pool.close()
    pool.join()

    return


def merge_microbemasst_tables(out_dir):
    files = [f for f in os.listdir(out_dir) if f.endswith('.pkl')]
    all_dfs = []
    for file in files:
        file_path = os.path.join(out_dir, file)
        df = pd.read_pickle(file_path)
        all_dfs.append(df)
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    merged_out_path = os.path.join(out_dir, "merged_microbemasst_table.tsv")
    merged_df.to_csv(merged_out_path, sep='\t', index=False)
    print(f"Merged microbe MASST table saved to {merged_out_path}, total {len(merged_df)} entries.")



def remove_humans_and_print_stats():
    df = pd.read_csv('masst/microbemasst/data/merged_microbemasst_table.tsv', sep='\t')
    print(f"Total entries: {len(df)}")
    
    # remove rows with Taxa_NCBI == 9606 (human)
    df = df[df['Taxa_NCBI'] != 9606].reset_index(drop=True)
    print(f"Entries after removing human (NCBI Taxonomy ID 9606): {len(df)}")
    
    # save cleaned table
    cleaned_out_path = 'masst/microbemasst/data/merged_microbemasst_table_no_human.tsv'
    df.to_csv(cleaned_out_path, sep='\t', index=False)
    
    # unique USIs
    unique_usis = df['lib_usi'].nunique()
    print(f"Unique USIs: {unique_usis}")
    # unique InChIKeys
    unique_inchikeys = df['inchikey_2d'].nunique()
    print(f"Unique InChIKeys: {unique_inchikeys}")
    # unique matched NCBI Taxonomy IDs
    unique_ncbi_taxa = df['Taxa_NCBI'].nunique()
    print(f"Unique matched NCBI Taxonomy IDs: {unique_ncbi_taxa}")


if __name__ == '__main__':    
    # # on server
    # processed_output_path = "/home/shipei/projects/synlib/masst/processed_output"
    # microbemasst_table_path = "/home/shipei/projects/synlib/masst/data/microbe_masst_table.csv"
    # out_dir = "/home/shipei/projects/synlib/masst/processed_output_with_microbe_info"
    
    # microbemasst_main(processed_output_path, out_dir, microbemasst_table_path)
    
    # merge_microbemasst_tables(out_dir)
    
    #############
    # on local
    remove_humans_and_print_stats()
