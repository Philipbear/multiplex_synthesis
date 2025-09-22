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
    
    df = df[['lib_usi', 'inchikey_2d', 'mri', 'mri_scan']]
    df['mri_scan'] = df['mri_scan'].astype(str)
    df['matched_usi'] = df['mri'] + ':' + df['mri_scan']

    return df['matched_usi'].unique(), df['lib_usi'].unique(), df['inchikey_2d'].unique()


def perform_analysis_main(processed_output_path):
    """
    Analyze MASST match distribution by USI and by structure, saving data files for plotting
    """
    
    print("\n=== Analyzing MASST Match Distribution ===")
    
    files = [f for f in os.listdir(processed_output_path) if f.endswith('.pkl')]
    
    pool = mp.Pool(n_cores)
    process_func = partial(process_file, result_dir=processed_output_path)
    results = list(tqdm(pool.imap(process_func, files), total=len(files), desc="Processing MASST files"))
    pool.close()
    pool.join()
    
    # Combine results from all files
    all_matched_usi = set()
    all_lib_usi = set()
    all_inchikey_2d = set()
    for matched_usi, lib_usi, inchikey_2d in results:
        all_matched_usi.update(matched_usi)
        all_lib_usi.update(lib_usi)
        all_inchikey_2d.update(inchikey_2d)

    print(f"Total unique matched USIs across all public files: {len(all_matched_usi)}")
    print(f"Total unique library USIs across all public files: {len(all_lib_usi)}")
    print(f"Total unique 2D InChIKeys: {len(all_inchikey_2d)}")

    return all_matched_usi


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # on server
    processed_output_path = "/home/shipei/projects/synlib/masst/processed_output"    
    perform_analysis_main(processed_output_path)
