import pandas as pd
import os
import multiprocessing as mp
from functools import partial
from tqdm import tqdm


'''
'spec_id', 'lib_usi', 'name', 'inchikey_2d', 'mri', 'mri_scan', 'repo', 'NCBITaxonomy', 'UBERONBodyPartName', 'DOIDCommonName', 'HealthStatus'
'''

n_cores = max(mp.cpu_count() - 1, 1)

def process_file_masst_match_distribution(file, result_dir):
        
    file_path = os.path.join(result_dir, file)

    df = pd.read_pickle(file_path)
    
    # Analyze by USI (lib_usi)
    print("Analyzing matches by USI...")
    usi_match_counts = df.groupby('lib_usi').size().reset_index()
    usi_match_counts.columns = ['lib_usi', 'match_count']
    
    # add names and inchikey_2d for reference
    usi_match_counts = usi_match_counts.merge(df[['lib_usi', 'name', 'inchikey_2d']].drop_duplicates(), on='lib_usi', how='left')

    # Analyze by structure (inchikey_2d)
    print("Analyzing matches by structure...")
    structure_match_counts = usi_match_counts.sort_values(by='match_count', ascending=False).drop_duplicates(subset=['inchikey_2d'])
    structure_match_counts = structure_match_counts[['inchikey_2d', 'match_count', 'name']]

    return usi_match_counts, structure_match_counts



def process_file_repo_distribution(file, result_dir):
    file_path = os.path.join(result_dir, file)
    df = pd.read_pickle(file_path)
    
    # === Analysis by USI ===
    # Create presence/absence matrix for USIs (for Venn plots)
    usi_repo_presence = df.groupby(['lib_usi', 'repo']).size().unstack(fill_value=0)
    usi_repo_presence = (usi_repo_presence > 0).astype(int)

    return usi_repo_presence


def analyze_masst_match_distribution(processed_output_path):
    """
    Analyze MASST match distribution by USI and by structure, saving data files for plotting
    """
    
    print("\n=== Analyzing MASST Match Distribution ===")
    
    files = [f for f in os.listdir(processed_output_path) if f.endswith('.pkl')]
    
    pool = mp.Pool(n_cores)
    process_func = partial(process_file_masst_match_distribution, result_dir=processed_output_path)
    results = list(tqdm(pool.imap(process_func, files), total=len(files), desc="Processing MASST files"))
    pool.close()
    pool.join()
    
    # Combine results from all files
    usi_match_counts = pd.concat([res[0] for res in results], ignore_index=True)
    structure_match_counts = pd.concat([res[1] for res in results], ignore_index=True)
    
    # Aggregate counts
    structure_match_counts = structure_match_counts.sort_values(by='match_count', ascending=False).drop_duplicates(subset=['inchikey_2d'])

    # Save data files for plotting
    usi_match_counts.to_csv("data/usi_match_counts.tsv", sep='\t', index=False)
    structure_match_counts.to_csv("data/structure_match_counts.tsv", sep='\t', index=False)

    return


def analyze_masst_repo_distribution(processed_output_path):
    """
    Analyze the distribution of MASST matches across different repositories
    Generate data for Venn plot analysis by USI and structure
    """
    
    print("\n=== Analyzing MASST Match Distribution by Repository ===")
    
    files = [f for f in os.listdir(processed_output_path) if f.endswith('.pkl')]
    pool = mp.Pool(n_cores)
    process_func = partial(process_file_repo_distribution, result_dir=processed_output_path)
    results = list(tqdm(pool.imap(process_func, files), total=len(files), desc="Processing MASST files for repo distribution"))
    pool.close()
    pool.join()
    
    # Combine results from all files
    usi_repo_presence = pd.concat(results, ignore_index=False).fillna(0).astype(int)

    # === Save data files for Venn plot analysis ===
    usi_repo_presence.to_csv("data/usi_repo_presence_matrix.tsv", sep='\t', index=True)
    
    return


def perform_analysis_main(processed_output_path):
    """
    Main function to perform MASST match distribution analysis
    """
    print("=== Starting MASST Match Distribution Analysis ===")
        
    # 1. Analyze MASST match distribution
    if os.path.exists("data/usi_match_counts.tsv") and os.path.exists("data/structure_match_counts.tsv"):
        print("Data files already exist, skipping analysis.")
    else:
        analyze_masst_match_distribution(processed_output_path)
    
    # 2. Analyze repo distribution of all MASST matches
    if os.path.exists("data/usi_repo_presence_matrix.tsv"):
        print("Repo distribution data files already exist, skipping analysis.")
    else:
        analyze_masst_repo_distribution(processed_output_path)

    print("Analysis completed successfully!")
    
    return


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    #####
    # on server
    processed_output_path = "/home/shipei/projects/synlib/masst/processed_output"    
    perform_analysis_main(processed_output_path)
