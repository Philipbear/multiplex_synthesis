import os
import pandas as pd
import multiprocessing as mp
from functools import partial
from tqdm import tqdm


def process_file(file, result_dir):
    file_path = os.path.join(result_dir, file)
    try:
        df = pd.read_csv(file_path, sep='\t', low_memory=False)
        if df.empty or df.isna().all().all():
            return None
            
        df['scan'] = file.split('all_')[1].split('_matches.tsv')[0]
        
        # drop cols of Status
        if 'Status' in df.columns:
            df.drop(columns=['Status'], inplace=True)
            
        # Add dataset column directly here to avoid another operation later
        df['dataset'] = df['USI'].apply(lambda x: x.split(':')[1])
        
        return df
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None


def get_all_matches_on_server(result_dir, n_cores=None):
    # Get all TSV files
    files = [f for f in os.listdir(result_dir) if f.endswith('_matches.tsv')]
    
    if n_cores is None:
        # Use all available cores except one
        n_cores = max(1, mp.cpu_count() - 1)
    
    print(f"Processing {len(files)} files using {n_cores} cores...")
    
    # Create a pool of workers
    pool = mp.Pool(n_cores)
    
    # Process files in parallel
    process_func = partial(process_file, result_dir=result_dir)
    results = list(tqdm(pool.imap(process_func, files), total=len(files), desc="Processing files"))
    
    # Close the pool
    pool.close()
    pool.join()
    
    # Filter out None results and concatenate all dataframes at once
    valid_dfs = [df for df in results if df is not None]
    print(f"Concatenating {len(valid_dfs)} valid dataframes...")
    
    # More efficient concatenation
    all_df = pd.concat(valid_dfs, ignore_index=True, copy=False)
    
    # Save the full results
    print("Saving all matches to 'all_masst_matches.tsv'")
    all_df.to_csv('all_masst_matches.tsv', sep='\t', index=False)
    
    # Filter and save the filtered results
    filtered_df = all_df[~all_df['dataset'].isin(['MSV000094559', 'MSV000094447'])]
    
    print(f"Processed {len(valid_dfs)} files, resulting in {len(all_df)} total rows")
    print(f"After filtering, {len(filtered_df)} rows remain")
    
    return all_df, filtered_df


if __name__ == "__main__":
    
    get_all_matches_on_server('output/all')
    print("All matches have been concatenated and saved.")