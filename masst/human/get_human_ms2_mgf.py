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
    
    # contains human matches
    df_human = df_grouped[df_grouped['ncbi_ids'].apply(lambda x: '9606' in x)].reset_index(drop=True)    
    print(f"Final human matches: {len(df_human):,} matches, {df_human['lib_usi'].nunique():,} unique lib_usi")
    
    # Save the filtered DataFrame    
    os.makedirs(out_dir, exist_ok=True)
    df_human.to_pickle(os.path.join(out_dir, 'human_masst_matches.pkl'))
    df_human.to_csv(os.path.join(out_dir, 'human_masst_matches.tsv'), sep='\t', index=False)
    
    # only human matches
    df_human_only = df_human[df_human['ncbi_ids'].apply(lambda x: len(x) == 1 and x[0] == '9606')].reset_index(drop=True)    
    print(f"Final human-only matches: {len(df_human_only):,} matches, {df_human_only['lib_usi'].nunique():,} unique lib_usi")
    
    # Save the human-only DataFrame
    df_human_only.to_pickle(os.path.join(out_dir, 'human_only_masst_matches.pkl'))
    df_human_only.to_csv(os.path.join(out_dir, 'human_only_masst_matches.tsv'), sep='\t', index=False)
    
    print("Data processing complete!")


def write_human_mgf(human_pkl_path, ms2lib_df_path, ms2lib_mgf_path, out_path, conjugates_only=True):
    
    # Load MS2Lib DataFrame
    ms2lib_df = pd.read_pickle(ms2lib_df_path)
    # dereplicate by usi
    ms2lib_df = ms2lib_df.drop_duplicates(subset='usi').reset_index(drop=True)    
    # dict from usi to scan
    usi_to_scan = ms2lib_df.set_index('usi')['scan'].to_dict()
    
    # load human matches
    print(f"Loading human matches from {human_pkl_path}...")
    human_df = pd.read_pickle(human_pkl_path)
    
    if conjugates_only:
        human_df = human_df[human_df['name'].str.contains('_')].reset_index(drop=True)
    
    human_df['scan'] = human_df['lib_usi'].map(usi_to_scan)
    reserved_scans = human_df['scan'].tolist()
    
    all_ms2_list = read_mgf(ms2lib_mgf_path)
    print(f"Loaded {len(all_ms2_list):,} spectra from MS2Lib MGF file.")
    
    # filter spectra for human matches
    human_ms2_list = []
    for spec in tqdm(all_ms2_list, desc="Filtering spectra"):
        if int(spec['SCANS']) in reserved_scans:
            human_ms2_list.append(spec)
    print(f"Filtered to {len(human_ms2_list):,} spectra matching human USIs.")
    
    # write to MGF file
    write_mgf(human_ms2_list, out_path)
    print(f"Written {len(human_ms2_list):,} spectra to {out_path}.")


def read_mgf(library_mgf):
    with open(library_mgf, 'r') as file:
        spectrum_list = []
        for line in file:
            # empty line
            _line = line.strip()
            if not _line:
                continue
            elif line.startswith('BEGIN IONS'):
                spectrum = {}
                # initialize spectrum
                mz_list = []
                intensity_list = []
            elif line.startswith('END IONS'):
                if len(mz_list) == 0:
                    continue
                spectrum['mz_ls'] = mz_list
                spectrum['intensity_ls'] = intensity_list
                spectrum_list.append(spectrum)
                continue
            else:
                # if line contains '=', it is a key-value pair
                if '=' in _line:
                    # split by first '='
                    key, value = _line.split('=', 1)
                    spectrum[key] = value
                else:
                    # if no '=', it is a spectrum pair
                    this_mz, this_int = _line.split()
                    try:
                        mz_list.append(float(this_mz))
                        intensity_list.append(float(this_int))
                    except:
                        continue
                    
        return spectrum_list
    

def write_mgf(spec_list, out_path):
    """
    Write the filtered library to a file.
    """
    with open(out_path, 'w', encoding='utf-8') as f:
        for spec in spec_list:
            f.write('BEGIN IONS\n')
            # f.write(f'NAME={spec["NAME"]}\n')
            f.write(f'PEPMASS={spec["PEPMASS"]}\n')
            # f.write(f'MSLEVEL=2\n')
            # f.write(f'TITLE={spec["TITLE"]}\n')
            # f.write(f'SMILES={spec["SMILES"]}\n')
            # f.write(f'INCHI={spec["INCHI"]}\n')
            # f.write(f'INCHIAUX={spec["INCHIAUX"]}\n')
            # f.write(f'ADDUCT={spec["ADDUCT"]}\n')
            f.write(f'CHARGE=1+\n')
            f.write(f'SCANS={spec["SCANS"]}\n')

            mzs = spec['mz_ls']
            intensities = spec['intensity_ls']
            for mz, intensity in zip(mzs, intensities):
                mz = round(mz, 5)
                intensity = round(intensity, 2)
                f.write(f'{mz} {intensity}\n')

            f.write('END IONS\n\n')

    return


if __name__ == '__main__':
    
    # on server
    # get_human_usi(
    #     merged_all_masst_path='/home/shipei/projects/synlib/masst/data/all_masst_matches_with_metadata.pkl',
    #     out_dir='/home/shipei/projects/synlib/masst/human',
    #     min_matches_per_usi=3
    # )
    
    # on local
    
    ## 25837 MSMS
    # write_human_mgf(
    #     human_pkl_path='masst/human/data/human_masst_matches.pkl',
    #     ms2lib_df_path='data_cleaning/cleaned_data/ms2_all_df.pkl',
    #     ms2lib_mgf_path='data_cleaning/cleaned_data/ms2_all.mgf',
    #     out_path='masst/human/data/human_ms2.mgf',
    #     conjugates_only=False
    # )
    
    ## 2121 MSMS
    # write_human_mgf(
    #     human_pkl_path='masst/human/data/human_only_masst_matches.pkl',
    #     ms2lib_df_path='data_cleaning/cleaned_data/ms2_all_df.pkl',
    #     ms2lib_mgf_path='data_cleaning/cleaned_data/ms2_all.mgf',
    #     out_path='masst/human/data/human_only_ms2.mgf',
    #     conjugates_only=False
    # )
    
    ## 11769 MSMS
    write_human_mgf(
        human_pkl_path='masst/human/data/human_masst_matches.pkl',
        ms2lib_df_path='data_cleaning/cleaned_data/ms2_all_df.pkl',
        ms2lib_mgf_path='data_cleaning/cleaned_data/ms2_all.mgf',
        out_path='masst/human/data/human_ms2_conjugates_only.mgf',
        conjugates_only=True
    )
    
    ## 1563 MSMS
    write_human_mgf(
        human_pkl_path='masst/human/data/human_only_masst_matches.pkl',
        ms2lib_df_path='data_cleaning/cleaned_data/ms2_all_df.pkl',
        ms2lib_mgf_path='data_cleaning/cleaned_data/ms2_all.mgf',
        out_path='masst/human/data/human_only_ms2_conjugates_only.mgf',
        conjugates_only=True
    )