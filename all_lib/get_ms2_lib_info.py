"""
get needed info for ms2 lib
name	exact_mass	scan	smiles	inchi	usi	mz	adduct	formula	inchikey	2d_inchikey
"""

import pandas as pd
from utils import smiles_to_formula_inchikey, smiles_to_npclassifier, calc_monoisotopic_mass
from tqdm import tqdm
import os


def get_ms2_lib_info(mgf_folder, out_name, get_npclassifier=False):
    
    mgf_files = os.listdir(mgf_folder)
    mgf_files = [f for f in mgf_files if f.startswith('MULTIPLEX') and not f.startswith('.')]

    df = pd.DataFrame()
    for mgf_file in mgf_files:
        print(f"Processing {mgf_file}")
        mgf_path = os.path.join(mgf_folder, mgf_file)
        mgf_df = read_mgf_to_df(mgf_path)
        # drop mz_ls and intensity_ls cols
        mgf_df = mgf_df.drop(columns=['mz_ls', 'intensity_ls'])
        df = pd.concat([df, mgf_df], ignore_index=True)

    df = df[['NAME', 'PEPMASS', 'SMILES', 'INCHI', 'PUBMED', 'SPECTRUMID']]

    # rename columns
    df.columns = ['name', 'mz', 'smiles', 'inchi', 'usi', 'spec_id']
    
    # split name by the last space
    df['adduct'] = df['name'].apply(lambda x: x.split(' ')[-1])
    df['name'] = df['name'].apply(lambda x: ' '.join(x.split(' ')[:-1]))

    # Get all unique valid SMILES
    unique_smiles = df['smiles'].dropna().unique()
    print(f"Found {len(unique_smiles)} unique SMILES")

    # Create a dictionary to store results for each unique SMILES
    smiles_data = {}

    # Process each unique SMILES once
    for smiles in tqdm(unique_smiles, desc='Processing unique SMILES'):
        formula, inchikey = smiles_to_formula_inchikey(smiles)
        exact_mass = round(calc_monoisotopic_mass(formula), 4)
        inchikey_2d = inchikey.split('-')[0]

        if get_npclassifier:
            np_class, np_superclass, np_pathway = smiles_to_npclassifier(smiles)

            # Store all results in dictionary
            smiles_data[smiles] = {
                'formula': formula,
                'exact_mass': exact_mass,
                'inchikey': inchikey,
                '2d_inchikey': inchikey_2d,
                'np_class': np_class,
                'np_superclass': np_superclass,
                'np_pathway': np_pathway
            }
        else:
            smiles_data[smiles] = {
                'formula': formula,
                'exact_mass': exact_mass,
                'inchikey': inchikey,
                '2d_inchikey': inchikey_2d
            }

    # Fill the dataframe using the precomputed values
    if get_npclassifier:
        info_cols = ['formula', 'exact_mass', 'inchikey', '2d_inchikey', 'np_class', 'np_superclass', 'np_pathway']
    else:
        info_cols = ['formula', 'exact_mass', 'inchikey', '2d_inchikey']
    for column in info_cols:
        df[column] = None  # Initialize columns with None

    # Apply the precomputed values to the dataframe
    for i, row in tqdm(df.iterrows(), total=len(df), desc='Filling dataframe'):
        smiles = row['smiles']
        if pd.notnull(smiles):
            for key, value in smiles_data[smiles].items():
                df.at[i, key] = value

    # Save the results
    df.to_csv(out_name, sep='\t', index=False)
    df.to_pickle(out_name.replace('.tsv', '.pkl'))

    return


def read_mgf_to_df(library_mgf):
    """
    Generate a dataframe from mgf file
    """
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
                    mz_list.append(float(this_mz))
                    intensity_list.append(float(this_int))

    df = pd.DataFrame(spectrum_list)

    return df


def basic_stats(pkl_path):
    df = pd.read_pickle(pkl_path)

    # Print basic statistics
    print("Basic Statistics:")
    print(f"Number of total spectra: {df['spec_id'].nunique()}")
    print(f"Number of unique USIs: {df['usi'].nunique()}")

    print("Considering all possible isomers -----")
    print(f"Number of unique inchikeys: {df['inchikey'].nunique()}")
    print(f"Number of unique 2D inchikeys: {df['2d_inchikey'].nunique()}")

    # dereplicate by USI
    df = df.drop_duplicates(subset='usi')
    print("For unique USIs -----")
    print(f"Number of unique inchikeys: {df['inchikey'].nunique()}")
    print(f"Number of unique 2D inchikeys: {df['2d_inchikey'].nunique()}")


def get_unique_usi_add_npclass():
    
    df = pd.read_pickle('all_lib/data/ms2_all_df.pkl')
    
    # sort by spec_id
    df = df.sort_values(by='spec_id').reset_index(drop=True)
    
    # dereplicate by usi, keep the first one (smallest spec_id)
    df = df.drop_duplicates(subset='usi', keep='first').reset_index(drop=True)
    print(f"Number of unique USIs: {len(df)}")
    
    # add npclassifier info
    unique_smiles = df['smiles'].dropna().unique()
    print(f"Number of unique SMILES: {len(unique_smiles)}")
    smiles_to_npclass = {}
    for smiles in tqdm(unique_smiles, desc='Processing unique SMILES for NPClassifier'):
        np_class, np_superclass, np_pathway = smiles_to_npclassifier(smiles)
        smiles_to_npclass[smiles] = (np_class, np_superclass, np_pathway)
        
    df['np_class'] = df['smiles'].map(lambda x: smiles_to_npclass.get(x, (None, None, None))[0])
    df['np_superclass'] = df['smiles'].map(lambda x: smiles_to_npclass.get(x, (None, None, None))[1])
    df['np_pathway'] = df['smiles'].map(lambda x: smiles_to_npclass.get(x, (None, None, None))[2])
    
    # save
    df.to_csv('all_lib/data/ms2_all_df_unique_usi.tsv', sep='\t', index=False)
    df.to_pickle('all_lib/data/ms2_all_df_unique_usi.pkl')


if __name__ == '__main__':
    
    # get_ms2_lib_info('all_lib/data', 'all_lib/data/ms2_all_df.tsv')
    
    # basic_stats('all_lib/data/ms2_all_df.pkl')
    
    get_unique_usi_add_npclass()    
