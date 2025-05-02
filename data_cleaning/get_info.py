"""
get needed info for ms2 lib or reactants (mass, cmpd class, inchikey, etc.)
"""

import pandas as pd
from utils import smiles_to_formula_inchikey, smiles_to_npclassifier, calc_monoisotopic_mass
from tqdm import tqdm


def get_ms2_lib_info(library_mgf, out_name):
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
                    try:
                        mz_list.append(float(this_mz))
                        intensity_list.append(float(this_int))
                    except:
                        continue

    df = pd.DataFrame(spectrum_list)

    # Get all unique valid SMILES
    unique_smiles = df['SMILES'].dropna().unique()
    print(f"Found {len(unique_smiles)} unique SMILES")

    # Create a dictionary to store results for each unique SMILES
    smiles_data = {}

    # Process each unique SMILES once
    for smiles in tqdm(unique_smiles, desc='Processing unique SMILES'):
        formula, inchikey = smiles_to_formula_inchikey(smiles)

        if formula:
            mass = calc_monoisotopic_mass(formula)
        else:
            mass = None

        if inchikey:
            inchikey_2d = inchikey.split('-')[0]
        else:
            inchikey_2d = None

        np_class, np_superclass, np_pathway = smiles_to_npclassifier(smiles)

        # Store all results in dictionary
        smiles_data[smiles] = {
            'formula': formula,
            'inchikey': inchikey,
            'exact_mass': mass,
            '2d_inchikey': inchikey_2d,
            'np_class': np_class,
            'np_superclass': np_superclass,
            'np_pathway': np_pathway
        }

    # Fill the dataframe using the precomputed values
    for column in ['formula', 'inchikey', 'exact_mass', '2d_inchikey', 'np_class', 'np_superclass', 'np_pathway']:
        df[column] = None  # Initialize columns with None

    # Apply the precomputed values to the dataframe
    for i, row in tqdm(df.iterrows(), total=len(df), desc='Filling dataframe'):
        smiles = row['SMILES']
        if pd.notnull(smiles) and smiles in smiles_data:
            for key, value in smiles_data[smiles].items():
                df.at[i, key] = value

    # Save the results
    df.to_csv(out_name, sep='\t', index=False)
    df.to_pickle(out_name.replace('.tsv', '.pkl'))

    return


def get_reactant_info():

    df = pd.read_csv('raw_data/all_reactants_preprocessed.tsv', sep='\t', low_memory=False)

    df['np_pathway'] = None
    for k in range(3):  # sometimes API fails, try 3 times
        for i, row in tqdm(df.iterrows(), total=len(df), desc='Getting NP classifier'):
            if row['np_pathway'] is not None:
                continue

            smiles = row['corrected_SMILES']
            np_class, np_superclass, np_pathway = smiles_to_npclassifier(smiles)
            df.at[i, 'np_class'] = np_class
            df.at[i, 'np_superclass'] = np_superclass
            df.at[i, 'np_pathway'] = np_pathway

            formula, inchikey = smiles_to_formula_inchikey(smiles)
            df.at[i, 'inchikey'] = inchikey
            if inchikey:
                df.at[i, '2d_inchikey'] = inchikey.split('-')[0]
            else:
                df.at[i, '2d_inchikey'] = None

    # save
    df.to_csv('cleaned_data/all_reactants.tsv', sep='\t', index=False)
    df.to_pickle('cleaned_data/all_reactants.pkl')


if __name__ == '__main__':

    # get_ms2_lib_info('cleaned_data/ms2_all.mgf', 'cleaned_data/ms2_all_df.tsv')
    # get_ms2_lib_info('cleaned_data/ms2_filtered.mgf', 'cleaned_data/ms2_filtered_df.tsv')

    get_reactant_info()
