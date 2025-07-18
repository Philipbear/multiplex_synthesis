"""
get needed info for ms2 lib
"""

import pandas as pd
from utils import smiles_to_formula_inchikey, smiles_to_npclassifier, calc_monoisotopic_mass
from tqdm import tqdm


def get_ms2_lib_info(library_tsv, out_name, get_npclassifier=False):

    df = pd.read_csv(library_tsv, sep='\t', low_memory=False)

    df = df[['COMPOUND_NAME', 'MOLECULEMASS', 'EXTRACTSCAN', 'SMILES', 'INCHI', 'PUBMED',
             'EXACTMASS', 'ADDUCT']]

    # rename columns
    df.columns = ['name', 'exact_mass', 'scan', 'smiles', 'inchi', 'usi', 'mz', 'adduct']

    # Get all unique valid SMILES
    unique_smiles = df['smiles'].dropna().unique()
    print(f"Found {len(unique_smiles)} unique SMILES")

    # Create a dictionary to store results for each unique SMILES
    smiles_data = {}

    # Process each unique SMILES once
    for smiles in tqdm(unique_smiles, desc='Processing unique SMILES'):
        formula, inchikey = smiles_to_formula_inchikey(smiles)

        if inchikey:
            inchikey_2d = inchikey.split('-')[0]
        else:
            inchikey_2d = None

        if get_npclassifier:
            np_class, np_superclass, np_pathway = smiles_to_npclassifier(smiles)

            # Store all results in dictionary
            smiles_data[smiles] = {
                'formula': formula,
                'inchikey': inchikey,
                '2d_inchikey': inchikey_2d,
                'np_class': np_class,
                'np_superclass': np_superclass,
                'np_pathway': np_pathway
            }
        else:
            smiles_data[smiles] = {
                'formula': formula,
                'inchikey': inchikey,
                '2d_inchikey': inchikey_2d
            }

    # Fill the dataframe using the precomputed values
    if get_npclassifier:
        info_cols = ['formula', 'inchikey', '2d_inchikey', 'np_class', 'np_superclass', 'np_pathway']
    else:
        info_cols = ['formula', 'inchikey', '2d_inchikey']
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


if __name__ == '__main__':

    get_ms2_lib_info('cleaned_data/ms2_all.tsv', 'cleaned_data/ms2_all_df.tsv', get_npclassifier=False)
    get_ms2_lib_info('cleaned_data/ms2_filtered.tsv', 'cleaned_data/ms2_filtered_df.tsv')
