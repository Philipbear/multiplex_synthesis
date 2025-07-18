"""
get needed info for reactants (mass, cmpd class, inchikey, etc.)
"""

import pandas as pd
from utils import smiles_to_formula_inchikey, smiles_to_npclassifier, calc_monoisotopic_mass
from tqdm import tqdm


def get_reactant_info():

    # if not already done, load raw data
    try:
        df = pd.read_pickle('data_cleaning/cleaned_data/all_reactants.pkl')
    except FileNotFoundError:
        print('Loading raw data...')
        df = pd.read_csv('data_cleaning/raw_data/all_reactants_preprocessed.tsv', sep='\t', low_memory=False)
        df['np_class'] = None
        df['np_superclass'] = None
        df['np_pathway'] = None

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
    df.to_csv('data_cleaning/cleaned_data/all_reactants.tsv', sep='\t', index=False)
    df.to_pickle('data_cleaning/cleaned_data/all_reactants.pkl')


if __name__ == '__main__':

    get_reactant_info()
