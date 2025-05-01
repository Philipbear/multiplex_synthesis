import pandas as pd
from utils import smiles_to_inchikey, smiles_to_npclassifier
from tqdm import tqdm


def get_reactant_class():

    df = pd.read_csv('raw_data/all_reactants_preprocessed.tsv', sep='\t', low_memory=False)
    df['inchikey'] = df['corrected_SMILES'].apply(smiles_to_inchikey)

    # remove rows with missing inchikey
    df = df.dropna(subset=['inchikey']).reset_index(drop=True)

    df['2d_inchikey'] = df['inchikey'].apply(lambda x: x.split('-')[0])

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

    # save
    df.to_csv('all_reactants.tsv', sep='\t', index=False)


if __name__ == '__main__':
    get_reactant_class()
