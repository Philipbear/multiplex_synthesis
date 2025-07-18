"""
get needed info for ms2 lib
"""

import pandas as pd
from utils import smiles_to_npclassifier
from tqdm import tqdm


def get_ms2_lib_info():

    df = pd.read_pickle('data_cleaning/cleaned_data/ms2_all_df.pkl')
    
    # dereplicate the dataframe by 'usi' and keep the first occurrence
    df = df.drop_duplicates(subset=['usi'], keep='first').reset_index(drop=True)

    # Get all unique valid SMILES
    unique_smiles = df['smiles'].dropna().unique()
    print(f"Found {len(unique_smiles)} unique SMILES")

    # Create a dictionary to store results for each unique SMILES
    smiles_data = {}

    # Process each unique SMILES once
    for smiles in tqdm(unique_smiles, desc='Processing unique SMILES'):

        np_class, np_superclass, np_pathway = smiles_to_npclassifier(smiles)

        # Store all results in dictionary
        smiles_data[smiles] = {
            'np_class': np_class,
            'np_superclass': np_superclass,
            'np_pathway': np_pathway
        }
        
    # Fill the dataframe using the precomputed values
    for column in ['np_class', 'np_superclass', 'np_pathway']:
        df[column] = None  # Initialize columns with None

    # Apply the precomputed values to the dataframe
    for i, row in tqdm(df.iterrows(), total=len(df), desc='Filling dataframe'):
        smiles = row['smiles']
        if pd.notnull(smiles):
            for key, value in smiles_data[smiles].items():
                df.at[i, key] = value

    # Save the results
    df.to_csv('masst/umap/data/ms2_all_df_unique_usi.tsv', sep='\t', index=False)
    df.to_pickle('masst/umap/data/ms2_all_df_unique_usi.pkl')

    return


if __name__ == '__main__':

    get_ms2_lib_info()
    
