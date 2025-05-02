import pandas as pd
import pickle


def gen_data():
    df = pd.read_csv('../../raw_data/all_ms2_df.tsv', sep='\t', low_memory=False)

    print('unique cmpd inchis:', len(df['INCHI'].unique()))
    print('unique cmpd smiles:', len(df['SMILES'].unique()))

    # dereplicate by inchikey
    df = df.drop_duplicates(subset=['inchikey'], keep='first')

    df = df[df['exact_mass'].notnull()].reset_index(drop=True)

    mono_mass_ls = df['exact_mass'].tolist()

    # save
    with open('mono_mass_ls.pkl', 'wb') as f:
        pickle.dump(mono_mass_ls, f)


if __name__ == '__main__':
    gen_data()

