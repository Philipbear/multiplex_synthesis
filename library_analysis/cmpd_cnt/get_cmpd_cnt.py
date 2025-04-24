import pandas as pd
from tqdm import tqdm


def gen_data():
    df = pd.read_csv('../../raw_data/all_ms2_df.tsv', sep='\t', low_memory=False)

    reaction_records = pd.read_csv('../../raw_data/reaction_records.tsv', sep='\t')

    # reaction id
    df['reaction_id'] = df['TITLE'].apply(lambda x: x.split('.mz')[0].split(':')[2])  ######


    def get_product_no(x):
        try:
            return int(x)
        except:
            return None

    reaction_records['No_of_products_expected'] = reaction_records['No_of_products_expected'].apply(get_product_no)
    reaction_records = reaction_records[reaction_records['No_of_products_expected'].notnull()].reset_index(drop=True)
    reaction_records = reaction_records[reaction_records['No_of_products_expected'] > 0].reset_index(drop=True)

    # fill in observed products
    for i, row in tqdm(reaction_records.iterrows(), total=len(reaction_records)):
        rxn_id = row['Reaction_ID']

        sub_df = df[df['reaction_id'] == rxn_id]

        observed_products = len(sub_df['SMILES'].unique())

        reaction_records.at[i, 'observed_products'] = observed_products

    # save
    reaction_records.to_csv('reaction_df.tsv', sep='\t', index=False)


if __name__ == '__main__':
    gen_data()

