import pandas as pd
from tqdm import tqdm


# Define the class order for natural product pathways
CLASS_ORDER = [
    'Fatty acids',
    'Shikimates and Phenylpropanoids',
    'Terpenoids',
    'Alkaloids',
    'Amino acids and Peptides',
    'Carbohydrates',
    'Polyketides'
]


def main():
    """Process a single TSV file and return categorized data."""
    reactants_df = pd.read_pickle('/Users/shipei/Documents/projects/multiplex_synthesis/data_cleaning/cleaned_data/all_reactants.pkl')
    # compound_name to lower
    reactants_df['compound_name'] = reactants_df['compound_name'].str.lower()
    # dict from name to np_pathway
    name_to_np_pathway = reactants_df.set_index('compound_name')['np_pathway'].to_dict()

    df = pd.read_pickle('../../data_cleaning/cleaned_data/ms2_all_df.pkl')


    def split_name(name):
        """Split the name into two parts."""
        name = name.split(' (known isomer')[0].lower()
        if '_' not in name:
            return name, None, None
        names = name.split('_')
        if len(names) == 2:
            return names[0], names[1], None
        else:
            return names[0], names[1], '_'.join(names[2:])

    df['npp_1'] = None
    df['npp_2'] = None
    df['npp_3'] = None
    for i, row in tqdm(df.iterrows(), total=len(df), desc='Processing dataframe'):
        name = row['name']
        name1, name2, name3 = split_name(name)
        npp_1 = get_primary_pathway(name_to_np_pathway.get(name1, None))
        npp_2 = get_primary_pathway(name_to_np_pathway.get(name2, None))
        npp_3 = get_primary_pathway(name_to_np_pathway.get(name3, None))
        df.at[i, 'npp_1'] = npp_1
        df.at[i, 'npp_2'] = npp_2
        df.at[i, 'npp_3'] = npp_3

    # df = df[df['npp_2'].notnull()].reset_index(drop=True)

    # Count occurrences of each combination, including None values
    result = df.fillna('None').groupby(['npp_1', 'npp_2', 'npp_3']).size().reset_index(name='count')

    # save
    result.to_csv('cmpd_class_summary.tsv', sep='\t', index=False)


def get_primary_pathway(pathway_ls):
    """Extract the primary pathway from a comma-separated list."""
    if pathway_ls is None or not pathway_ls:
        return None

    # Split by comma and get the first pathway
    return pathway_ls[0]



if __name__ == '__main__':

    main()
