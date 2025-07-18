import pandas as pd
from tqdm import tqdm


def main():
    """Process a single TSV file and return categorized data."""
    reactants_df = pd.read_pickle('/Users/shipei/Documents/projects/multiplex_synthesis/data_cleaning/cleaned_data/all_reactants.pkl')
    # compound_name to lower
    reactants_df['compound_name'] = reactants_df['compound_name'].str.lower()
    # dict from name to class
    name_to_class_dict = reactants_df.set_index('compound_name')['np_pathway'].to_dict()

    df = pd.read_pickle('../../data_cleaning/cleaned_data/ms2_all_df.pkl')
    
    # contain '_' in name
    df = df[df['name'].str.contains('_')].reset_index(drop=True)
    # contain only one '_'
    df = df[df['name'].str.count('_') == 1].reset_index(drop=True)

    def split_name(name):
        """Split the name into two parts."""
        name = name.split(' (known isomer')[0].lower()
        names = name.split('_')
        return names[0], names[1]

    df['npp_1'] = None
    df['npp_2'] = None
    for i, row in tqdm(df.iterrows(), total=len(df), desc='Processing dataframe'):
        name = row['name']
        name1, name2 = split_name(name)
        npp_1 = get_primary_class(name_to_class_dict.get(name1, None))
        npp_2 = get_primary_class(name_to_class_dict.get(name2, None))
        df.at[i, 'npp_1'] = npp_1
        df.at[i, 'npp_2'] = npp_2

    # remove rows where npp_1 or npp_2 is None
    df = df[df['npp_1'].notnull() & df['npp_2'].notnull()].reset_index(drop=True)

    # Count occurrences of each combination
    result = df.groupby(['npp_1', 'npp_2']).size().reset_index(name='count')

    # save
    result.to_csv('data/cmpd_class_summary.tsv', sep='\t', index=False)


def get_primary_class(class_ls):
    """Extract the primary class from a comma-separated list."""
    if class_ls is None or not class_ls:
        return None

    # Split by comma and get the first class
    return class_ls[0]


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()