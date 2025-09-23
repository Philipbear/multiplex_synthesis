import pandas as pd
import pickle


def gen_shown_rank(rank='phylum'):

    df = pd.read_csv('masst/umap/data/all_redu_lineage.tsv', sep='\t', low_memory=False)

    # remove rows with rank empty or null
    df = df[df[rank].notnull() & (df[rank] != '')].reset_index(drop=True)

    df['shown_rank'] = df[rank]

    # # Specifically for humans
    # df.loc[df['NCBI'] == 9606, 'shown_rank'] = 'Humans'

    # # For rodents
    # rodent_species = [10088, 10090, 10105, 10114, 10116]
    # df.loc[df['NCBI'].isin(rodent_species), 'shown_rank'] = 'Rodents'

    # save dict from NCBI to shown_rank
    ncbi_to_shown_rank = df.set_index('NCBI')['shown_rank'].to_dict()

    # save as pickle
    with open(f'masst/umap/data/ncbi_to_{rank}.pkl', 'wb') as f:
        pickle.dump(ncbi_to_shown_rank, f)
    
##########################

def gen_show_name():

    def get_show_name(row):
        if pd.isnull(row['kingdom']):
            return row['superkingdom']    
        return row['kingdom']

    df = pd.read_csv('masst/umap/data/all_redu_lineage.tsv', sep='\t', low_memory=False)
    df['show_name'] = df.apply(lambda x: get_show_name(x), axis=1)

    # map: Viridiplantae: Plants; Metazoa: Animals; Fungi: Fungi; Bacteria: Bacteria; All others: Others
    df['show_name'] = df['show_name'].replace({
        'Viridiplantae': 'Plants',
        'Metazoa': 'Animals',
        'Fungi': 'Fungi',
        'Bacteria': 'Bacteria'
    })
    # fill NaN with 'Others'
    mask = ~df['show_name'].isin(['Plants', 'Animals', 'Fungi', 'Bacteria'])
    df.loc[mask, 'show_name'] = 'Others'

    # Specifically for human
    df.loc[df['NCBI'] == 9606, 'show_name'] = 'Humans'

    # For rodents use 'Rodents'
    rodent_species = [10088, 10090, 10105, 10114, 10116]
    df.loc[df['NCBI'].isin(rodent_species), 'show_name'] = 'Rodents'

    print("show names:", df['show_name'].value_counts())

    # save dict from NCBI to show_name
    ncbi_to_show_name = df.set_index('NCBI')['show_name'].to_dict()

    # save as pickle
    with open('masst/umap/data/ncbi_to_show_name.pkl', 'wb') as f:
        pickle.dump(ncbi_to_show_name, f)


if __name__ == '__main__':
    ranks = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    
    for rank in ranks:
        print(f"Generating shown_rank for {rank}...")
        gen_shown_rank(rank)
    
    print("Generating show_name mapping...")
    gen_show_name()
    
    print("All mappings generated successfully!")