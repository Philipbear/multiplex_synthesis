import pandas as pd


def main():
    df = pd.read_csv('masst/human/data/human_only_usis_raw_count.tsv', sep='\t')

    usis = df['lib_usi'].unique().tolist()

    ms2_df = pd.read_pickle('all_lib/data/ms2_all_df_unique_usi.pkl')
    ms2_df = ms2_df[ms2_df['usi'].isin(usis)].reset_index(drop=True)

    # save
    ms2_df.to_csv('masst/human/data/human_only_usis.tsv', sep='\t', index=False)
    
    ## how many drugs
    drugs = pd.read_csv('all_lib/data/drug_df.tsv', sep='\t')
    drug_usis = drugs['PUBMED'].unique().tolist()
    
    drug_usis_in_masst = list(set(usis) & set(drug_usis))
    print(f"{len(drug_usis_in_masst):,} drugs in MASST human only matches")
    
    
def analysis():
    df = pd.read_csv('masst/human/data/human_only_usis.tsv', sep='\t')
    print(df['np_pathway'].value_counts())
    print(df['np_superclass'].value_counts())
    print(df['np_class'].value_counts()[:20])
    

if __name__ == "__main__":
    # main()    
    analysis()

