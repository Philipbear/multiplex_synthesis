import pandas as pd



def summary(df_path='data/all_masst_matches.tsv'):
    
    df = pd.read_csv(df_path, sep='\t', low_memory=False)
    
    # remove matches to syn libraries
    df = df[df['dataset'] != 'MSV000094559'].reset_index(drop=True)

