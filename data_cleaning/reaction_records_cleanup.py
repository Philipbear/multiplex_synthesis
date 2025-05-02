import pandas as pd


def clean_up_reaction_records():
    df = pd.read_csv('raw_data/reaction_records_raw.tsv', sep='\t', low_memory=False)

    df = df[df['Reaction_ID'].notnull()].reset_index(drop=True)

    df['Reaction_ID'] = df['Reaction_ID'].apply(lambda x: x.split('.mz')[0])

    # dereplicate by reaction id
    df = df.drop_duplicates(subset=['Reaction_ID'], keep='first')

    print('total number of records:', len(df))

    # save
    df.to_csv('reaction_records.tsv', sep='\t', index=False)



if __name__ == '__main__':

    clean_up_reaction_records()
