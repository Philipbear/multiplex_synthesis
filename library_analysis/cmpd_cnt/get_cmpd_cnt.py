import pandas as pd
import os
from tqdm import tqdm


def gen_data():
    df = pd.read_pickle('../../data_cleaning/cleaned_data/ms2_all_df.pkl')
    df['reaction_id'] = df['usi'].apply(lambda x: x.split(':')[0])

    cmpd_csv_folders = ['/Users/shipei/Documents/projects/multiplex_synthesis/data_cleaning/raw_data/cmpd_csv/BA_resplit',
                        '/Users/shipei/Documents/projects/multiplex_synthesis/data_cleaning/raw_data/cmpd_csv/nonBA_resplit']
    cmpd_csv_files = []
    for folder in cmpd_csv_folders:
        for file in os.listdir(folder):
            if file.endswith('.csv') and not file.startswith('.'):
                cmpd_csv_files.append(os.path.join(folder, file))

    all_summary_df = pd.DataFrame()
    for cmpd_csv_file in tqdm(cmpd_csv_files, desc='Processing compound CSV files'):
        cmpd_df = pd.read_csv(cmpd_csv_file, sep=',', low_memory=False)

        # group by unique_sample_id, count the unique SMILES
        cmpd_summary_df = cmpd_df.groupby('unique_sample_id')['SMILES'].nunique().reset_index()
        cmpd_summary_df = cmpd_summary_df.rename(columns={
            'unique_sample_id': 'reaction_id',
            'SMILES': 'expected_cmpd_no'
        })
        all_summary_df = pd.concat([all_summary_df, cmpd_summary_df], ignore_index=True)

    # sort by expected_cmpd_no and dereplicate
    all_summary_df = all_summary_df.sort_values(by='expected_cmpd_no', ascending=False).drop_duplicates(subset=['reaction_id'], keep='first').reset_index(drop=True)
    all_summary_df['observed_cmpd_no'] = all_summary_df['reaction_id'].map(df.groupby('reaction_id')['smiles'].nunique())
    all_summary_df['observed_spec_no'] = all_summary_df['reaction_id'].map(df.groupby('reaction_id').size())

    # save
    all_summary_df.to_csv('cmpd_cnt_summary.tsv', sep='\t', index=False)


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    gen_data()
