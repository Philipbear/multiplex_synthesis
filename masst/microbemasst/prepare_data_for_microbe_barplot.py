import pandas as pd


def summarize_microbemasst_class_results():
    # we need this to add np_pathway info to microbemasst results
    ms2_df = pd.read_pickle('all_lib/data/ms2_all_df_unique_usi.pkl')
    ms2_df = ms2_df[['usi', 'np_pathway']]
    ms2_df['np_pathway'] = ms2_df['np_pathway'].apply(lambda x: x[0] if isinstance(x, list) else 'Unclassified')
    # rename usi to lib_usi
    ms2_df = ms2_df.rename(columns={'usi': 'lib_usi'})
    
    # load microbemasst results
    all_microbemasst_path = "masst/microbemasst/data/merged_microbemasst_table.tsv"
    all_microbemasst_df = pd.read_csv(all_microbemasst_path, sep='\t')
    all_microbemasst_df['Taxa_NCBI'] = all_microbemasst_df['Taxa_NCBI'].astype(str)
    
    # add np_pathway info
    all_microbemasst_df = all_microbemasst_df.merge(ms2_df, on='lib_usi', how='left')
    
    # load lineage info    
    lineage_df = pd.read_csv('masst/umap/data/all_redu_lineage.tsv', sep='\t')
    lineage_df = lineage_df[['NCBI', 'class', 'phylum']]
    # rename NCBI to Taxa_NCBI
    lineage_df = lineage_df.rename(columns={'NCBI': 'Taxa_NCBI'})
    # convert to str
    lineage_df['Taxa_NCBI'] = lineage_df['Taxa_NCBI'].astype(str)

    # add class and phylum info
    all_microbemasst_df = all_microbemasst_df.merge(lineage_df, on='Taxa_NCBI', how='left')
    
    # generate summary table (cols: class, phylum, all unique np_pathway)
    np_pathways = ['Alkaloids', 'Amino acids and peptides', 'Fatty acids', 'Shikimates and phenylpropanoids', 'Terpenoids', 'Unclassified']
    summary_list = []
    classes = all_microbemasst_df['class'].dropna().unique()
    for c in classes:
        class_df = all_microbemasst_df[all_microbemasst_df['class'] == c].reset_index(drop=True)
        # get most common phylum
        phylum = class_df['phylum'].mode()[0] if not class_df['phylum'].isna().all() else 'Unknown'
        total_entries = len(class_df)
        entry = {'class': c, 'phylum': phylum, 'total_entries': total_entries}
        for np in np_pathways:
            count = len(class_df[class_df['np_pathway'] == np])
            entry[np] = count
        summary_list.append(entry)
        
    summary_df = pd.DataFrame(summary_list)
    summary_out_path = "masst/microbemasst/data/microbemasst_class_summary.tsv"
    summary_df.to_csv(summary_out_path, sep='\t', index=False)
    print(f"Saved microbemasst class summary to {summary_out_path}, total {len(summary_df)} classes.")
    return


if __name__ == '__main__':
    
    summarize_microbemasst_class_results()

