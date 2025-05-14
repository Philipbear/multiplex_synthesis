import pandas as pd

"""
['name', 'exact_mass', 'scan', 'smiles', 'inchi', 'usi', 'mz', 'adduct',
       'formula', 'inchikey', '2d_inchikey']
"""

def stats():
    df = pd.read_pickle('data_cleaning/cleaned_data/ms2_all_df.pkl')
    print(df.columns)

    print('All library:')    
    print('Spectra:', df.shape[0])
    print('Unique 3D structures:', df['inchikey'].nunique())
    print('Unique 2D structures:', df['2d_inchikey'].nunique())
    print('Unique USIs:', df['usi'].nunique())
    # unique structures for unique USIs
    df = df.drop_duplicates(subset=['usi'])
    print('Unique 3D structures for unique USIs (one annotation per USI):', df['inchikey'].nunique())
    print('Unique 2D structures for unique USIs (one annotation per USI):', df['2d_inchikey'].nunique())
    
    df = pd.read_pickle('data_cleaning/cleaned_data/ms2_filtered_df.pkl')
    print('Filtered library:')
    print('Spectra:', df.shape[0])
    print('Unique 3D structures:', df['inchikey'].nunique())
    print('Unique 2D structures:', df['2d_inchikey'].nunique())
    print('Unique USIs:', df['usi'].nunique())
    # unique structures for unique USIs
    df = df.drop_duplicates(subset=['usi'])
    print('Unique 3D structures for unique USIs (one annotation per USI):', df['inchikey'].nunique())
    print('Unique 2D structures for unique USIs (one annotation per USI):', df['2d_inchikey'].nunique())
    
if __name__ == '__main__':
    stats()
    