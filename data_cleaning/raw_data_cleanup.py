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


def read_mgf_to_df(library_mgf, out_name=None):
    """
    Generate a dataframe from mgf file
    """
    with open(library_mgf, 'r') as file:
        spectrum_list = []
        for line in file:
            # empty line
            _line = line.strip()
            if not _line:
                continue
            elif line.startswith('BEGIN IONS'):
                spectrum = {}
                # initialize spectrum
                mz_list = []
                intensity_list = []
            elif line.startswith('END IONS'):
                if len(mz_list) == 0:
                    continue
                spectrum['mz_ls'] = mz_list
                spectrum['intensity_ls'] = intensity_list
                spectrum_list.append(spectrum)
                continue
            else:
                # if line contains '=', it is a key-value pair
                if '=' in _line:
                    # split by first '='
                    key, value = _line.split('=', 1)
                    spectrum[key] = value
                else:
                    # if no '=', it is a spectrum pair
                    this_mz, this_int = _line.split()
                    try:
                        mz_list.append(float(this_mz))
                        intensity_list.append(float(this_int))
                    except:
                        continue

    df = pd.DataFrame(spectrum_list)

    if out_name is None:
        out_name = library_mgf.split('/')[-1].split('.')[0] + '_df.tsv'
    df.to_csv(out_name, sep='\t', index=False)

    return


if __name__ == '__main__':

    clean_up_reaction_records()
    read_mgf_to_df('ms2_all.mgf', 'ms2_all_df.tsv')
    read_mgf_to_df('ms2_filtered.mgf', 'ms2_filtered_df.tsv')
