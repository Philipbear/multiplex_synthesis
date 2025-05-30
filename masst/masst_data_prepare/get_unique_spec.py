"""
get unique ms2 from ms2 lib, for MASST
"""

import pandas as pd


def get_unique_spec():
    df = pd.read_pickle('data_cleaning/cleaned_data/ms2_all_df.pkl')

    usi_counts = df['usi'].value_counts().reset_index()
    usi_counts.columns = ['usi', 'frequency']

    # group by usi
    df = df.groupby('usi').first().reset_index()
    df = df[['usi', 'scan']]
    df = df.merge(usi_counts, on='usi', how='left')

    # sort by scan
    df = df.sort_values(by='scan').reset_index(drop=True)

    # save
    df.to_csv('masst/data_prepare/ms2_all_unique_usi.tsv', sep='\t', index=False)


def gen_mgf_with_unique_spec():

    df = pd.read_csv('masst/data_prepare/ms2_all_unique_usi.tsv', sep='\t', low_memory=False)
    df['scan'] = df['scan'].astype(str)
    keep_scan_ls = df['scan'].tolist()

    # read mgf
    spectrum_list = read_mgf_to_df('data_cleaning/cleaned_data/ms2_all.mgf')

    # filter by scan
    spec_list = []
    for spec in spectrum_list:
        if spec['SCANS'] in keep_scan_ls:
            spec_list.append(spec)
        else:
            continue

    # write mgf
    write_mgf(spec_list, 'masst/data_prepare/ms2_all_unique_usi.mgf')

    return


def read_mgf_to_df(library_mgf):
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
                    try:
                        this_mz, this_int = _line.split()
                        mz_list.append(float(this_mz))
                        intensity_list.append(float(this_int))
                    except:
                        print(_line)
                        continue

    return spectrum_list



def write_mgf(spec_list, out_path):
    """
    Write the filtered library to a file.
    """
    with open(out_path, 'a', encoding='utf-8') as f:
        for spec in spec_list:
            f.write('BEGIN IONS\n')
            # f.write(f'NAME={spec["NAME"]}\n')
            f.write(f'PEPMASS={spec["PEPMASS"]}\n')
            # f.write(f'MSLEVEL=2\n')
            # f.write(f'TITLE={spec["TITLE"]}\n')
            # f.write(f'SMILES={spec["SMILES"]}\n')
            # f.write(f'INCHI={spec["INCHI"]}\n')
            # f.write(f'INCHIAUX={spec["INCHIAUX"]}\n')
            # f.write(f'ADDUCT={spec["ADDUCT"]}\n')
            f.write(f'SCANS={spec["SCANS"]}\n')

            mzs = spec['mz_ls']
            intensities = spec['intensity_ls']
            for mz, intensity in zip(mzs, intensities):
                mz = round(mz, 5)
                intensity = round(intensity, 2)
                f.write(f'{mz} {intensity}\n')

            f.write('END IONS\n\n')

    return


if __name__ == '__main__':
    
    get_unique_spec()

    gen_mgf_with_unique_spec()
