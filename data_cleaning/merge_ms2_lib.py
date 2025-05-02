import pandas as pd
import os

"""
Merge mgf and tsv files (from GNPS2 workflow output) into one mgf file and one tsv file

mgf: NAME, PEPMASS, MSLEVEL, TITLE, SMILES, INCHI, INCHIAUX, ADDUCT, SCANS
"""

def main(folder_path, out_tsv, out_mgf):

    # remove output files if they exist
    if os.path.isfile(out_tsv):
        os.remove(out_tsv)
    if os.path.isfile(out_mgf):
        os.remove(out_mgf)

    # list all mgf files in the folder
    mgf_files = [f for f in os.listdir(folder_path) if f.endswith('.mgf')]
    mgf_files = sorted(mgf_files)

    scan_no = 1
    for mgf in mgf_files:
        mgf_path = os.path.join(folder_path, mgf)
        print('Processing:', mgf_path)

        # read mgf to df
        spec_list = read_mgf_to_df(mgf_path)
        # write mgf
        scan_start = scan_no
        scan_no = write_mgf(spec_list, out_mgf, scans_start=scan_no)
        print(f'Wrote {len(spec_list)} spectra')

        # read tsv to df
        tsv_path = os.path.join(folder_path, mgf.replace('.mgf', '.tsv'))
        df = pd.read_csv(tsv_path, sep='\t', low_memory=False)

        # renumber EXTRACTSCAN from scan_start
        df['EXTRACTSCAN'] = df['EXTRACTSCAN'].apply(lambda x: int(x) + scan_start - 1)

        df['FILENAME'] = os.path.basename(out_mgf)
        df['INCHIAUX'] = None
        df['COMPOUND_NAME'] = df['COMPOUND_NAME'].apply(lambda x: x.replace('structural isomers', 'isomers').replace('peaks in run', 'peaks'))
        # round to 5 decimal places
        df['MOLECULEMASS'] = df['MOLECULEMASS'].apply(lambda x: round(x, 5))
        df['EXACTMASS'] = df['EXACTMASS'].apply(lambda x: round(x, 5))
        df['ADDUCT'] = df['ADDUCT'].apply(lambda x: x.split('[')[1].split(']')[0])

        # append df to tsv
        num_rows = append_df_to_tsv(df, out_tsv)
        print(f'Appended {num_rows} spectra')


def append_df_to_tsv(df, file_path):

    # Check if file exists
    file_exists = os.path.isfile(file_path)

    if file_exists:
        # Append without header
        df.to_csv(file_path, sep='\t', mode='a', header=False, index=False, na_rep='N/A')
    else:
        # Create new file with header
        df.to_csv(file_path, sep='\t', mode='w', header=True, index=False, na_rep='N/A')

    return df.shape[0]  # Return number of rows appended


def write_mgf(spec_list, out_path, scans_start=0):
    """
    Write the filtered library to a file.
    """
    with open(out_path, 'a', encoding='utf-8') as f:
        for spec in spec_list:
            f.write('BEGIN IONS\n')
            # f.write(f'NAME={spec["NAME"]}\n')
            # f.write(f'PEPMASS={spec["PEPMASS"]}\n')
            # f.write(f'MSLEVEL=2\n')
            # f.write(f'TITLE={spec["TITLE"]}\n')
            # f.write(f'SMILES={spec["SMILES"]}\n')
            # f.write(f'INCHI={spec["INCHI"]}\n')
            # f.write(f'INCHIAUX={spec["INCHIAUX"]}\n')
            # f.write(f'ADDUCT={spec["ADDUCT"]}\n')
            f.write(f'SCANS={scans_start}\n')

            mzs = spec['mz_ls']
            intensities = spec['intensity_ls']
            for mz, intensity in zip(mzs, intensities):
                mz = round(mz, 5)
                intensity = round(intensity, 2)
                f.write(f'{mz} {intensity}\n')

            f.write('END IONS\n\n')
            scans_start += 1

    return scans_start


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

                    ################### special case ###################
                    if _line == 'NCCc1cc(O)c(O)cc1':
                        spectrum['SMILES'] = _line
                        continue

                    # if no '=', it is a spectrum pair
                    try:
                        this_mz, this_int = _line.split()
                        mz_list.append(float(this_mz))
                        intensity_list.append(float(this_int))
                    except:
                        print(_line)
                        continue

    return spectrum_list


if __name__ == '__main__':

    main('raw_data/all', 'cleaned_data/ms2_all.tsv', 'cleaned_data/ms2_all.mgf')
    main('raw_data/filtered', 'cleaned_data/ms2_filtered.tsv', 'cleaned_data/ms2_filtered.mgf')

    # df = pd.read_csv('cleaned_data/ms2_all.tsv', sep='\t', low_memory=False)
    # df = pd.read_csv('cleaned_data/ms2_filtered.tsv', sep='\t', low_memory=False)

    # print(df.shape)
    # print(df['INCHI'].nunique())
