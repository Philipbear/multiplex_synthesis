import pandas as pd
import os
from tqdm import tqdm
import numpy as np


def get_drug_and_conjugates(out_path, drug_name='ibuprofen', use_filtered_ms2lib=False):
    """
    Get the drug and its conjugates from the MS2 library.
    
    drug found by inchikey, then find all conjugates by the drug name.
    """
    os.makedirs(out_path, exist_ok=True)
        
    def get_conjugates_from_ms2_df(ms2_df_path, name):
        print(f"Processing MS2 data from {ms2_df_path}...")

        ms2_df = pd.read_pickle(ms2_df_path)
        ms2_df['name'] = ms2_df['name'].apply(lambda x: x.split(' (known')[0])
        
        # adduct filtering: only M+H
        ms2_df = ms2_df[ms2_df['adduct'] == 'M+H'].reset_index(drop=True)
        
        # filter to names that contain the drug name
        matched_df = ms2_df[ms2_df['name'].str.contains(name, case=False)].reset_index(drop=True)
        
        # sort by scan
        matched_df.sort_values(by='scan', inplace=True)

        return matched_df

    # get the conjugates from the MS2 library
    out_ms2_list = []
    
    _lib_txt = 'all' if not use_filtered_ms2lib else 'filtered'

    #########################
    # drug library
    matched_df = get_conjugates_from_ms2_df(f'drug_data/cleaned_data/ms2_{_lib_txt}_df.pkl', drug_name)
    
    # #########
    # matched_df = matched_df[~matched_df['name'].str.contains('1-hydroxyIbuprofen', case=False)].reset_index(drop=True)
    # #########

    # filter spectra
    reserved_scans = matched_df['scan'].unique().tolist()
    all_ms2_list = read_mgf(f'drug_data/cleaned_data/ms2_{_lib_txt}.mgf')
    print(f"Loaded {len(all_ms2_list):,} spectra from MS2Lib MGF file.")

    final_scans = []  # pass both drug and MassQL
    for spec in tqdm(all_ms2_list, desc="Filtering spectra"):
        if int(spec['SCANS']) not in reserved_scans:
            continue

        # _peaks = np.array(list(zip(spec['mz_ls'], spec['intensity_ls'])))
        # _mask = _ms2_filter_by_frags(_peaks, MZ_1, mz_tol=0.01) or _ms2_filter_by_frags(_peaks, MZ_2, mz_tol=0.01)
        # if not _mask:
        #     continue
        
        final_scans.append(int(spec['SCANS']))
        out_ms2_list.append(spec)

    matched_df = matched_df[matched_df['scan'].isin(final_scans)].reset_index(drop=True)
    print(f"Filtered to {len(out_ms2_list):,} spectra.")

    #########################
    # write the filtered spectra to a new MGF file
    # first, re-number the scans in the output list
    for i, spec in enumerate(out_ms2_list):
        spec['SCANS'] = i + 1

    out_mgf_path = os.path.join(out_path, f'{drug_name}_conjugates.mgf')
    write_mgf(out_ms2_list, out_mgf_path)
    print(f"Filtered spectra written to {out_mgf_path}")
    
    # new scan: from 1 to n, where n is the number of rows
    matched_df['new_scan'] = range(1, len(matched_df) + 1)
    matched_df.to_csv(os.path.join(out_path, f'{drug_name}_ms2_df.tsv'), sep='\t', index=False)


def read_mgf(library_mgf):
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
                    
        return spectrum_list
    

def write_mgf(spec_list, out_path):
    """
    Write the filtered library to a file.
    """
    with open(out_path, 'w', encoding='utf-8') as f:
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
            f.write(f'CHARGE=1+\n')
            f.write(f'SCANS={spec["SCANS"]}\n')

            mzs = spec['mz_ls']
            intensities = spec['intensity_ls']
            for mz, intensity in zip(mzs, intensities):
                mz = round(mz, 5)
                intensity = round(intensity, 2)
                f.write(f'{mz} {intensity}\n')

            f.write('END IONS\n\n')

    return


def _ms2_filter_by_frags(peaks, t_frag_mz, mz_tol=0.01):
    """
    Filter MS2 peaks
    """
    idx = np.abs(peaks[:, 0] - t_frag_mz) <= mz_tol
    if not np.any(idx):
        return False
    return True


def filter_masst_redu_df(drug_name, drug_out_path, all_results='drug_data/data/all_masst_matches_with_metadata.pkl'):
    
    df = pd.read_pickle(all_results)

    drug_df = pd.read_csv(f'{drug_out_path}/{drug_name}_ms2_df.tsv', sep='\t')
    reserved_usis = drug_df['usi'].unique().tolist()

    filtered_df = df[df['lib_usi'].isin(reserved_usis)]

    # save
    filtered_df.to_pickle(f'{drug_out_path}/{drug_name}_masst_matches.pkl')
    filtered_df.to_csv(f'{drug_out_path}/{drug_name}_masst_matches.tsv', sep='\t', index=False)


if __name__ == "__main__":
    
    MZ_1 = 161.1325
    MZ_2 = 207.1380
    
    out_path = 'target_drugs/ibuprofen_conjugates/data'
    get_drug_and_conjugates(out_path, drug_name='ibuprofen')
    
    filter_masst_redu_df('ibuprofen', out_path)
