'''
get all unique USIs that have matches to humans, and prepare mgf, for mol networking
'''

import pandas as pd
import os
from tqdm import tqdm


def write_human_mgf(human_df_path, ms2lib_df_path, ms2lib_mgf_path, out_path):
    
    # Load MS2Lib DataFrame
    ms2lib_df = pd.read_pickle(ms2lib_df_path)
    # dict from usi to spec_id
    usi_to_spec_id = ms2lib_df.set_index('usi')['spec_id'].to_dict()

    # load human matches
    print(f"Loading human matches from {human_df_path}...")
    human_df = pd.read_csv(human_df_path, sep='\t')
    
    human_df['spec_id'] = human_df['lib_usi'].map(usi_to_spec_id)
    reserved_spec_ids = human_df['spec_id'].tolist()
    
    all_ms2_list = read_mgf(ms2lib_mgf_path)
    print(f"Loaded {len(all_ms2_list):,} spectra from MS2Lib MGF file.")
    
    # filter spectra for human matches
    human_ms2_list = []
    for spec in tqdm(all_ms2_list, desc="Filtering spectra"):
        if spec['SPECTRUMID'] in reserved_spec_ids:
            human_ms2_list.append(spec)
    print(f"Filtered to {len(human_ms2_list):,} spectra matching human USIs.")
    
    # write to MGF file
    write_mgf(human_ms2_list, out_path)
    print(f"Written {len(human_ms2_list):,} spectra to {out_path}.")


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
        for i, spec in enumerate(spec_list, 1):
            f.write('BEGIN IONS\n')
            # f.write(f'NAME={spec["NAME"]}\n')
            f.write(f'PEPMASS={spec["PEPMASS"]}\n')
            # f.write(f'MSLEVEL=2\n')
            # f.write(f'TITLE={spec["TITLE"]}\n')
            # f.write(f'SMILES={spec["SMILES"]}\n')
            # f.write(f'INCHI={spec["INCHI"]}\n')
            # f.write(f'INCHIAUX={spec["INCHIAUX"]}\n')
            # f.write(f'ADDUCT={spec["ADDUCT"]}\n')
            f.write(f'SPECTRUMID={spec["SPECTRUMID"]}\n')
            f.write(f'CHARGE=1+\n')
            f.write(f'SCANS={i}\n')

            mzs = spec['mz_ls']
            intensities = spec['intensity_ls']
            for mz, intensity in zip(mzs, intensities):
                mz = round(mz, 5)
                intensity = round(intensity, 2)
                f.write(f'{mz} {intensity}\n')

            f.write('END IONS\n\n')

    return


if __name__ == '__main__':
    
    # 2609 MSMS
    write_human_mgf(
        human_df_path='masst/human/data/human_only_usis_raw_count.tsv',
        ms2lib_df_path='all_lib/data/ms2_all_df_unique_usi.pkl',
        ms2lib_mgf_path='all_lib/data/ms2_all.mgf',
        out_path='masst/human/data/human_only_ms2.mgf'
    )