import os
import pandas as pd

"""
Merge mgf files into one mgf file
"""

def main(folder_path, out_mgf):

    # remove output files if they exist
    if os.path.isfile(out_mgf):
        os.remove(out_mgf)

    # list all mgf files in the folder
    mgf_files = [f for f in os.listdir(folder_path) if f.endswith('.mgf') and not f.startswith('.')]
    mgf_files = sorted(mgf_files)

    scan_no = 1
    for mgf in mgf_files:
        mgf_path = os.path.join(folder_path, mgf)
        print('Processing:', mgf_path)

        # read mgf to df
        spec_list = read_mgf_to_df(mgf_path)
        print(f'Found {len(spec_list)} spectra')

        # write mgf
        scan_start = scan_no
        scan_no = write_mgf(spec_list, out_mgf, scans_start=scan_start)
        print(f'Wrote {len(spec_list)} spectra')


def write_mgf(spec_list, out_path, scans_start=0):
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
            # f.write(f'ADDUCT={spec["ADDUCT"]}\n')
            f.write(f'SPECTRUMID={spec["SPECTRUMID"]}\n')
            f.write(f'SCANS={scans_start}\n')

            mzs = spec['mz_ls']
            intensities = spec['intensity_ls']
            for mz, intensity in zip(mzs, intensities):
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
                    # if no '=', it is a spectrum pair
                    this_mz, this_int = _line.split()
                    mz_list.append(float(this_mz))
                    intensity_list.append(float(this_int))

    return spectrum_list


if __name__ == '__main__':

    main('all_lib/data', 'all_lib/data/ms2_all.mgf')
