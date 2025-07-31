import pandas as pd
import os
from tqdm import tqdm


def get_drug_and_conjugates(out_path, drug_name='ibuprofen', inchikey='HEFNNWSXXWATRW-UHFFFAOYSA-N'):
    """
    Get the drug and its conjugates from the MS2 library.
    
    drug found by inchikey, then find all conjugates by the drug name.
    """
    os.makedirs(out_path, exist_ok=True)
        
    # load reactants, find all names associated with the given inchikey
    reactant_df = pd.read_pickle('data_cleaning/cleaned_data/all_reactants.pkl')
    reactant_df = reactant_df[reactant_df['inchikey'] == inchikey].reset_index(drop=True)    
    all_names = reactant_df['compound_name'].unique().tolist()
    print(f"Found {len(all_names)} names for the given inchikey: {inchikey}")
    print(f"Names: {', '.join(all_names)}")

    def get_conjugates_from_ms2_df(ms2_df_path, inchikey, all_names):
        print(f"Processing MS2 data from {ms2_df_path}...")

        ms2_df = pd.read_pickle(ms2_df_path)
        ms2_df['name'] = ms2_df['name'].apply(lambda x: x.split(' (known')[0])
        
        # first part, drug itself
        drug_df = ms2_df[ms2_df['inchikey'] == inchikey].reset_index(drop=True)
        
        # second part, conjugates
        conjugate_df = ms2_df[ms2_df['name'].str.contains('_')].reset_index(drop=True)
        conjugate_df['name_ls'] = conjugate_df['name'].apply(lambda x: x.split('_'))
        # any part matches any name in all_names
        _mask = conjugate_df['name_ls'].apply(lambda x: any(name in x for name in all_names))
        conjugate_df = conjugate_df[_mask].reset_index(drop=True)
        # drop the 'name_ls' column
        conjugate_df.drop(columns=['name_ls'], inplace=True)
        
        # concatenate drug_df and conjugate_df
        matched_df = pd.concat([drug_df, conjugate_df], ignore_index=True)
        # sort by scan
        matched_df.sort_values(by='scan', inplace=True)

        return matched_df

    # get the conjugates from the MS2 library
    out_ms2_list = []
    #########################
    # large library
    matched_df1 = get_conjugates_from_ms2_df('data_cleaning/cleaned_data/ms2_all_df.pkl', inchikey, all_names)
    matched_df1['library'] = 'large'
    matched_df1.to_csv(os.path.join(out_path, 'ms2_df1.tsv'), sep='\t', index=False)
    
    # filter spectra
    reserved_scans_1 = matched_df1['scan'].unique().tolist()
    all_ms2_list_1 = read_mgf('data_cleaning/cleaned_data/ms2_all.mgf')
    print(f"Loaded {len(all_ms2_list_1):,} spectra from MS2Lib MGF file.")

    for spec in tqdm(all_ms2_list_1, desc="Filtering spectra"):
        if int(spec['SCANS']) in reserved_scans_1:
            out_ms2_list.append(spec)
    print(f"Filtered to {len(out_ms2_list):,} spectra.")

    #########################
    # drug library
    matched_df2 = get_conjugates_from_ms2_df('drug_data/cleaned_data/ms2_all_df.pkl', inchikey, all_names)
    matched_df2['library'] = 'drug'
    matched_df2.to_csv(os.path.join(out_path, 'ms2_df2.tsv'), sep='\t', index=False)

    # filter spectra
    reserved_scans_2 = matched_df2['scan'].unique().tolist()
    all_ms2_list_2 = read_mgf('drug_data/cleaned_data/ms2_all.mgf')
    print(f"Loaded {len(all_ms2_list_2):,} spectra from MS2Lib MGF file.")

    for spec in tqdm(all_ms2_list_2, desc="Filtering spectra"):
        if int(spec['SCANS']) in reserved_scans_2:
            out_ms2_list.append(spec)
    print(f"Filtered to {len(out_ms2_list):,} spectra.")

    #########################
    # write the filtered spectra to a new MGF file
    # first, re-number the scans in the output list
    for i, spec in enumerate(out_ms2_list):
        spec['SCANS'] = i + 1

    out_mgf_path = os.path.join(out_path, f'{drug_name}_conjugates.mgf')
    write_mgf(out_ms2_list, out_mgf_path)
    print(f"Filtered spectra written to {out_mgf_path}")
    
    # write the combined DataFrame to a TSV file
    combined_df = pd.concat([matched_df1, matched_df2], ignore_index=True)
    # new scan: from 1 to n, where n is the number of rows
    combined_df['new_scan'] = range(1, len(combined_df) + 1)
    combined_df.to_csv(os.path.join(out_path, f'{drug_name}_combined_ms2_df.tsv'), sep='\t', index=False)


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


if __name__ == "__main__":
    
    out_path = 'target_drugs/5ASA_conjugates/filter/data'
    get_drug_and_conjugates(out_path, drug_name='5ASA',
                            inchikey='KBOPZPXVLCULAV-UHFFFAOYSA-N')
