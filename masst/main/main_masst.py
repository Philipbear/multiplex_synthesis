import pandas as pd
import os
from tqdm import tqdm
from masst_utils import fast_masst_spectrum, DataBase
import multiprocessing as mp
from functools import partial


def read_mgf_to_list(library_mgf):
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

    return spectrum_list


def masst_one_spec(spec_dict, out_dir,
                   mgf_identifier_field_name='SPECTRUMID', 
                   min_cos=0.7, min_peaks=4, ms1_tol=0.05, ms2_tol=0.05, analog=False, analog_mass_below=130, analog_mass_above=200):
    """
    Perform MASST search for one spectrum dictionary
    :return: MASST results as dataframe
    """
    spec_id = spec_dict.get(mgf_identifier_field_name, 'unknown_id')
    out_path = os.path.join(out_dir, f"{spec_id}.tsv")
    
    # Check if file already exists
    if os.path.exists(out_path):
        return {'status': 'skipped', 'reason': 'already_processed', 'spec_id': spec_id}
    
    mzs = spec_dict['mz_ls']
    intensities = spec_dict['intensity_ls']
    precursor_mz = float(spec_dict.get('PEPMASS', 0))

    # skip if no precursor mz or too few peaks
    if precursor_mz == 0 or len(mzs) < min_peaks:
        return {'status': 'skipped', 'reason': 'low_quality_data', 'spec_id': spec_id}

    result = fast_masst_spectrum(
        mzs,
        intensities,
        precursor_mz,
        precursor_charge=1,
        precursor_mz_tol=ms1_tol,
        mz_tol=ms2_tol,
        min_cos=min_cos,
        analog=analog,
        analog_mass_below=analog_mass_below,
        analog_mass_above=analog_mass_above,
        database=DataBase.metabolomicspanrepo_index_nightly,
        min_signals=min_peaks
    )
    
    if result is None:  # API call failed
        return {'status': 'failed', 'reason': 'api_error', 'spec_id': spec_id}

    result = result[result['matching_peaks'] >= min_peaks].reset_index(drop=True)
    
    # Save results even if empty (to mark as processed)
    result.to_csv(out_path, sep='\t', index=False)
    
    if len(result) == 0:
        return {'status': 'success', 'reason': 'no_matches', 'spec_id': spec_id}
    else:
        return {'status': 'success', 'reason': 'matches_found', 'spec_id': spec_id, 'num_matches': len(result)}
    

def process_spectrum_batch(spectrum_batch, out_dir,
                           mgf_identifier_field_name='SPECTRUMID', min_cos=0.7, min_peaks=4, 
                           ms1_tol=0.05, ms2_tol=0.05, 
                           analog=False, analog_mass_below=130, analog_mass_above=200):
    """
    Process a batch of spectra
    """
    results = []
    for spec_dict in spectrum_batch:
        result = masst_one_spec(spec_dict, out_dir,
                                mgf_identifier_field_name=mgf_identifier_field_name, 
                                min_cos=min_cos, min_peaks=min_peaks, 
                                ms1_tol=ms1_tol, ms2_tol=ms2_tol, 
                                analog=analog, analog_mass_below=analog_mass_below, analog_mass_above=analog_mass_above)
        if result is not None:
            results.append(result)
    return results


def get_processing_stats(batch_results):
    """
    Calculate processing statistics from batch results
    """
    stats = {
        'total_processed': 0,
        'already_processed': 0,
        'skipped_insufficient_data': 0,
        'api_errors': 0,
        'successful_no_matches': 0,
        'successful_with_matches': 0,
        'other_errors': 0
    }
    
    for batch_result in batch_results:
        if batch_result:
            for result in batch_result:
                stats['total_processed'] += 1
                
                if result['status'] == 'skipped':
                    if result['reason'] == 'already_processed':
                        stats['already_processed'] += 1
                    elif result['reason'] == 'insufficient_data':
                        stats['skipped_insufficient_data'] += 1
                elif result['status'] == 'failed':
                    if result['reason'] == 'api_error':
                        stats['api_errors'] += 1
                    else:
                        stats['other_errors'] += 1
                elif result['status'] == 'success':
                    if result['reason'] == 'no_matches':
                        stats['successful_no_matches'] += 1
                    elif result['reason'] == 'matches_found':
                        stats['successful_with_matches'] += 1
    
    return stats


def main_masst_mgf(library_mgf, out_dir, n_cores=None, batch_size=10,
                   mgf_identifier_field_name='SPECTRUMID', 
                   min_cos=0.7, min_peaks=4, ms1_tol=0.05, ms2_tol=0.05, 
                   analog=False, analog_mass_below=130, analog_mass_above=200):
    """
    Parallel MASST processing of all spectra in an MGF file
    
    Args:
        library_mgf: Path to MGF file
        out_dir: Output directory for results
        n_cores: Number of cores to use (default: CPU count - 1)
        batch_size: Number of spectra per batch
    """
    if n_cores is None:
        n_cores = max(mp.cpu_count() - 1, 1)
    
    print(f"Reading MGF file: {library_mgf}")
    spectrum_list = read_mgf_to_list(library_mgf)
    print(f"Found {len(spectrum_list)} spectra")
    
    if not spectrum_list:
        print("No valid spectra found in MGF file")
        return
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Check how many files already exist
    existing_files = set()
    if os.path.exists(out_dir):
        existing_files = {f.replace('.tsv', '') for f in os.listdir(out_dir) if f.endswith('.tsv')}
    
    print(f"Found {len(existing_files)} already processed spectra")
    
    # Split spectra into batches
    spectrum_batches = [spectrum_list[i:i + batch_size] 
                       for i in range(0, len(spectrum_list), batch_size)]
    
    print(f"Processing {len(spectrum_batches)} batches with {n_cores} cores...")
    
    # Create partial function with fixed out_dir
    process_func = partial(process_spectrum_batch, out_dir=out_dir,
                           mgf_identifier_field_name=mgf_identifier_field_name, min_cos=min_cos, min_peaks=min_peaks,
                           ms1_tol=ms1_tol, ms2_tol=ms2_tol,
                           analog=analog, analog_mass_below=analog_mass_below, analog_mass_above=analog_mass_above)

    # Process batches in parallel
    with mp.Pool(n_cores) as pool:
        batch_results = list(tqdm(
            pool.imap(process_func, spectrum_batches),
            total=len(spectrum_batches),
            desc="Processing spectrum batches"
        ))
    
    # Calculate and print statistics
    stats = get_processing_stats(batch_results)
    
    print(f"\n=== Processing Statistics ===")
    print(f"Total spectra processed: {stats['total_processed']:,}")
    print(f"Already processed (skipped): {stats['already_processed']:,}")
    print(f"Skipped (insufficient data): {stats['skipped_insufficient_data']:,}")
    print(f"API errors: {stats['api_errors']:,}")
    print(f"Successful (no matches): {stats['successful_no_matches']:,}")
    print(f"Successful (with matches): {stats['successful_with_matches']:,}")
    print(f"Other errors: {stats['other_errors']:,}")
    print(f"Results saved to: {out_dir}")

   
if __name__ == "__main__":
    
    # Run parallel MASST processing
    main_masst_mgf(
        library_mgf="/home/shipei/projects/synlib/masst/main/data/ms2_all_unique_usi.mgf",
        out_dir="/home/shipei/projects/synlib/masst/main/data/masst_results",
        n_cores=5,
        batch_size=10,
        mgf_identifier_field_name='SPECTRUMID',
        min_cos=0.7,
        min_peaks=4,
        ms1_tol=0.05,
        ms2_tol=0.05,
        analog=False,
        analog_mass_below=130,
        analog_mass_above=200
    )