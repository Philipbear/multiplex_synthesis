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


def filter_processable_spectra(spectrum_list, out_dir, mgf_identifier_field_name='SPECTRUMID', min_peaks=4):
    """
    Filter spectra to only include those that need processing and have sufficient data
    
    Returns:
        tuple: (processable_spectra, skip_stats)
    """
    processable_spectra = []
    skip_stats = {
        'already_processed': 0,
        'insufficient_data': 0,
        'no_precursor_mz': 0,
        'total_skipped': 0
    }
    
    for spec_dict in spectrum_list:
        spec_id = spec_dict.get(mgf_identifier_field_name, 'unknown_id')
        out_path = os.path.join(out_dir, f"{spec_id}.tsv")
        
        # Check if already processed
        if os.path.exists(out_path):
            skip_stats['already_processed'] += 1
            continue
        
        # Check precursor mz
        try:
            precursor_mz = float(spec_dict.get('PEPMASS', 0))
        except (ValueError, TypeError):
            precursor_mz = 0
        
        if precursor_mz == 0:
            skip_stats['no_precursor_mz'] += 1
            continue
        
        # Check number of peaks
        if len(spec_dict.get('mz_ls', [])) < min_peaks:
            skip_stats['insufficient_data'] += 1
            continue
        
        # If we get here, spectrum is processable
        processable_spectra.append(spec_dict)
    
    skip_stats['total_skipped'] = skip_stats['already_processed'] + skip_stats['insufficient_data'] + skip_stats['no_precursor_mz']
    
    return processable_spectra, skip_stats


def masst_one_spec(spec_dict, out_dir,
                   mgf_identifier_field_name='SPECTRUMID', 
                   min_cos=0.7, min_peaks=4, ms1_tol=0.05, ms2_tol=0.05, analog=False, analog_mass_below=130, analog_mass_above=200):
    """
    Perform MASST search for one spectrum dictionary
    Note: This function now assumes the spectrum has already been validated for processing
    """
    spec_id = spec_dict.get(mgf_identifier_field_name, 'unknown_id')
    out_path = os.path.join(out_dir, f"{spec_id}.tsv")
    
    mzs = spec_dict['mz_ls']
    intensities = spec_dict['intensity_ls']
    precursor_mz = float(spec_dict.get('PEPMASS', 0))

    try:
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
    
    except Exception as e:
        return {'status': 'failed', 'reason': f'exception: {str(e)}', 'spec_id': spec_id}
    

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


def get_processing_stats(batch_results, skip_stats):
    """
    Calculate processing statistics from batch results and skip stats
    """
    stats = {
        'total_input': 0,
        'already_processed': skip_stats['already_processed'],
        'skipped_no_precursor': skip_stats['no_precursor_mz'],
        'skipped_insufficient_data': skip_stats['insufficient_data'],
        'total_skipped': skip_stats['total_skipped'],
        'attempted_processing': 0,
        'api_errors': 0,
        'successful_no_matches': 0,
        'successful_with_matches': 0,
        'other_errors': 0
    }
    
    for batch_result in batch_results:
        if batch_result:
            for result in batch_result:
                stats['attempted_processing'] += 1
                
                if result['status'] == 'failed':
                    if result['reason'] == 'api_error':
                        stats['api_errors'] += 1
                    else:
                        stats['other_errors'] += 1
                elif result['status'] == 'success':
                    if result['reason'] == 'no_matches':
                        stats['successful_no_matches'] += 1
                    elif result['reason'] == 'matches_found':
                        stats['successful_with_matches'] += 1
    
    stats['total_input'] = stats['total_skipped'] + stats['attempted_processing']
    
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
    
    # Filter spectra to remove those that don't need processing
    print("Filtering spectra for processing...")
    processable_spectra, skip_stats = filter_processable_spectra(
        spectrum_list, out_dir, mgf_identifier_field_name, min_peaks
    )
    
    print(f"Spectra breakdown:")
    print(f"  Total input: {len(spectrum_list):,}")
    print(f"  Already processed: {skip_stats['already_processed']:,}")
    print(f"  No precursor m/z: {skip_stats['no_precursor_mz']:,}")
    print(f"  Insufficient peaks (< {min_peaks}): {skip_stats['insufficient_data']:,}")
    print(f"  Total skipped: {skip_stats['total_skipped']:,}")
    print(f"  Ready for processing: {len(processable_spectra):,}")
    
    if not processable_spectra:
        print("No spectra need processing. Exiting.")
        return
    
    # Split processable spectra into batches
    spectrum_batches = [processable_spectra[i:i + batch_size] 
                       for i in range(0, len(processable_spectra), batch_size)]
    
    print(f"Processing {len(spectrum_batches)} batches with {n_cores} cores...")
    
    # Create partial function with fixed parameters
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
    stats = get_processing_stats(batch_results, skip_stats)
    
    print(f"\n=== Final Processing Statistics ===")
    print(f"Total input spectra: {stats['total_input']:,}")
    print(f"Already processed (skipped): {stats['already_processed']:,}")
    print(f"No precursor m/z (skipped): {stats['skipped_no_precursor']:,}")
    print(f"Insufficient peaks (skipped): {stats['skipped_insufficient_data']:,}")
    print(f"Total skipped: {stats['total_skipped']:,}")
    print(f"Attempted processing: {stats['attempted_processing']:,}")
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