import logging
import pandas as pd
import os
import time
import requests
import json
from enum import Enum, auto


HOST = "https://api.fasst.gnps2.org"  # new API


class DataBase(Enum):
    metabolomicspanrepo_index_nightly = auto()  # all gnps data
    gnpsdata_index = auto()  # all gnps data
    gnpsdata_index_11_25_23 = auto()  # all gnps data
    gnpslibrary = auto()  # gnps library
    massivedata_index = auto()
    massivekb_index = auto()


def split_usi(usi: str):
    """
    Split a repo USI into dataset, file, scan
    """
    parts = usi.split(":")
    if len(parts) < 5 or parts[0] != "mzspec":
        raise ValueError("Invalid USI format")
    return {
        "dataset": parts[1],
        "file": parts[2],
        "scan": parts[4]
    }


def fast_masst_spectrum(
    mzs,
    intensities,
    precursor_mz,
    precursor_charge=1,
    precursor_mz_tol=0.05,
    mz_tol=0.05,
    min_cos=0.7,
    analog=False,
    analog_mass_below=130,
    analog_mass_above=200,
    database=DataBase.metabolomicspanrepo_index_nightly,
    min_signals=3,
):
    """

    :param mzs:
    :param intensities:
    :param precursor_mz:
    :param precursor_charge:
    :param precursor_mz_tol:
    :param mz_tol:
    :param min_cos:
    :param analog:
    :param analog_mass_below:
    :param analog_mass_above:
    :param database:
    :return: (MASST results as json, filtered data points as array of array [[x,y],[...]]
    """
    # relative intensity and precision
    dps = [[mz, intensity] for mz, intensity in zip(mzs, intensities)]
    spec_dict = {
        "n_peaks": len(dps),
        "peaks": dps,
        "precursor_mz": precursor_mz,
        "precursor_charge": abs(precursor_charge),
    }

    try:
        result = fast_masst_spectrum_dict(
            spec_dict,
            precursor_mz_tol,
            mz_tol,
            min_cos,
            analog,
            analog_mass_below,
            analog_mass_above,
            database,
            min_signals,
        )
        
        # Check if API call failed (result is None)
        if result is None:
            return None
            
        # Check if API returned valid structure but no results
        if 'results' not in result:
            return None  # API call failed - invalid response structure
            
        # Successful API call - check if there are results
        if len(result['results']) == 0:
            # Successful API call but no matches found
            return pd.DataFrame(columns=['delta_mass', 'cosine', 'matching_peaks', 'dataset', 'file', 'scan'])
        
        # Successful API call with matches
        result_df = pd.DataFrame(result['results'])[['Delta Mass', 'USI', 'Cosine', 'Matching Peaks']]
        
        # clean up output
        result_df[['dataset', 'file', 'scan']] = result_df['USI'].apply(lambda x: pd.Series(split_usi(x)))
        result_df = result_df.drop(columns=['USI'])
        result_df.columns = ['delta_mass', 'cosine', 'matching_peaks', 'dataset', 'file', 'scan']
        
        return result_df
        
    except Exception as e:
        # API call failed or other error
        logging.error(f"MASST API call failed: {e}")
        return None
    

def fast_masst_spectrum_dict(
    spec_dict: dict,
    precursor_mz_tol=0.05,
    mz_tol=0.05,
    min_cos=0.7,
    analog=False,
    analog_mass_below=130,
    analog_mass_above=200,
    database=DataBase.metabolomicspanrepo_index_nightly,
    min_signals=3,
):
    """

    :param spec_dict: dictionary with GNPS json like spectrum. Example:
    {"n_peaks":8,"peaks":[[80.97339630126953,955969.8125],[81.98159790039062,542119.1875],[98.98410034179688,
    483893632.0],[116.99469757080078,1605324.25],[127.0155029296875,182958080.0],[131.01019287109375,878951.375],
    [155.0467071533203,73527152.0],[183.07809448242188,16294011.0]],"precursor_charge":1,"precursor_mz":183.078,
    "splash":"splash10-0002-9500000000-fe91737b5df956c7e69e"}
    :param precursor_mz_tol:
    :param mz_tol:
    :param min_cos:
    :param analog:
    :param analog_mass_below:
    :param analog_mass_above:
    :param database:
    :return: MASST results as json
    """
    
    max_intensity = max([v[1] for v in spec_dict["peaks"]])
    dps = [
        [round(dp[0], 5), round(dp[1] / max_intensity * 100.0, 1)]
        for dp in spec_dict["peaks"]
    ]
    dps = [dp for dp in dps if dp[1] >= 0.1]

    spec_dict["peaks"] = dps
    spec_dict["n_peaks"] = len(dps)
    if spec_dict["n_peaks"] < min_signals:
        return None

    spec_json = json.dumps(spec_dict)

    # trying to get database name, check if string or enum
    if isinstance(database, DataBase):
        database = database.name

    params = {
        "library": str(database),
        "analog": "Yes" if analog else "No",
        "delta_mass_below": analog_mass_below,
        "delta_mass_above": analog_mass_above,
        "pm_tolerance": precursor_mz_tol,
        "fragment_tolerance": mz_tol,
        "cosine_threshold": min_cos,
        "query_spectrum": spec_json,
    }
    return _fast_masst(params)


def _fast_masst(params, host: str = HOST, blocking: bool = True, timeout: int = 5):
    """
    :param params: dict of the query input and parameters
    :param host: base URL for the MASST API endpoint
    :param blocking: whether to wait for results or return immediately with task_id
    :param timeout: request timeout in seconds
    :return: dict with the MASST results. [results] contains the individual matches, [grouped_by_dataset] contains
             all datasets and their titles
    """
    query_url = os.path.join(host, "search")

    r = requests.post(query_url, json=params, timeout=timeout)
    logging.debug("fastMASST response={}".format(r.status_code))
    r.raise_for_status()

    task_id = r.json()["id"]
    params["task_id"] = task_id

    if not blocking:
        params["status"] = "PENDING"
        return params

    return blocking_for_results(params, host=host)


def blocking_for_results(query_parameters_dictionary, host: str = HOST):
    task_id = query_parameters_dictionary["task_id"]

    retries_max = 120
    current_retries = 0
    while True:
        logging.debug(f"WAITING FOR RESULTS, retries {current_retries}, taskid: {task_id}")

        r = requests.get(os.path.join(host, f"search/result/{task_id}"), timeout=30)
        r.raise_for_status()
        payload = r.json()

        # still running?
        if isinstance(payload, dict) and payload.get("status") == "PENDING":
            time.sleep(1)
            current_retries += 1
            if current_retries >= retries_max:
                logging.exception("Timeout waiting for results from FASST API")
                raise TimeoutError("Timeout waiting for results from FASST API")
            continue

        return payload
    
    
if __name__ == "__main__":

    # example usage
    result_df = fast_masst_spectrum(
        [88.0391, 42.0338, 74.0236, 137.0142, 74.0058, 120.0114],
        [100, 47, 12, 10, 8, 7],
        178.0530,
        precursor_charge=1,
        precursor_mz_tol=0.02,
        mz_tol=0.02,
        min_cos=0.7,
        analog=False,
        analog_mass_below=130,
        analog_mass_above=200,
        database=DataBase.metabolomicspanrepo_index_nightly,
        min_signals=4,
    )
    
    print(result_df)
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    result_df.to_csv("masst_results.tsv", sep="\t", index=False)
    