"""Utility helpers used by the screening pipeline.

Small helpers for config loading, hashing and results file creation.
"""

import json
import hashlib
import os
import numpy as np
import pandas as pd
from pathlib import Path


def file_exists_in_folder(filepath):
    """Return True if `filepath` exists and is a regular file."""
    return os.path.exists(filepath) and os.path.isfile(filepath)


def get_hash(data):
    """Create an md5 hash for dictionaries or numpy arrays.

    Used to produce reproducible identifiers for pools and prediction
    caches.
    """
    if type(data) is dict:
        # Convert dictionary to a JSON string
        json_string = json.dumps(data, sort_keys=True).encode('utf-8')
        return hashlib.md5(json_string).hexdigest()
    else:
        try:
            return hashlib.md5(data.astype("uint8")).hexdigest()
        except:
            return hashlib.md5(np.ascontiguousarray(data.astype("uint8"))).hexdigest()
    

def get_pool_and_criteria_hash(path_str, criteria):
    """Return a reproducible short hash and pool name for a candidate pool + criteria."""
    # Normalize path using pathlib
    path = Path(path_str).resolve()  # Handles OS-specific separators
    parts = path.parts
    pool_name = parts[1].split('.')[0] if len(parts) > 1 else None
    
    # Prepare combined data
    hash_input = {
        "pool": pool_name,
        "criteria": criteria  # Keep the list of dicts as-is
    }
    # Serialize with sorted keys for consistency
    serialized = json.dumps(hash_input, sort_keys=True)
    path_hash = hashlib.md5(serialized.encode('utf-8')).hexdigest()
    return path_hash, pool_name


def SetupConfig(fn_config):
    """Load screening config JSON and enrich it with derived fields.

    Adds `script_dir`, `path_res`, `inputs`, `outputs`, element properties
    and determines which temperatures will be evaluated based on
    `config['criteria']`.

    Returns:
        (config, time_log) tuple where `time_log` is a dict read from a
        previous run if present.
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(script_dir, flush=True)

    fn_config = os.path.join(script_dir, fn_config)
    with open(fn_config, 'r') as file:
        config = json.load(file)

    config['script_dir'] = script_dir + "//"
    # Load model config
    with open(config['script_dir']+config['model_folder']+'config.json', 'r') as file:
        model_config = json.load(file)
        
    # Define results path
    config['path_res'] = os.path.join(script_dir, config["results_folder"])
    
    os.makedirs(config['path_res'], exist_ok=True)    
    print('Results will be saved to:', config['path_res'], flush=True)

    config['inputs'] = model_config['inputs']
    config['element_labels'] = model_config['inputs'][1:]
    config['outputs'] = model_config['outputs']
    config['yield_strength_temperatures'] = [0, 298, 1100]  # Kelvin
    config['output_clip01_idx'] = model_config['output_clip01_idx']

    if model_config["n_models_in_ensemble"] > 1:
        config['model'] = 'keras_ensemble_nn'
    else:
        config['model'] = 'keras_nn'

    # Check which temperatures needs to be checked based on the criteria
    config['T_criteria'] = []
    config['tc']['tc_temperatures'] = []
    for criterion in config['criteria']:
        if criterion['T'] not in config['T_criteria']:
            config['T_criteria'].append(criterion['T'])
            config['tc']['tc_temperatures'].append(criterion['T'])
            
    return config


def CreateResultsFiles(config, eval_sim):
    """Create final Excel result files for feasible compositions.

    Combines input compositions, selected phase fractions and per-phase
    compositions and writes `feasible_compositions.xlsx` under
    `config['path_res']`.
    """

    T = config["phase_composition_evaluation_temperature"]
    a2_comp_var = [f'{el}_in_A2' for el in config['element_labels']]
    ab_comp_var = [f'{el}_in_B2' for el in config['element_labels']]
    df_phases = config['phases']

    df = eval_sim['x_feasible']
    df_bcc_a2 = eval_sim['y_feasible'][T][a2_comp_var]
    df_bcc_b2 = eval_sim['y_feasible'][T][ab_comp_var]
    df_phases = eval_sim['y_feasible'][T][df_phases]

    # Combine with BCCA and BCCB DataFrames
    df_combined = pd.concat([
        df.reset_index(drop=True),
        df_phases.reset_index(drop=True),
        df_bcc_a2.reset_index(drop=True),
        df_bcc_b2.reset_index(drop=True)
        ], axis=1)

    # Write to Excel
    with pd.ExcelWriter(
        f"{config['path_res']}//feasible_compositions.xlsx",
        engine="xlsxwriter") as writer:
        df_combined.to_excel(writer, sheet_name='Viable Compositions', index=False)