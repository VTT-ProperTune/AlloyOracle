"""Candidate evaluation helpers.

Implements loss computation, surrogate-based preselection and
Thermo‑Calc validation orchestration used by `screening/main.py`.
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
import pandas as pd
import hashlib
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import surrogate
import sys
sys.path.insert(1, '../shared')
from postprocess_outputs import postprocess_calphad_data
from run_tc import run_single_equilibrium_batch

def ComputeLossForCriterion(y_pred, criterion):
    """Compute per-sample loss for a single criterion.

    The function supports several constraint types defined in `criterion['constraint']`:
    - 'less': loss = max(y_pred - limit, 0)
    - 'greater': loss = max(limit - y_pred, 0)
    - 'equal': loss = abs(y_pred - limit)
    - 'between': loss = distance to nearest bound when outside [lower, upper]
    - 'sum': loss = abs(sum(values) - limit) (expects 2D input)
    - 'first_greater_than_others': loss = max(0, others - first)

    Args:
        y_pred (array-like): Predicted value(s) for the criterion. For
            multi-output checks this may be a 2D array.
        criterion (dict): Criterion specification containing at least
            'constraint' and 'limit' keys (and for some types 'T'/'output').

    Returns:
        numpy.ndarray: Loss values per sample (flattened if necessary).
    """
    y_pred = np.asarray(y_pred, dtype=float)
    constraint = criterion['constraint']

    if constraint == 'less':
        # the value must be less than the defined limit
        loss = np.maximum(y_pred - criterion['limit'], 0)

    elif constraint == 'greater':
        # the value must be greater than the defined limit
        loss = np.maximum(criterion['limit'] - y_pred, 0)

    elif constraint == 'equal':
        # the value must be equal to the defined limit
        loss = np.abs(y_pred - criterion['limit'])

    elif constraint == 'between':
        # the value must be within the two defined limit(s)
        if len(criterion['limit']) != 2:
            raise Exception("Error in ComputeLossForCriterion: The length of list 'limit' in criterion definition for constraint 'between' must be 2.")
        lower_limit, upper_limit = criterion['limit']
        loss = np.where(
            (y_pred >= lower_limit) & (y_pred <= upper_limit),
            0,
            np.minimum(np.abs(y_pred - lower_limit), np.abs(y_pred - upper_limit))
        )
    elif constraint == 'sum':
        # the sum of the values must be equal to the limit 
        _sum = np.sum(y_pred, axis=1)
        loss = np.abs(_sum - criterion['limit'])

    elif constraint == 'first_greater_than_others':
        # the first value must be greater than others
        max_diff = np.maximum(0, y_pred[:, 1:] - y_pred[:, [0]])
        loss = np.max(max_diff, axis=1)

    else:
        raise Exception(
            'Valid constraints ["less", "greater", "equal", "between", "sum", "first_greater_than_others"]')
    return loss


def get_losses(x, y, config, simulated=False):
    """Compute per-criterion losses and a combined total loss DataFrame.

    For each criterion in `config['criteria']` this function extracts the
    appropriate predicted outputs (from `y`) and computes the loss using
    `ComputeLossForCriterion`. It also prints counts for several
    thresholds to help inspect feasibility rates.

    Args:
        x (pd.DataFrame): Candidate input pool (used only for sizing/logging).
        y (dict): Mapping temperature -> predictions. When `simulated` is
            True the function expects y[T] to be a DataFrame of direct
            simulated outputs; otherwise y[T] should be a dict-like with
            `'y_pred'` (and for ensembles `'y_std'`).
        config (dict): Screening configuration containing `criteria` and
            `outputs` definitions.
        simulated (bool): If True, treat `y` as simulation outputs.

    Returns:
        pd.DataFrame: DataFrame with one column per criterion and a final
        `'Total loss'` column summarising per-sample loss.
    """

    # Convert to a DataFrame is needed
    if isinstance(y, np.ndarray):
        print('Converting output data from np.ndarrays to pd.DataFrames')
        for T in y:
            y[T] = pd.DataFrame(y[T], columns=config['outputs'])
            
    # Process all candidates at once, one criterion at a time
    losses_criteria = np.zeros((x.shape[0], len(config['criteria'])))
    loss_cols = []
    for criterion_idx, criterion in enumerate(config['criteria']):
        loss_cols.append(f'C{criterion_idx+1} loss')
        if simulated:
            y_pred = y[criterion['T']][criterion['output']]
        else:
            y_pred = y[criterion['T']]['y_pred'][criterion['output']]
        # Compute the loss for this criterion for all samples at once
        criterion_losses = ComputeLossForCriterion(y_pred, criterion)
        losses_criteria[:, criterion_idx] = criterion_losses.ravel()

    # Sum up the losses for each sample
    losses_total = np.sum(losses_criteria, axis=1)
    loss_cols += ['Total loss']
    losses_combined = np.column_stack((losses_criteria, losses_total))
    losses = pd.DataFrame(losses_combined, columns=loss_cols)

    # Print feasible candidate counts for each threshold
    thresholds = [config['loss_threshold_sim'], config['loss_threshold']]
    print("\nFeasible candidate counts:")
    for th in thresholds:
        count = np.sum(losses_total <= th)
        print(f"Threshold {th}: {count} feasible candidates ({count/len(losses_criteria[:, 0]):.6%})")

    return losses

    
def get_feasible(x, y, losses, config, simulated=False):
    """Filter candidates that meet the loss threshold and return results.

    Args:
        x (pd.DataFrame): Candidate inputs.
        y (dict): Predictions or simulations keyed by temperature.
        losses (pd.DataFrame): Losses produced by `get_losses`.
        config (dict): Screening configuration containing `loss_threshold` and
            `loss_threshold_sim`.
        simulated (bool): If True use `loss_threshold_sim` and expect `y[T]`
            to be simulated DataFrames.

    Returns:
        dict: A dictionary with keys `'x_feasible'`, `'y_feasible'`,
        `'losses_feasible'`.

    Raises:
        Exception: if no feasible candidates are found.
    """
    threshold = config['loss_threshold_sim'] if simulated else config['loss_threshold']
    
    idx_feasible = losses['Total loss'] <= threshold
    losses_feasible = losses.loc[idx_feasible]
    x_feasible = x.loc[idx_feasible]
    y_feasible = {}
    for T in y:
        y_feasible[T] = y[T].loc[idx_feasible] if simulated else y[T]['y_pred'].loc[idx_feasible]
        
    n_feasible = x_feasible.shape[0]
    if n_feasible < 1:
        raise Exception("NO CANDIDATES FOUND, EXITING")

    results = {
        'x_feasible': x_feasible,
        'y_feasible': y_feasible,
        'losses_feasible': losses_feasible,
        'n_feasible': n_feasible
    }
    
    return results


def sort_feasible(results):
    """Sort feasible candidates by ascending total loss.

    Returns the same `results` dict with `'x_feasible'`,
    `'losses_feasible'` and `'y_feasible'` re-ordered.
    """
    # Extract components
    x_feasible = results['x_feasible']
    y_feasible = results['y_feasible']
    losses_feasible = results['losses_feasible']

    # Sort by Total loss
    sorted_idx = losses_feasible['Total loss'].argsort()

    # Apply sorting
    x_sorted = x_feasible.iloc[sorted_idx]
    losses_sorted = losses_feasible.iloc[sorted_idx]

    # Sort y_feasible for each temperature
    y_sorted = {T: df.iloc[sorted_idx] for T, df in y_feasible.items()}

    # Update results
    results['x_feasible'] = x_sorted
    results['losses_feasible'] = losses_sorted
    results['y_feasible'] = y_sorted

    return results


def SurrogateEvaluation(config, results_folder=None):

    print('Surrogate evaluation', flush=True)
    # The evaluation results should contain:
    # Loss calculated for all compositions in the given pool
    print("Loading candidate pool", flush=True)
    x_pool = pd.read_hdf(config["filepath_candidate_pool"], key="df")
    print('Size of the candidate pool:', x_pool.shape[0], flush=True)
    ranges_df = x_pool.agg(['min', 'max'])
    config['concentration_ranges'] = ranges_df.T.to_dict(orient='index')
    for criterion in config['criteria']:
        y_idx = []
        for output_name in criterion['output']:
            y_idx.append(config['outputs'].index(output_name))    
        criterion['y_idx'] = y_idx
    # Save the updated config to the results folder
    with open(config['path_res']+"//config.json", "w") as f:
        json.dump(config, f, indent = 4)

    # Make predictions for the pool
    y_pool = surrogate.PredictBatch(x_pool, config)
        
    # Get losses
    losses = get_losses(x_pool, y_pool, config)
    eval_pred = get_feasible(x_pool, y_pool, losses, config)
    eval_pred = sort_feasible(eval_pred)
        
    print('_'*33, flush=True)
    print('Surrogate evaluation results:', flush=True)
    print('# feasible:', eval_pred['n_feasible'], flush=True)
    print('_'*33, flush=True)

    return eval_pred, config


def ThermoCalcEvaluation(config, eval_pred):
    """
    Performs validation of the potential candidates found with a surrogate
    model.
    """
    df = run_tc_simulations(eval_pred['x_feasible'], config)
    print('Postprocessing CALPHAD validation data', flush=True)
    x_sim, y_sim = ProcessSimulationData(df, config)

    # Evaluate simulation results against the criteria
    print('Evaluating the simulated data points', flush=True)
    losses = get_losses(x_sim, y_sim, config, simulated=True)
    eval_sim = get_feasible(x_sim, y_sim, losses, config, simulated=True)
    eval_sim = sort_feasible(eval_sim)
    
    print('_'*33, flush=True)
    print('CALPHAD evaluation results:', flush=True)
    print('# feasible:', eval_sim['n_feasible'], flush=True)
    print('_'*33, flush=True)
    
    return eval_sim


def hash_for_chunk(compositions_chunk: np.ndarray) -> str:
    """
    Create a deterministic hash based on the compositions AND relevant config values.
    This prevents collisions when the same compositions are run under different settings.
    """
    h = hashlib.md5()
    h.update(np.ascontiguousarray(compositions_chunk).view(np.uint8))
    return h.hexdigest()


def run_tc_simulations(df_compositions, config):
    """Run Thermo‑Calc simulations in parallel for given compositions.

    Splits compositions into batches and runs `run_single_equilibrium_batch`
    in parallel using a process pool. Results are concatenated and
    duplicates removed before returning a DataFrame.
    """
    compositions = df_compositions.copy().values
    
    print('Starting ThermoCalc simulations', flush=True)
    tc_config = config['tc']

    # Create folder for the simulation subsets
    output_dir = Path("data_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine number of workers based on number of CPU cores available
    num_workers = config['multiprocessing_ncpu']
    
    # Calculate number of batches
    batch_size = tc_config['tc_batch_size']
    
    # If compositions are fewer than batch_size, adjust batch_size
    num_compositions = compositions.shape[0]
    if num_compositions < batch_size:
        # Distribute compositions roughly evenly across workers
        batch_size = max(1, num_compositions // num_workers)
    
    # Create chunks based on batch size
    chunks = [compositions[i:i + batch_size] for i in range(0, len(compositions), batch_size)]
    n_chunks = len(chunks)
    print(f'Batch size: {batch_size}', flush=True)
    print(f'Number of compositions: {compositions.shape[0]}', flush=True)
    print(f'-> Number of batches: {n_chunks}', flush=True)
    
    # Run simulations asynchronously
    # Prepare worker function
    worker = partial(run_single_equilibrium_batch, config=tc_config)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {}
        for idx, chunk in enumerate(chunks):
            chunk_hash = hash_for_chunk(chunk)
            out_path = output_dir / f"data_{chunk_hash}.h5"
            fut = executor.submit(worker, chunk, config['element_labels'], out_path)
            future_to_idx[fut] = idx
        num_submitted = len(future_to_idx)
        if num_submitted == 0:
            pass
        else: 
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Error in batch {idx+1}: {e}", flush=True)

    # After all simulations complete, load results from files
    results = []
    for chunk in chunks:
        chunk_hash = hash_for_chunk(chunk)
        out_path = output_dir / f"data_{chunk_hash}.h5"
        if os.path.exists(out_path):
            df = pd.read_hdf(out_path, key="df")
            if not df.empty:
                results.append(df)
    
    shutil.rmtree(output_dir, ignore_errors=True)
    
    print(f"Loaded {len(results)} result DataFrames.", flush=True)
    df = pd.concat(results, ignore_index=True)
    print(f"Combined shape: {df.shape} (before removing duplicates)", flush=True)
    df = df.drop_duplicates()
    print(f"Combined shape: {df.shape} (after duplicate removal)", flush=True)
    
    return df


def ProcessSimulationData(df, config):
    """Postprocess raw TC DataFrame into structured (x_sim, y_sim_temp).

    - Adds ordering and A2/B2 calculations
    - Selects only compositions for which all required `T_criteria` exist

    Returns:
        x_sim: DataFrame of unique compositions (elements only)
        y_sim_temp: dict mapping temperature -> DataFrame of outputs
    """
    # Postprocess the simulation data - calculate a2 and b2 phase fractions and
    # a2 composition based on the bcc phase ordering which is calculated based
    # on the site fractions
    df_pp = postprocess_calphad_data(df, config['element_labels'], config['tc']['tc_phases'], config['tc']['bcc_site_fraction_absdiff_cutoff'])
    
    # Add postprocessed data to the dataframe
    for col in df_pp:
        if col not in df:
            df[col] = df_pp[col]
    # postprocessing can add to other phases and affect liquid through normalization
    # they must be manually overwritten as they already exist in df
    df['LIQUID'] = df_pp['LIQUID']
    df['OTHER_PHASES'] = df_pp['OTHER_PHASES']
    df['n_A2_phases'] = df_pp['n_A2_phases']
    df['multi_A2'] = df_pp['multi_A2']
    
    # Select relevant columns
    x_check = df[['T_input'] + config['element_labels']] # take input columns
    y_check = df.drop(columns=['T_input'] + config['element_labels']) # take all other columns but inputs
    y_cols = y_check.columns # save for later naming the columns in restructured dfs

    # Find all unique compositions (exclude temperature)
    unique_compositions = x_check[config['element_labels']].drop_duplicates()
    
    T_criteria = config['T_criteria']
    y_sim_temp = {T: [] for T in T_criteria}
    x_sim = []
    # Iterate over unique compositions
    for _, comp_row in unique_compositions.iterrows():
        # Filter rows matching this composition
        mask = (x_check[config['element_labels']] == comp_row.values).all(axis=1)
        subset_x = x_check.loc[mask]
        subset_y = y_check.loc[mask]
        temps = subset_x['T_input'].values
        # Check if all required temperatures are present
        if all(T in temps for T in T_criteria):
            # Append y values for each required temperature
            for T in T_criteria:
                y_val = subset_y.loc[subset_x['T_input'] == T].iloc[0]
                y_sim_temp[T].append(y_val.values)  # Keep as array for now
            x_sim.append(comp_row.values)
    
    # Convert lists to DataFrames
    for T in y_sim_temp:
        y_sim_temp[T] = pd.DataFrame(y_sim_temp[T], columns=y_cols)
    x_sim = pd.DataFrame(np.array(x_sim), columns=config['element_labels'])

    return x_sim, y_sim_temp