"""
Author: Mikko Tahkola mikko.tahkola@vtt.fi, VTT 2025
"""

import os
from pathlib import Path
import time
import hashlib
import math
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from run_tc import run_single_equilibrium_batch, print_to_log

###############################################################################
# CONFIGURATION
###############################################################################
config = {
    # Log name
    'log_name': 'compositions_dataset_b',
    # Path to the composition dataset (concentration values in range 0...100)
    'fn_composition_dataset': 'compositions_dataset_b.h5',
    # Path where the calculation data will be stored (and searched from)
    "fn_save_data": 'data_compositions_dataset_b',
    # How many compositions are included in each batch
    # Each batch is saved with unique hash, and loaded if its already exist.
    "batch_size": 1,
    # Whether to reset the calculation state after each composition
    "tc_reset_state_after_each_composition": True,
    # Interval to free memory by invalidating results, set to <= 0 to disable
    "invalidate_results_interval": 0,
    # The ThermoCalc database to use
    "tc_database": "TCHEA7",
    # Enable / disable global optimization
    "tc_global_optimization": True,
    # Temperatures in which the solution with each composition is computed
    "tc_temperatures": [1600,1500,1400,1300,1200,1100,1000,900,800,700],
    # The phase fractions to extract from the calculation result
    "tc_phases": ['LIQUID',
               'BCC_B2#1',
               'BCC_B2#2',
               'BCC_B2#3',
               'OTHER_PHASES'],
    # Maximum number of iterations Thermo-Calc uses in the calculation
    "tc_max_number_of_iter": 5000,
    # Whether to use "tc_max_number_of_iter" (False) or default 500 first, and on failure retry with "tc_max_number_of_iter" (True)
    "tc_force_higher_max_iter": False
}

###############################################################################


def format_dd_hh_mm_ss(seconds: float) -> str:
    """
    Format seconds as DD-HH:MM:SS.
    """
    if math.isinf(seconds) or seconds < 0:
        return "--:--:--:--"
    # Round to nearest second for stable display
    total = int(round(seconds))
    days, rem = divmod(total, 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{days:02d}-{hours:02d}:{minutes:02d}:{secs:02d}"


def load_composition_data(fn):
    """
    Loads the composition dataset and returns the names of the elements
    """
    data = pd.read_hdf(fn, key='df')
    elements = data.columns
    return data.values.astype(np.float32), elements.tolist()


def hash_for_chunk(compositions_chunk: np.ndarray) -> str:
    """
    Create a deterministic hash based on the compositions AND relevant config values.
    This prevents collisions when the same compositions are run under different settings.
    """
    h = hashlib.md5()
    h.update(np.ascontiguousarray(compositions_chunk).view(np.uint8))
    return h.hexdigest()


###############################################################################


def main():
    
    manager = multiprocessing.Manager()
    log_file_main = f"progress_{config['log_name']}_main.log"
    log_file = f"progress_{config['log_name']}.log"
    lock = manager.Lock()

    # Load composition data
    compositions, elements = load_composition_data(config['fn_composition_dataset'])
    compositions = compositions[:8,:]
    
    config['element_labels'] = elements
    
    # Create results folder
    output_dir = Path(config["fn_save_data"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine number of workers based on number of CPU cores available
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
    
    # Calculate number of batches
    batch_size = config['batch_size']
    
    # Create chunks based on batch size
    chunks = [compositions[i:i + batch_size] for i in range(0, len(compositions), batch_size)]
    n_chunks = len(chunks)
    print_to_log(f'Batch size: {batch_size}', log_file_main)
    print_to_log(f'Number of compositions: {compositions.shape[0]}', log_file_main)
    print_to_log(f'-> Number of batches: {n_chunks}', log_file_main)
    
    start_time = time.time()
    
    # Run simulations asynchronously
    # Prepare worker function
    worker = partial(run_single_equilibrium_batch, config=config, log_file=log_file, lock=lock)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        
        completed = 0
        n_compositions_done = 0
        future_to_idx = {}
        # Submit only missing batches
        for idx, chunk in enumerate(chunks):
            chunk_hash = hash_for_chunk(chunk)
            out_path = output_dir / f"data_{chunk_hash}.h5"
            if os.path.exists(out_path):
                completed += 1
                n_compositions_done += batch_size
                continue
            # Submit for execution only if missing
            fut = executor.submit(worker, chunk, config['element_labels'], out_path)
            future_to_idx[fut] = idx

        num_submitted = len(future_to_idx)
        
        if num_submitted == 0:
            print_to_log("No batches submitted (all outputs already exist).", log_file_main)
        else: 
            percent = (completed / n_chunks) * 100
            print_to_log(f"{completed}/{n_chunks} batches already completed ({percent:.1f}%).", log_file_main)
            print_to_log("Starting simulations for the remaining batches", log_file_main)
            completed = 0
            # Print progress as futures finish
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as e:
                    print_to_log(f"Error in batch {idx+1}: {e}", log_file_main)
                
                # Update progress
                completed += 1
                n_compositions_done += batch_size
                
                elapsed_time = time.time() - start_time
                percent = (completed / num_submitted) * 100
                
                # Calculate ETA based on average time per batch so far
                avg_time_per_batch = elapsed_time / completed
                remaining_batches = num_submitted - completed
                eta = avg_time_per_batch * remaining_batches
                
                elapsed_str = format_dd_hh_mm_ss(elapsed_time)
                eta_str = format_dd_hh_mm_ss(eta)
    
                print_to_log(f"{completed}/{num_submitted} batches completed ({percent:.1f}%, {n_compositions_done}/{compositions.shape[0]} compositions) | "
                      f"Elapsed: {elapsed_str} | ETA: {eta_str}", log_file_main)


    # After all simulations complete, load results from files
    results = []
    print_to_log("All simulations finished. Loading results from disk...", log_file_main)
    for chunk in chunks:
        chunk_hash = hash_for_chunk(chunk)
        out_path = output_dir / f"data_{chunk_hash}.h5"
        if os.path.exists(out_path):
            try:
                df = pd.read_hdf(out_path, key="df")
                if not df.empty:
                    results.append(df)
            except Exception as e:
                print_to_log(f"Failed to load {out_path}: {e}", log_file_main)
    
    print_to_log(f"Loaded {len(results)} result DataFrames.", log_file_main)

    # Concatenate the results and save combined file
    # Combine all DataFrames into one
    df = pd.concat(results, ignore_index=True)
    
    print_to_log(f"Combined shape: {df.shape} (before removing duplicates)", log_file_main)
    
    # Drop completely duplicate rows
    df = df.drop_duplicates()
    
    print_to_log(f"Combined shape: {df.shape} (after duplicate removal)", log_file_main)

    savename = 'data_'+config['fn_composition_dataset'].split('//')[-1].split('.')[0]+'.h5'
    fn = os.path.join(config['fn_save_data'], savename)
    df.to_hdf(fn, key='df', mode='w')

    print_to_log(f'CALPHAD data saved in: {fn}', log_file_main)
    
if __name__=="__main__":
    main()