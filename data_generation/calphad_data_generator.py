"""Script to run Thermo‑Calc calculations over a composition pool.

Creates batch files for a given composition dataset and runs equilibrium
calculations in parallel producing HDF outputs per batch hash.
"""

import os
import hashlib
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
sys.path.insert(1, '../shared')
from run_tc import run_single_equilibrium_batch

###############################################################################
# CONFIGURATION
###############################################################################
config = {
    # Path to the composition dataset (concentration values in range 0...100)
    'fn_composition_dataset': 'compositions_b.h5',
    # Temperatures in which the solution with each composition is computed
    "tc_temperatures": [1600,1500,1400,1300,1200,1100,1000,900,800,700],
    # The phase fractions to extract from the calculation result
    "tc_phases": ['LIQUID',
               'BCC_B2#1',
               'BCC_B2#2',
               'BCC_B2#3',
               'BCC_B2#4',
               'OTHER_PHASES']
}


def hash_for_chunk(compositions_chunk: np.ndarray) -> str:
    """
    Create a deterministic hash based on the compositions AND relevant config values.
    This prevents collisions when the same compositions are run under different settings.
    """
    h = hashlib.md5()
    h.update(np.ascontiguousarray(compositions_chunk).view(np.uint8))
    return h.hexdigest()


def main():
    """Run batched Thermo‑Calc calculations for the configured composition dataset.

    Submits batches to a process pool using `run_single_equilibrium_batch`
    and concatenates the resulting HDF files into a single dataset.
    """

    batch_size = 250

    compositions = pd.read_hdf(config['fn_composition_dataset'], key='df')
    config['element_labels'] = compositions.columns.tolist()
    
    compositions = compositions.values.astype(np.float32)

    # Determine number of workers based on number of CPU cores available
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))

    # Create chunks
    chunks = [compositions[i:i + batch_size] for i in range(0, len(compositions), batch_size)]

    # Run simulations asynchronously
    # Prepare worker function
    worker = partial(run_single_equilibrium_batch, config=config)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {}
        for idx, chunk in enumerate(chunks):
            chunk_hash = hash_for_chunk(chunk)
            out_path = f"data_chunk_{chunk_hash}.h5"
            if os.path.exists(out_path):
                continue
            fut = executor.submit(worker, chunk, config['element_labels'], out_path)
            future_to_idx[fut] = idx
        num_submitted = len(future_to_idx)
        if num_submitted == 0:
            pass
        else: 
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    # results are not used here
                    result = future.result()
                except Exception as e:
                    print(f"Chunk {idx} generated an exception: {e}")
                    pass

    # After all simulations complete, load results from files
    results = []
    for chunk in chunks:
        chunk_hash = hash_for_chunk(chunk)
        out_path = f"data_chunk_{chunk_hash}.h5"
        if os.path.exists(out_path):
            try:
                df = pd.read_hdf(out_path, key="df")
                if not df.empty:
                    results.append(df)
            except Exception as e:
                pass

    # Concatenate the results and save combined file
    # Combine all DataFrames into one
    df = pd.concat(results, ignore_index=True)
    # Drop completely duplicate rows
    df = df.drop_duplicates()
    df.to_hdf('dataset_b.h5', key='df', mode='w')
    print('Done. Combined results saved to: dataset_b.h5')

if __name__=="__main__":
    main()