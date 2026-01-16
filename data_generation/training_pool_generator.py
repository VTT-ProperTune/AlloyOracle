"""Generate and filter composition pools for surrogate training.

Contains DoE generation, stratified bin sampling and helper filters used to
produce composition datasets consumed by the CALPHAD data generator.
"""

import numpy as np
import pandas as pd
import multiprocessing
from collections import defaultdict


def get_viable_compositions(base, r, element_id, cumusum, n_elements, min_elements, max_elements):
    """
    Recursively generates viable alloy compositions based on element concentration constraints.

    Parameters:
        base (list): Current partial composition vector.
        r (dict): Dictionary mapping element names to their allowed concentration values.
        element_id (int): Index of the current element being processed.
        cumsum (float): Current sum of concentrations in the composition.
        n_elements (int): Total number of elements in the composition.
        min_elements (int): Minimum number of non-zero elements allowed in a composition.
        max_elements (int): Maximum number of non-zero elements allowed in a composition.

    Returns:
        list: A list of valid composition vectors (lists of floats).
    """
    element = list(r.keys())[element_id]
    element_id += 1
    compositions = []
    for amount in r[element]:
        new_base = base + [amount]
        cumusum = sum(new_base)
        # Count non-zeros so far
        count_non_zeros = np.count_nonzero(new_base)
        # Don't continue if sum of concentrations is > 100% or max-elements constraints
        if cumusum >= 100.1 or count_non_zeros > max_elements:
            break  # assumes r[element] is sorted ascending; use 'continue' if not guaranteed
        # Add to dataset if sum ~ 100 and non-zero count within [min_elements, max_elements]
        if (99.9 < cumusum < 100.1) and (min_elements <= count_non_zeros <= max_elements):
            padded = new_base + [0] * (n_elements - len(new_base)) # add zeros to end
            compositions.append(padded)
        # Otherwise, continue recursion if there are elements left and non-zeros can be added
        elif (element_id < n_elements) and (count_non_zeros <= max_elements):
            compositions.extend(
                get_viable_compositions(new_base, r, element_id, cumusum,
                                      n_elements, min_elements, max_elements))
    return compositions


def bin_stratified_per_group(df, elements, K=20, bin_width=5, seed=123):
    """
    For each presence-pattern group:
      - For each such element, form bins [0, w), [w, 2w), ... up to that element’s max in the group.
      - Randomly sample up to K compositions from each (element, bin) without replacement.

    Returns:
      df_selected, df_remaining, selection_log  (and optionally indices)
        - df_selected: rows selected by the per-(element, bin) quotas
        - df_remaining: all rows NOT selected
        - selection_log: one row per (group pattern, element, bin) with counts
    """
    rng = np.random.default_rng(seed)
    X = df[elements].to_numpy()

    # Group compositions by element presence pattern
    X = df[elements].to_numpy()
    presence = (X > 1e-4).astype(np.uint8)
    patterns = [tuple(row.tolist()) for row in presence]
    pattern_to_indices = defaultdict(list)
    for idx, pat in enumerate(patterns):
        pattern_to_indices[pat].append(idx)

    selected_global = set()
    selection_rows = []

    for pattern, idxs in pattern_to_indices.items():
        idxs = np.array(idxs, dtype=int)
        if len(idxs) == 0:
            continue
        # Elements present in this group
        elements_in_group = [j for j, v in enumerate(pattern) if v == 1]
        # Shuffle element order to reduce systematic preference
        rng.shuffle(elements_in_group)
        # Group compositions Xg
        Xg = X[idxs, :]

        log = []
        selected_in_group = set()

        for j in elements_in_group:
            elem_name = elements[j]
            element_concentrations = Xg[:, j]
            max_val = float(element_concentrations.max())
            if max_val <= 0:
                continue

            # Define bin edges from 0 to max_val + 1e-12 (inclusive last bin)
            _max = max_val + 1e-12
            if max_val < bin_width:
                edges = np.array([0.0, _max])
            else:
                n_bins = int(np.ceil(max_val / bin_width))
                edges = np.linspace(0.0, n_bins * bin_width, n_bins + 1)
                edges[-1] = _max # ensure the last edge includes the max

            # For each bin, find candidates and sample up to K
            for b in range(len(edges) - 1):
                low, high = edges[b], edges[b+1]
                mask = (element_concentrations >= low) & (element_concentrations < high)
                cand_local = np.where(mask)[0]
                if len(cand_local) < 1:
                    continue

                # Map local indices back to global row indices
                cand_global = idxs[cand_local]

                # Exclude already selected (global or in this group pass)
                if selected_in_group or selected_global:
                    cand_global = np.array(
                        [g for g in cand_global
                         if (g not in selected_in_group and g not in selected_global)],
                        dtype=int
                    )

                if len(cand_global) == 0:
                    continue

                # If the number of compositions in the bin is < K, take all
                # otherwise take K compositions
                take = min(K, len(cand_global))
                chosen = rng.choice(cand_global, size=take, replace=False)

                # Save selections
                for g in chosen:
                    selected_global.add(int(g))
                    selected_in_group.add(int(g))

                log.append({
                    "pattern": ''.join(map(str, pattern)),
                    "group_size": len(idxs),
                    "element": elem_name,
                    "bin_low": low,
                    "bin_high": high,
                    "candidates": len(cand_local),
                    "chosen": int(take)
                })

        # Save group log lines
        selection_rows.extend(log)

    # Build selected and remaining
    selected_idx = np.array(sorted(selected_global), dtype=int)
    all_idx = np.arange(len(df), dtype=int)
    if selected_idx.size == 0:
        remaining_idx = all_idx
    else:
        mask = np.ones(len(df), dtype=bool)
        mask[selected_idx] = False
        remaining_idx = all_idx[mask]

    df_selected = df.iloc[selected_idx].reset_index(drop=True)
    df_remaining = df.iloc[remaining_idx].reset_index(drop=True)
    selection_log = pd.DataFrame(selection_rows)
    return df_selected, df_remaining, selection_log
    

def remove_duplicates(df_new, df_existing, element_cols, decimals=6):
    """Return rows in `df_new` that are not present in `df_existing`.

    Comparison is done by rounding element columns to `decimals` and
    performing a left-join subtraction.
    """
    new_key = df_new[element_cols].round(decimals).copy()
    old_key = df_existing[element_cols].round(decimals).copy()
    new_key['_idx'] = df_new.index
    merged = new_key.merge(old_key.drop_duplicates(), on=element_cols, how='left', indicator=True)
    keep_idx = merged.loc[merged['_merge'] == 'left_only', '_idx']
    return df_new.loc[keep_idx].reset_index(drop=True)

        
def filter_invalid_compositions(df, min_elements, max_elements):
    """Filter compositions by number of non-zero elements and sum-to-100.

    Returns only rows where the count of non-zero entries is within
    `[min_elements, max_elements]` and the numeric columns sum to 100.
    """
    nonzero_counts = (np.abs(df.values) > 1e-6).sum(axis=1)
    df = df[(nonzero_counts >= min_elements) & (nonzero_counts <= max_elements)].reset_index(drop=True)

    num_sum = df.select_dtypes(include="number").sum(axis=1)
    mask = np.isclose(num_sum, 100, rtol=1e-6, atol=1e-6)
    return df[mask]


def generate_doe(concentration_values, num_processes, elements, n_elements, min_elements, max_elements):
    """Build a design-of-experiments grid using recursive composition generation.

    Uses a multiprocessing pool to expand the compositions starting from
    different `Cr` base values.
    """
    bases = [[amount] for amount in concentration_values['Cr']]
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            get_viable_compositions,
            [(base, concentration_values, 1, -1.0, n_elements, min_elements, max_elements) for base in bases]
        )
    compositions = [row for chunk in results for row in chunk]
    compositions = np.array(compositions, dtype=np.float32)
    df = pd.DataFrame(compositions, columns=elements)
    df = filter_invalid_compositions(df, min_elements, max_elements)
    return df


def main():
    """Generate and save composition pools used to produce Dataset B.

    Produces two grids (5% and 2.5%), removes duplicates and saves the
    combined composition set to `compositions_b.h5`.
    """
    ############################################
    
    # Configuration
    num_processes = 12 # number of cores to use in the DoE generation
    random_seed = 123 # use fixed random seed for reproducibility
    # Elements to be included in the composition space
    elements = ["Cr","Al","Fe","Ni","Ti","V","Mo","W","Mn","Si","Co"]
    min_elements = 2 # minimum number of elements in an alloy
    max_elements = 11 # maximum number of elements in an alloy
    n_elements = len(elements)
    
    ############################################
    # Generate the 5% grid
    step = 5 # step size used in the grid sampling (e.g. 2.5 -> 2.5% composition intervals)
    cr_range = np.arange(50, 90+0.01, step=step).tolist()
    element_range = np.arange(0, 50+0.01, step=step).tolist()
    concentration_values = {}
    for e in elements:
        concentration_values[e] = element_range
    concentration_values['Cr'] = cr_range
    df_5pct = generate_doe(concentration_values, num_processes, elements, n_elements, min_elements, max_elements)
    print('Number of compositions in the 5% grid:', df_5pct.shape[0])
    
    ############################################
    # Generate the 2.5% grid
    step = 2.5 # step size used in the grid sampling (e.g. 2.5 -> 2.5% composition intervals)
    cr_range = np.arange(50, 90+0.01, step=step).tolist()
    element_range = np.arange(0, 50+0.01, step=step).tolist()
    concentration_values = {}
    for e in elements:
        concentration_values[e] = element_range
    concentration_values['Cr'] = cr_range
    df_2pct = generate_doe(concentration_values, num_processes, elements, n_elements, min_elements, max_elements)
    print('Number of compositions in the 2.5% grid:', df_2pct.shape[0])
    
    ############################################
    # Filter out the 5% grid compositions from the 2.5% set
    n0 = df_2pct.shape[0]
    df_2pct = remove_duplicates(df_2pct, df_5pct, elements, decimals=6)
    n_diff = n0 - df_2pct.shape[0]
    print('Removed', n_diff, 'compositions from the 2.5% set as they are included in the 5% set')
    
    ############################################
    # Get the pseudo-randomly selected 2.5% grid samples for model development
    df_2pct_dev, df_2pct_remaining, sel_log = bin_stratified_per_group(
        df=df_2pct,
        elements=elements,
        K=17,                # how many samples to select per (group, element, bin)
        bin_width=5,         # 5% concentration bins
        seed=random_seed)
    
    ############################################
    # Get the pseudo-randomly selected 2.5% grid samples for model testing
    df_test, df_2pct_remaining, sel_log = bin_stratified_per_group(
        df=df_2pct_remaining,
        elements=elements,
        K=4,                 # how many samples to select per (group, element, bin)
        bin_width=5,         # 5% concentration bins
        seed=random_seed)
    
    ############################################
    # Now the 5% grid compositions and those selected from the 2.5% can be combined
    # df_2pct_dev was not used in the final dataset B, but its generated here
    # for reproducibility because those samples are removed from the 2.5% grid
    # samples before selecting the datapoints for the test set)
    # df_dev_combined = pd.concat([df_5pct, df_2pct_dev])
    
    # Instead, the 5% grid and samples selected to "df_test" were combined
    df_combined = pd.concat([df_5pct.reset_index(drop=True), df_test.reset_index(drop=True)], axis=0)
    
    # Check for duplicates (none should be found)
    n0 = df_combined.shape[0]
    df_combined = df_combined.drop_duplicates()
    n_diff = n0 - df_combined.shape[0]
    print('Removed', n_diff, 'compositions from the resulting set as duplicates')
    
    ############################################
    # Save the generated dataset
    df_combined.to_hdf('compositions_b.h5', key='df')
    
    # Next, this composition dataset can be given as input to calphad_data_generator.py
    # to generate the CALPHAD data using Thermo-Calc

if __name__=="__main__":
    main()