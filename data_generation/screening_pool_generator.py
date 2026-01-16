"""Generate composition pools for alloy screening."""

import numpy as np
import pandas as pd
import multiprocessing


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
    """Generates grid pool with 2.5% concentration interval."""
    ############################################
    
    # Configuration
    num_processes = 12 # number of cores to use in the DoE generation
    # Elements to be included in the composition space
    elements = ["Cr","Al","Fe","Ni","Ti","V","Mo","W","Mn","Si","Co"]
    min_elements = 2 # minimum number of elements in an alloy
    max_elements = 11 # maximum number of elements in an alloy
    n_elements = len(elements)
    
    ############################################
    # Generate the 2.5% grid
    step = 2.5 # step size used in the grid sampling (e.g. 2.5 -> 2.5% composition intervals)
    cr_range = np.arange(50, 90+0.01, step=step).tolist()
    element_range = np.arange(0, 50+0.01, step=step).tolist()
    concentration_values = {}
    for e in elements:
        concentration_values[e] = element_range
    concentration_values['Cr'] = cr_range
    df = generate_doe(concentration_values, num_processes, elements, n_elements, min_elements, max_elements)
    print('Number of compositions in the 2.5% grid:', df.shape[0])
    
    # Check for duplicates (none should be found)
    n0 = df.shape[0]
    df = df.drop_duplicates()
    n_diff = n0 - df.shape[0]
    print('Removed', n_diff, 'compositions from the resulting set as duplicates')
    
    ############################################
    # Save the generated dataset
    df.to_hdf('composition_pool_screening.h5', key='df')

if __name__=="__main__":
    main()