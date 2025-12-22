"""
Author: Mikko Tahkola mikko.tahkola@vtt.fi, VTT 2025
"""


import pandas as pd
import numpy as np

def add_phase_order_columns(df,elements, phases, cutoff=1e-3, exclude=('LIQUID', 'OTHER_PHASES')):
    """
    Calculates sum(abs diffs) of site fractions to determine the phase ordering.
    Adds two columns per phase:
      - f"{phase}_order" (0/1)
      - f"{phase}_sum_site_frac_abs_diff" (float)

    Parameters
    ----------
    df : DataFrame
        Input data with columns like Y_{phase}_{element}#1 and #2
    elements : list[str]
        Element symbols (e.g., ["Cr","Al","Fe",...])
    phases : list[str]
        Phase names including e.g. "BCC_B2#1", ..., possibly "LIQUID", "OTHER_PHASES"
    cutoff : float
        Threshold to declare ordering (diff > cutoff)
    exclude : tuple[str]
        Phases to skip

    Returns
    -------
    DataFrame
        DataFrame with new columns added.
    """

    df = df.copy()

    idx = df.index

    for phase in phases:
        if phase in exclude:
            continue

        diffs = []
        for el in elements:
            c1 = f"Y_{phase}_{el}#1"
            c2 = f"Y_{phase}_{el}#2"
            s1 = df.get(c1)
            if s1 is None:
                s1 = pd.Series(0.0, index=idx)
            s2 = df.get(c2)
            if s2 is None:
                s2 = pd.Series(0.0, index=idx)
            diffs.append((s1 - s2).abs().to_numpy(dtype=float))

        if not diffs:
            sum_absdiff = np.zeros(len(idx), dtype=float)
            any_gt = np.zeros(len(idx), dtype=bool)
        else:
            diffs_arr = np.vstack(diffs)
            sum_absdiff = diffs_arr.sum(axis=0)
            any_gt = (sum_absdiff > cutoff)

        df[f"{phase}_sum_site_frac_abs_diff"] = sum_absdiff
        df[f"{phase}_order"] = np.where(any_gt, 1, 0)

    return df


def normalize_rows(df, tol=1e-12):
    arr = df.to_numpy(float)
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > tol, row_sums, 1.0)  # avoid division by zero
    arr = arr / row_sums
    return pd.DataFrame(arr, columns=df.columns, index=df.index)


def compute_A2_B2_and_A2_composition(df, max_b2_index=3, tol=1e-12):
    """
    Calculates (sum of) A2 and (sum of) B2 phase fractions and phase-fraction-weighted A2 composition.
    Also calculates the number of A2 and B2 phases.
    """
    out = df.copy()

    # Identify elements from X_BCC_B2#1_<element>
    prefix = "X_BCC_B2#1_"
    elements = [c[len(prefix):] for c in out.columns if c.startswith(prefix)]
    if not elements:
        raise ValueError("No element columns of form 'X_BCC_B2#1_<element>' found.")

    # Normalize nominal composition if inputs look like percentage
    if sum(out.iloc[0][elements]) > 99:
        out[elements] /= 100.0

    # Collect all BCC_B2 phases
    phase_cols = [c for c in out.columns if c.startswith("BCC_B2#") and not c.endswith("_order") and not c.endswith("_sum_site_frac_abs_diff")]
    # Separate allowed and extra phases
    allowed_phases = []
    extra_phases = []
    for p in phase_cols:
        idx = int(p.split("#")[1])
        if idx <= max_b2_index:
            allowed_phases.append(p)
        else:
            extra_phases.append(p)
            
    # Aggregate extra phases into OTHER_PHASES
    if extra_phases:
        out["OTHER_PHASES"] += out[extra_phases].fillna(0).sum(axis=1)
        out.drop(columns=extra_phases, inplace=True)
        
    out["LIQUID"] = out["LIQUID"].clip(0, 1).fillna(0)
    for c in ["OTHER_PHASES"] + allowed_phases:
        out[c] = out[c].fillna(0)

    # Collect compositions for allowed phases
    phase_fracs = []
    phase_orders = []
    phase_comps = []
    indices = []  # numeric i for BCC_B2#i
    for p in allowed_phases:
        idx = int(p.split("#")[1])
        order_flag = out[f"{p}_order"].astype(int).to_numpy()
        frac = out[p].to_numpy(float)
        comps = np.vstack([out[f"X_{p}_{el}"].to_numpy(float) for el in elements]).T
        phase_fracs.append(frac)
        phase_orders.append(order_flag)
        phase_comps.append(comps)
        indices.append(idx)                   # numeric index

    phase_fracs = np.array(phase_fracs)
    phase_orders = np.array(phase_orders)
    phase_comps = np.array(phase_comps)
    
    
    # Count how many disordered and ordered phases have non-negligible fraction (> tol)
    num_a2 = np.sum((phase_orders == 0) & (phase_fracs > tol), axis=0)
    num_b2 = np.sum((phase_orders == 1) & (phase_fracs > tol), axis=0)
    out["multi_A2"] = (num_a2 > 1)
    out["multi_B2"] = (num_b2 > 1)
    out["n_A2_phases"] = num_a2
    out["n_B2_phases"] = num_b2
    
    # Compute A2 and B2 phase fractions
    A2_frac = np.sum(np.where(phase_orders == 0, phase_fracs, 0.0), axis=0)
    B2_frac = np.sum(np.where(phase_orders == 1, phase_fracs, 0.0), axis=0)
    out["A2"] = A2_frac
    out["B2"] = B2_frac

    # Compute global element amounts for A2 and B2
    A2_glob = np.sum(np.where(phase_orders[:, :, None] == 0, phase_fracs[:, :, None] * phase_comps, 0.0), axis=0)
    
    # Normalize compositions within A2
    X_A2 = np.full_like(A2_glob, 0)
    rows_a2 = np.where(A2_frac > tol)[0]
    if rows_a2.size:
        XA2_rows = A2_glob[rows_a2, :] / A2_frac[rows_a2, None]
        XA2_rows /= np.where(XA2_rows.sum(axis=1, keepdims=True) > tol, XA2_rows.sum(axis=1, keepdims=True), 1.0)
        X_A2[rows_a2, :] = XA2_rows
    
    # Output per-element compositions within A2
    for j, el in enumerate(elements):
        out[f"{el}_in_A2"] = X_A2[:, j]

    return out


def postprocess_calphad_data(df, element_labels, tc_phases, cutoff):
    # Calculate ordering from site fractions
    df_pp = add_phase_order_columns(df, element_labels, tc_phases, cutoff=cutoff)
    # Get a2 and b2 phase fractions, and a2 composition
    df_pp = compute_A2_B2_and_A2_composition(df_pp)
    return df_pp   
