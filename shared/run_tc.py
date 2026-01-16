"""Thermo‑Calc integration helpers using the `tc_python` API.

Provides a batched runner (`run_single_equilibrium_batch`) that is
safe to call from multiple processes and writes HDF outputs.
"""

import numpy as np
import pandas as pd
import time
from tc_python import *


def get_tc_data_columns(config, elements):
    """
    Gets the output columns for the TC data
    """
    columns = ['T_input']
    columns += elements
    phase_composition_labels = []
    volume_fraction_labels = []
    site_fraction_labels = []
    for phase in config['tc_phases']:
        if phase != 'OTHER_PHASES':
            volume_fraction_labels.append("V_f_" + phase)
            for el in elements:
                phase_composition_labels.append('X_'+phase+"_"+el)
                for i in range(1,3):
                    site_fraction_labels.append('Y_'+phase+'_'+el+'#'+str(i))
    phases = list(config.get('tc_phases', []))
    volume_fraction_labels = list(volume_fraction_labels or [])
    phase_composition_labels = list(phase_composition_labels or [])
    site_fraction_labels = list(site_fraction_labels or [])
    columns += phases + volume_fraction_labels + phase_composition_labels + site_fraction_labels
    return columns
 
def run_single_equilibrium_batch(compositions, elements, out_path, config) -> pd.DataFrame:
    """
    Runs single equilibrium calculation for a set of compositions in given
    temperatures
    """
    def safe_get(calc_result, key):
        try:
            return calc_result.get_value_of(key)
        except:
            return 0

    def get_phase_mole_fraction(calc_result, phases):
            return {f"X_{p}": safe_get(calc_result, f"NP({p})") for p in phases if p != 'OTHER_PHASES'}

    def get_phase_volume_fraction(calc_result, phases):
            return {f"V_f_{p}": safe_get(calc_result, f"VPV({p})") for p in phases if p != 'OTHER_PHASES'}

    def get_phase_composition(calc_result, phases, elements):
        out = {}
        for p in phases:
            if p == 'OTHER_PHASES': continue
            for e in elements:
                out[f"X_{p}_{e}"] = safe_get(calc_result, f"X({p},{e})")
        return out

    def get_site_fractions(calc_result, phases, elements):
        out = {}
        for p in phases:
            if p == 'OTHER_PHASES': continue
            for e in elements:
                for i in range(1, 3):
                    out[f"Y_{p}_{e}#{i}"] = safe_get(calc_result, f"Y({p},{e}#{i})")
        return out

    def attempt_calculate(calculation, temperature_K, n_max_iter, compositions_idx, n_compositions_total):
        """
        Try equilibrium calculation with default iterations, then retry with higher limit if needed.
        """
        compositions_idx += 1
        bookmark_before = calculation.bookmark_state()  # Bookmark BEFORE calculation
        try:
            return calculation.set_condition("T", temperature_K).calculate()
        except CalculationException as e1:
            calculation.set_state_to_bookmark(bookmark_before)
            try:
                calc_with_more_iters = calculation.with_options(
                    SingleEquilibriumOptions().set_max_no_of_iterations(n_max_iter)
                )
                return calc_with_more_iters.set_condition("T", temperature_K).calculate()
            except CalculationException as e2:
                calculation.set_state_to_bookmark(bookmark_before)
                raise CalculationException(
                    f"Equilibrium failed after retry at T={temperature_K} K (n_max_iter={n_max_iter})."
                ) from e2

    if compositions.ndim == 1:
        compositions = compositions.reshape(1, -1)
    x, y = [], []
    columns = get_tc_data_columns(config, elements)
    n_compositions = compositions.shape[0]
    # Progress counters & interval
    with TCPython(logging_policy=LoggingPolicy.NONE) as tc_session:

        # Build system once per process
        tc_system = (
            tc_session
            .set_cache_folder("cache")
            # .set_cache_folder(f"cache_{os.getpid()}")  # unique per process
            .select_database_and_elements("TCHEA7", elements)
            .get_system()
        )

        for i in range(compositions.shape[0]):
            try:
                # Build a calculation boject for current composition from the system
                calculation = (
                    tc_system
                    .with_single_equilibrium_calculation()
                    .set_component_to_suspended(ALL_COMPONENTS)
                    .set_component_to_entered("Cr")  # reference component
                )
                calculation.enable_global_minimization()
                # Set composition conditions
                for idx_el, el_label in enumerate(elements):
                    if el_label == "Cr":
                        continue
                    amount = compositions[i, idx_el]
                    if amount > 1e-5:
                        calculation.set_component_to_entered(el_label)
                        calculation.set_condition(f"X({el_label})", amount * 1e-2)
    
                for temperature in config['tc_temperatures']:

                    try:
                        calc_result = attempt_calculate(
                            calculation,
                            temperature + 273.15,
                            5000,
                            i,
                            n_compositions,
                        )
                        
                        mole_fractions = get_phase_mole_fraction(calc_result, config['tc_phases'])
                        volume_fractions = get_phase_volume_fraction(calc_result, config['tc_phases'])
                        phase_compositions = get_phase_composition(calc_result, config['tc_phases'], elements)
                        site_fractions = get_site_fractions(calc_result, config['tc_phases'], elements)
                        mole_fractions['OTHER_PHASES'] = 1 - sum(mole_fractions.values())
                        output_data = list(mole_fractions.values()) \
                                        + list(volume_fractions.values()) \
                                        + list(phase_compositions.values()) \
                                        + list(site_fractions.values())
    
                        input_data = [temperature] + list(compositions[i, :])
                        x.append(input_data)
                        y.append(output_data)
                        
                    except CalculationException as e:
                        # Skip this T if it fails even after retry; count failed data point
                        pass
            except CalculationException as e:
                # Skip this composition if setup fails
                pass
        
    if x and y:
        data = np.hstack((np.array(x), np.array(y)))
        df = pd.DataFrame(data=data, columns=columns)
    else:
        df = pd.DataFrame(columns=columns)
        
    # Save if not empty and return df
    if isinstance(df, pd.DataFrame) and not df.empty:
        df.to_hdf(out_path, key="df", mode="w")
    
    return df
