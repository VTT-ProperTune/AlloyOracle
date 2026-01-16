"""Entry point for screening pipeline.

Loads configuration, runs surrogate preselection and Thermo‑Calc validation
and writes result files to the configured results folder.
"""

import os
import multiprocessing
import filtering
import utilities as aux
import candidate_evaluation

def main():

    fn_config = "config_screening.json"

    print('\n'*2)
    print(fn_config)
    print('\n'*2)
    
    # Load screening config
    print('Setting up screening configuration','\n'+'_'*48, flush=True)
    config = aux.SetupConfig(fn_config)
    if config['multiprocessing_ncpu'] is None:
        num_processes = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
        config['multiprocessing_ncpu'] = num_processes

    print("config['preselection_method']:", config['preselection_method'])
    print("config['n_compositions_validate_max']:", config['n_compositions_validate_max'])

    # Surrogate evaluation
    eval_pred, config = candidate_evaluation.SurrogateEvaluation(config, results_folder=config['results_folder'])

    if config['preselection_method'] is None or config['n_compositions_validate_max'] is None:
        print('Evaluating all candidates found with the surrogate', flush=True)
    else:
        if eval_pred['x_feasible'].shape[0] > config['n_compositions_validate_max']:
            print('Number of potentially feasible candidates exceeds the set value for n_compositions_validate_max in the given configuration file', flush=True)
            print(f"Limiting the number of candidates to {config['n_compositions_validate_max']}", flush=True)
            
            if config['preselection_method'] == 'kmeans':
                eval_pred = filtering.LimitNumberOfCompositions(eval_pred, config)
            else:
                raise Exception(f"config['preselection_method'] was set to {config['preselection_method']} which is not a valid method.")
    
    # ThermoCalc validation
    print('ThermoCalc evaluation','\n'+'_'*48, flush=True)
    eval_sim = candidate_evaluation.ThermoCalcEvaluation(config, eval_pred)

    # Create results files
    print('Creating results files','\n'+'_'*48, flush=True)
    # config['n_clusters_in_results'] = [2] 
    aux.CreateResultsFiles(config, eval_sim)
    
    
if __name__ == "__main__":
    main()