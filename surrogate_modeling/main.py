"""Train a surrogate using given hyperparameters
"""

import json, os
import plotting
import pandas as pd
import surral as al
import random

def main():
    
    with open("config_surrogate.json", 'r') as json_file:
        config = json.load(json_file)
    config = al.save_config(config)

    params = config['hyperparameters']
    for key in params:
        print(f'{key}: {params[key]}')

    data = pd.read_hdf(config['fn_dataset_training'], key='df')

    config['data_labels_all'] = list(data.columns)

    data_test = pd.read_hdf(config['fn_dataset_testing'], key='df')
    x_train, y_train = data[config['inputs']].values,  data[config['outputs']].values
    x_test, y_test = data_test[config['inputs']].values, data_test[config['outputs']].values
        
    data = {'train': {'x': x_train, 'y': y_train},
            'test': {'x': x_test, 'y': y_test}}

    print('Dataset sizes:')
    for key in data:
        print(key)
        for key2 in data[key]:
            print(key, data[key][key2].shape)
            
    # Save configuration file and create results folder
    config = al.save_config(config)

    print('Training the surrogate')
    model, pipelines, mtype = al.create_model_and_pipelines(data['train'], config, hps=params)
    al.save_model_and_pipelines(model, pipelines, mtype, config)

    print('Evaluating the surrogate')
    y_pred = {}
    for key in ['train','test']:
        y_pred[key] = {}
        y_pred[key]['pred'], y_pred[key]['std'] = al.predict(model, data[key]['x'], config, pipelines=pipelines, mtype=mtype)
        
    y_true = {'train': data['train']['y'], 'test': data['test']['y']}

    eval_results = al.compute_metrics(y_true, y_pred, config['outputs'])
    # Saving results
    print('Saving results')
    pd.DataFrame([eval_results]).to_excel(os.path.join(config['path_res'], 'results.xlsx'))

    plotting.save_r2_plot(y_true, y_pred, config)

if __name__ == "__main__":
    main()