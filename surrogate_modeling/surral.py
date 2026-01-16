"""Surrogate model training.

Contains dataset loading, pipeline definition and model building.
"""

import os
import json
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, ReLU, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.models import Model
tf.autograph.set_verbosity(0)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
##############################################################################
def load_dataset(fn, config, only_inputs=False, return_dataframe=False, load_all_outputs=False):
    """Load a pickled dataset and return inputs/outputs suitable for training.

    If `only_inputs` is True the function returns numpy input values only.
    If `return_dataframe` is True, (x_df, y_df) are returned instead of arrays.
    """
    data = pd.read_pickle(fn)
    data = data.dropna()
    if only_inputs:
        return data.values.astype(np.float32)
    x = data[config['inputs']]
    if load_all_outputs:
        y = data.drop(config['inputs'], axis='columns')
    else:
        y = data[config['outputs']]
    inputs = list(x.columns)
    input_renaming = {}
    for input_name in inputs:
        if input_name != 'T_input':
            try:
                input_renaming[input_name] = input_name.split('_')[1]
            except:
                input_renaming[input_name] = input_name
    x.rename(columns=input_renaming, inplace=True)
    x_new_label_order = ['T_input'] + config['element_labels']
    x = x[x_new_label_order]
    if load_all_outputs == False:
        y_new_label_order = config['outputs']
        y = y[y_new_label_order]
    if return_dataframe:
        return x, y
    return x.values, y.values
##############################################################################
def split_by_temperature_ratio_arrays(X, y, temperature_idx=0, ratio=0.8, random_state=42):
    """
    Splits X and y into two subsets based on the given ratio,
    preserving the distribution of temperature groups.

    Parameters:
    - X: numpy array of shape (n_samples, n_features)
    - y: numpy array of shape (n_samples,)
    - temp_idx: index of the temperature column in X
    - ratio: fraction for the first subset (e.g., 0.8 for 80%)
    - random_state: seed for reproducibility

    Returns:
    - X1, X2, y1, y2
    """
    np.random.seed(random_state)
    X1_list, X2_list, y1_list, y2_list = [], [], [], []

    # Get unique temperature values
    temps = np.unique(X[:, temperature_idx])

    for t in temps:
        # Indices for this temperature group
        group_idx = np.where(X[:, temperature_idx] == t)[0]

        # Shuffle indices
        shuffled_idx = np.random.permutation(group_idx)

        # Compute split point
        split_point = int(len(shuffled_idx) * ratio)

        # Split indices
        idx1 = shuffled_idx[:split_point]
        idx2 = shuffled_idx[split_point:]

        # Append subsets
        X1_list.append(X[idx1])
        X2_list.append(X[idx2])
        y1_list.append(y[idx1])
        y2_list.append(y[idx2])

    # Concatenate all groups
    X1 = np.vstack(X1_list)
    X2 = np.vstack(X2_list)
    y1 = np.concatenate(y1_list)
    y2 = np.concatenate(y2_list)

    return X1, X2, y1, y2
##############################################################################
class temperature_stratified_kfold:
    """K-fold splitter that preserves temperature-group membership.

    Use `split(X)` to yield `(train_idx, val_idx)` arrays where each fold
    contains a stratified share of every temperature group.
    """
    def __init__(self, n_splits=5, temp_idx=0, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.temp_idx = temp_idx
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        rng = np.random.default_rng(self.random_state)
        temps = np.unique(X[:, self.temp_idx])
        
        # Collect indices per temperature group
        temp_groups = {t: np.where(X[:, self.temp_idx] == t)[0] for t in temps}
        
        # Shuffle within each group if needed
        if self.shuffle:
            for t in temps:
                rng.shuffle(temp_groups[t])
        
        # Prepare folds
        folds = [[] for _ in range(self.n_splits)]
        for t in temps:
            indices = temp_groups[t]
            parts = np.array_split(indices, self.n_splits)
            for i in range(self.n_splits):
                folds[i].extend(parts[i])
        
        # Yield train/validation indices
        all_indices = set(range(len(X)))
        for i in range(self.n_splits):
            val_idx = np.array(folds[i])
            train_idx = np.array(list(all_indices - set(val_idx)))
            yield train_idx, val_idx
##############################################################################
def save_model_and_pipelines(model, pipelines, mtype, config):
    """Save trained Keras model(s) and associated pipelines to `config['path_res']`."""
    if mtype == 'keras_ensemble_nn':
        model['model'].save(config['path_res'] + '//model')
        weights_and_pipelines = {}
        weights_and_pipelines['weights'] = model['weights']
        weights_and_pipelines['pipelines'] = pipelines
        weights_and_pipelines['model_type'] = mtype
        with open(config['path_res'] + '//weights_and_pipelines.pkl', 'wb') as file:
            pickle.dump(weights_and_pipelines, file)
    elif mtype =='keras_nn':
        model.save(config['path_res'] + '//model')
        pipeline = {}
        pipeline['pipelines'] = pipelines
        pipeline['model_type'] = mtype
        with open(config['path_res'] + '//weights_and_pipelines.pkl', 'wb') as file:
            pickle.dump(pipeline, file)
##############################################################################
def define_pipelines(hps, x, y):
    """Create preprocessing pipelines for inputs and outputs based on hyperparams."""
    pipelines = {}
    xpipe, ypipe = [], []
    if hps['feat_input_poly']:
        xpipe.append(('poly_features', PolynomialFeatures(interaction_only=False, include_bias=False)))
    if hps['feat_input_std']:
        xpipe.append(('std_scaler', StandardScaler()))
    if hps['feat_output_std']:
        ypipe.append(('std_scaler', StandardScaler()))
    if len(xpipe) > 0:
        pipelines['x'] = Pipeline(xpipe).fit(x)
    else:
        pipelines['x'] = None
    if len(ypipe) > 0:
        pipelines['y'] = Pipeline(ypipe).fit(y)
    else:
        pipelines['y'] = None
    return pipelines
##############################################################################
def pipeline_transform(pipeline, arr):
    """Apply transform from sklearn Pipeline or return input unchanged."""
    if pipeline != None:
        return pipeline.transform(arr)
    else:
        return arr
##############################################################################
def pipeline_inverse_transform(pipeline, arr):
    """Inverse-transform using the pipeline if present, otherwise passthrough."""
    if pipeline != None:
        return pipeline.inverse_transform(arr)
    else:
        return arr
##############################################################################
def pipeline_transform_train_and_valid(pipelines, x, x_v, y, y_v):
    """Transform training and validation arrays using provided pipelines."""
    x_t = pipeline_transform(pipelines['x'], x)
    y_t = pipeline_transform(pipelines['y'], y)
    x_v = pipeline_transform(pipelines['x'], x_v)
    y_v = pipeline_transform(pipelines['y'], y_v)
    return x_t, x_v, y_t, y_v
###############################################################################
def NN(x, y, hps, config):
    """Build and compile a feed-forward Keras network based on `hps`."""
    activation = hps['activation']
    n_l = int(hps['n_l'])
    n_u = int(hps['n_u'])
    input_dim = x.shape[1]
    output_dim = y.shape[1]
    
    lr = hps['lr']
    
    # Define the input layer
    inputs = Input(shape=(input_dim,))
    
    # Define the shared hidden layers
    x = inputs
    for i in range(n_l):
        if activation == 'relu':
            x = Dense(n_u)(x)
            x = ReLU()(x)
        elif activation == 'leaky_relu':
            x = Dense(n_u)(x)
            x = LeakyReLU()(x)
        else:
            x = Dense(n_u, activation=activation)(x)
        
        if i < n_l - 1 and hps['batch_normalization']:
            # No batch normalization for the last layer before output
            x = BatchNormalization()(x)
            
        if hps['dropout'] > 0:
            x = Dropout(hps['dropout'])(x)
            
        if hps['layer_width_decrease'] and n_u >= output_dim:
            n_u = int(n_u / 2)

    output = Dense(output_dim)(x)
    
    # Create the final model
    model = Model(inputs=inputs, outputs=output)
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss=hps['loss_function'])
    
    return model
##############################################################################
def train_ensemble_bagging(x, y, hps, CALLBACKS, config, seed_multiplier=2):
    """Train an ensemble of models by stratified bagging over temperature groups."""
    model_specs = {}
    model_specs['weights'] = []
    pipelines_all = []
    kfold = temperature_stratified_kfold(n_splits=config['n_models_in_ensemble'],
                                       temp_idx=0,
                                       shuffle=True,
                                       random_state=config['random_seed']*seed_multiplier)
    i = 0
    for idx_t, idx_v in kfold.split(x):
        
        print(f'Training model {i+1}/{config["n_models_in_ensemble"]}')

        x_train_i, y_train_i = np.copy(x[idx_t,:]), np.copy(y[idx_t,:])
        x_valid_i, y_valid_i = np.copy(x[idx_v,:]), np.copy(y[idx_v,:])

        pipelines = define_pipelines(hps, x_train_i, y_train_i)
        x_train_i = pipeline_transform(pipelines['x'], x_train_i)
        y_train_i = pipeline_transform(pipelines['y'], y_train_i)
        x_valid_i = pipeline_transform(pipelines['x'], x_valid_i)
        y_valid_i = pipeline_transform(pipelines['y'], y_valid_i)
        tf.keras.backend.clear_session()
        tf.random.set_seed(config['random_seed']*seed_multiplier+i) 
        model = NN(x_train_i, y_train_i, hps, config)

        model, history = fit_model(model,
                                         x_train_i, y_train_i,
                                         x_valid_i, y_valid_i,
                                         hps,
                                         CALLBACKS)
        model_specs['weights'].append(model.get_weights())
        pipelines_all.append(pipelines)
        i += 1
    model_specs['model'] = model
    return model_specs, pipelines_all, 'keras_ensemble_nn'
##############################################################################
def fit_model(model, x, y, x_es, y_es, hps, callbacks):
    """Fit a compiled Keras model and return the fitted model and history."""
    history = model.fit(x, y,
                        validation_data=(x_es, y_es),
                        batch_size=int(hps['batch_size']),
                        epochs=int(hps['n_epochs_max']),
                        shuffle=True,
                        callbacks=callbacks, verbose=0)
    return model, history
##############################################################################
def create_model_and_pipelines(data, config, hps=None, ensemble=True, seed_multiplier=2):
    """Create and train either a single model or an ensemble and pipelines.

    Returns `(model_or_specs, pipelines, model_type)` where `model_type` is
    `'keras_nn'` or `'keras_ensemble_nn'`.
    """
    N = config['n_models_in_ensemble']
    if N <= 1:
        ensemble = False

    x_all, y_all = np.copy(data['x']), np.copy(data['y'])

    if not ensemble:
        x_train, x_es, y_train, y_es = split_by_temperature_ratio_arrays(
            x_all, y_all,
            temperature_idx=0,
            ratio=1-config["validation_split"],
            random_state=int(config['random_seed']*seed_multiplier)
        )
            
    CALLBACKS = []
    if hps['lr_plateau_reduce_factor'] > 0:
        CALLBACKS.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                              patience=10,
                                                              factor=hps['lr_plateau_reduce_factor']))
    CALLBACKS.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=int(hps['early_stopping_patience']),
                                                      restore_best_weights=True))

    if ensemble:
        return train_ensemble_bagging(x_all, y_all, hps, CALLBACKS, config, seed_multiplier=seed_multiplier)
    else:
        pipelines = define_pipelines(hps, x_train, y_train)
        x_train, x_es, y_train, y_es = pipeline_transform_train_and_valid(pipelines, x_train, x_es, y_train, y_es)
        tf.keras.backend.clear_session()
        tf.random.set_seed(config['random_seed']+seed_multiplier)
        model = NN(x_train, y_train, hps, config)
        model, history = fit_model(model, x_train, y_train, x_es, y_es, hps, CALLBACKS)

        return model, pipelines, 'keras_nn'
##############################################################################
def predict(m, x, config, pipelines=None, mean_only=False, mtype=None, batch_size=20000):
    """Wrapper to predict with a saved model or ensemble and return arrays.

    Supports the same semantics as `screening.surrogate.predict`.
    """
   
    if mtype == 'keras_ensemble_nn':
        y_preds = []
        model = m['model']
        for i in range(len(m)):
            # Get the weights of the correct model
            model.set_weights(m['weights'][i])
            x_t = pipeline_transform(pipelines[i]['x'], x)
            y_pred = model.predict(x_t, batch_size=batch_size, verbose=0)
            y_pred = pipeline_inverse_transform(pipelines[i]['y'], y_pred)
            y_preds.append(y_pred)
        y_preds = np.array(y_preds)
        y_mean = np.mean(y_preds, axis=0)
        y_std = np.std(y_preds, axis=0)
        
        # Clip predictions
        if 'output_clip01_idx' in config:
            clip01_idx = np.array(config['output_clip01_idx'], dtype=int)
            if clip01_idx.size > 0:
                y_mean[:, clip01_idx] = np.clip(y_mean[:, clip01_idx], 0.0, 1.0)
        if 'output_min0_idx' in config:
            min0_idx   = np.array(config['output_min0_idx'], dtype=int)
            if min0_idx.size > 0:
                y_mean[:, min0_idx] = np.clip(y_mean[:, min0_idx], 0.0, None)
        
        if mean_only==False:
            return y_mean, y_std
        else:
            return y_mean

    elif mtype == 'keras_nn':
        x = pipeline_transform(pipelines['x'], x)
        y_pred = m.predict(x, batch_size=batch_size, verbose=0)
        y_pred = pipeline_inverse_transform(pipelines['y'], y_pred)
        
        # Clip predictions
        if 'output_clip01_idx' in config:
            clip01_idx = np.array(config['output_clip01_idx'], dtype=int)
            if clip01_idx.size > 0:
                y_pred[:, clip01_idx] = np.clip(y_pred[:, clip01_idx], 0.0, 1.0)
        if 'output_min0_idx' in config:
            min0_idx   = np.array(config['output_min0_idx'], dtype=int)
            if min0_idx.size > 0:
                y_pred[:, min0_idx] = np.clip(y_pred[:, min0_idx], 0.0, None)
        
        return y_pred, None
    
    else:
        raise Exception("Implemented models: 'keras_ensemble_nn', 'keras_nn'")
##############################################################################
def compute_metrics(y_true, y_pred, outputs):
    """Compute RMSE/R2 and per-output metrics for prediction dicts."""
    r = {}
    for d in y_pred:
        y_i_pred = y_pred[d]['pred']
        r[f'{d}_rmse'] = root_mean_squared_error(y_true[d], y_i_pred)
        r[f'{d}_r2'] = r2_score(y_true[d], y_i_pred)
    for d in y_pred:
        y_i_pred = y_pred[d]['pred']
        for i in range(y_true[d].shape[1]):
            r[f'{d}_{outputs[i]}_rmse'] = root_mean_squared_error(y_true[d][:,i], y_i_pred[:,i])
            r[f'{d}_{outputs[i]}_r2'] = r2_score(y_true[d][:,i], y_i_pred[:,i])
    return r
##############################################################################
def save_config(config):
    """Persist a copy of `config` into a model folder.

    Returns the updated `config` with `path_res`.
    """        
    config['path_res'] = "model"

    # Create directory if it doesn't exist
    if not os.path.exists(config['path_res']):
        os.makedirs(config['path_res'])

    # Save config to JSON file
    config_path = os.path.join(config['path_res'], "config.json")
    with open(config_path, "w") as out_file:
        json.dump(config, out_file, indent=4)

    return config