"""Surrogate model helpers: loading, batching and prediction utilities.

This module centralises Keras model loading, pipeline transforms and a
md5-based prediction cache used by the screening pipeline.
"""

import os
import hashlib
import shutil
import time
import pickle
import numpy as np
import pandas as pd
import utilities as aux
import tensorflow as tf

def load_keras_model_and_pipelines(fn_model, fn_weights_and_pipelines, model_type):
    """Load a Keras model and its associated preprocessing pipelines.

    Supports single Keras models (`keras_nn`) and ensemble base models
    saved together with weight lists (`keras_ensemble_nn`). The
    `fn_weights_and_pipelines` pickle is expected to contain keys
    `'pipelines'`, `'model_type'` and (for ensembles) `'weights'`.
    """
    if model_type == 'keras_nn':
        model = tf.keras.models.load_model(fn_model, compile=False)
        with open(fn_weights_and_pipelines, 'rb') as pickle_file:
            weights_and_pipelines = pickle.load(pickle_file)
        pipelines = weights_and_pipelines['pipelines']
        model_type = weights_and_pipelines['model_type']
        
    elif model_type == 'keras_ensemble_nn':
        model = tf.keras.models.load_model(fn_model, compile=False)
        with open(fn_weights_and_pipelines, 'rb') as pickle_file:
            weights_and_pipelines = pickle.load(pickle_file)
        model = {'model': model,
                 'weights': weights_and_pipelines['weights']}
        pipelines = weights_and_pipelines['pipelines']
        model_type = weights_and_pipelines['model_type']
    return model, pipelines, model_type

def pipeline_transform(pipeline, arr):
    """Apply a fitted sklearn pipeline transform if present, otherwise pass through."""
    if pipeline != None:
        return pipeline.transform(arr)
    else:
        return arr


def pipeline_inverse_transform(pipeline, arr):
    """Inverse-transform predictions using the provided pipeline, or pass through."""
    if pipeline != None:
        return pipeline.inverse_transform(arr)
    else:
        return arr

def predict(m, x, config, pipelines=None, mean_only=False, mtype=None, batch_size=20000):
    """Predict using a model or ensemble and return DataFrame outputs.

    Args:
        m: model object or ensemble dict
        x: numpy array inputs (without temperature column)
        config: configuration dict with `outputs` and clipping indices
        pipelines: preprocessing pipelines
        mean_only: if True return only mean predictions for ensemble
        mtype: 'keras_nn' or 'keras_ensemble_nn'
        batch_size: keras predict batch size

    Returns:
        (y_pred_df, y_std_df) or (y_pred_df, None)
    """
    
    batch_size = int(batch_size)
    if mtype == 'keras_ensemble_nn':
        y_preds = []
        model = m['model']
        for i in range(len(m)):
            # Get the weights of the correct model
            model.set_weights(m['weights'][i])
            x_t = pipeline_transform(pipelines[i]['x'], x)
        
            if x_t.shape[0] == 0:
                raise ValueError("Input x_t has zero samples.")

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
        
        y_mean = pd.DataFrame(y_mean, columns=config['outputs'])
        y_std = pd.DataFrame(y_std, columns=config['outputs'])
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
        y_pred = pd.DataFrame(y_pred, columns=config['outputs'])
        return y_pred, None
    
    else:
        raise Exception("Implemented models: 'keras_ensemble_nn', 'keras_nn'")

def PredictBatch(x, config):
    """Predict outputs for a candidate pool and cache per-temperature files.
    """
    print('_'*33, flush=True)
    print('Predicting outputs for the pool', flush=True)

    # Evaluate pool based on predicted results
    print("Loading the surrogate model", flush=True)
    model, pipelines, mtype = load_keras_model_and_pipelines(
        fn_model=config['script_dir'] + config['model_folder'] + "model",
        fn_weights_and_pipelines=config['script_dir'] + config['model_folder'] + "weights_and_pipelines.pkl",
        model_type=config['model'])

    x = x.values
    predict_in_batches = False
    n_samples = x.shape[0]
    
    fn = {}
    _batching_limit = 20e6
    _chunk_size = 10e6

    fn_preds = 'data_preds'
    os.makedirs(fn_preds, exist_ok=True)

    print('Calculating the outputs at temperatures:', flush=True)
    for t in config['T_criteria']:
        print('T:', t, 'oC', flush=True)
    print('_'*33, flush=True)
    if n_samples > _batching_limit:
        predict_in_batches = True
        chunk_size = _chunk_size
        num_chunks = int(np.ceil(n_samples / chunk_size))

    def md5_hash(arr, temperature):
        h = hashlib.md5()
        h.update(arr.tobytes())
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(str(temperature).encode())
        return h.hexdigest()

    for T in config['T_criteria']:
        fn[T] = []
        print('Predicting for T:', T, 'oC', flush=True)
        if predict_in_batches:
            x_splits = np.array_split(x, num_chunks, axis=0)
            for i, x_i in enumerate(x_splits):
                file_id = md5_hash(x_i, T)
                filename = os.path.join(fn_preds, f"pred_{file_id}.pkl")
                x_i_with_t = np.hstack((np.full((x_i.shape[0], 1), T), x_i))

                y_i_pred, _ = predict(model, x_i_with_t, config,
                                                pipelines=pipelines, mtype=mtype)

                with open(filename, 'wb') as f:
                    pickle.dump({'y_pred': y_i_pred}, f)
                fn[T].append(filename)

        else:
            file_id = md5_hash(x, T)
            filename = os.path.join(fn_preds, f"pred_{file_id}.pkl")
            x_with_temp = np.hstack((np.full((x.shape[0], 1), T), x))
            y_pred, _ = predict(model, x_with_temp, config,
                                        pipelines=pipelines, mtype=mtype)
            with open(filename, 'wb') as f:
                pickle.dump({'y_pred': y_pred}, f)
            fn[T] = filename

    print('Reading results')
    y_pool = {}
    for T in config['T_criteria']:
        y_pool[T] = {}
        if predict_in_batches:
            for k, filename in enumerate(fn[T]):
                with open(filename, 'rb') as f:
                    d = pickle.load(f)
                for key in d:
                    if isinstance(d[key], np.ndarray):
                        arr = d[key]
                        if arr.ndim == 1:
                            arr = arr.reshape(-1, 1)
                        d[key] = pd.DataFrame(arr, columns=config['outputs'])

                if k == 0:
                    y_pool[T]['y_pred'] = d['y_pred']
                else:
                    y_pool[T]['y_pred'] = pd.concat(
                        [y_pool[T]['y_pred'], d['y_pred']], ignore_index=True)
        else:
            with open(fn[T], 'rb') as f:
                d = pickle.load(f)  
            for key in d:
                if isinstance(d[key], np.ndarray):
                    arr = d[key]
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                    d[key] = pd.DataFrame(arr, columns=config['outputs'])
            y_pool[T]['y_pred'] = d['y_pred']
    shutil.rmtree(fn_preds)
    return y_pool