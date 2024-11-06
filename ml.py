import copy
import itertools as it
import json
import joblib
import logging
import numpy as np
import os
import pickle
import subprocess
from pathlib import Path

from config import RANDOM_STATE
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def get_binary_scorers():
    specificity_scorer = lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, pos_label=0)
    specificity_scorer.__name__ = 'specificity_scorer'

    return [
        metrics.accuracy_score, 
        metrics.f1_score, 
        metrics.precision_score, 
        metrics.recall_score, 
        specificity_scorer
    ]


def get_multiclass_scorers():
    specificity_score = lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, average='macro', labels=[0, 1, 2])
    specificity_score.__name__ = 'specificity_score'

    multiclass_scorers = [metrics.accuracy_score]

    for score in [metrics.f1_score, metrics.precision_score, metrics.recall_score]:
            multiclass_score = lambda y_true, y_pred, score=score : score(y_true, y_pred, average='macro', zero_division=0)
            multiclass_score.__name__ = score.__name__
            multiclass_scorers.append(multiclass_score)
    multiclass_scorers.append(specificity_score)
    return multiclass_scorers
    

def np_jsonify(data):
    """Recursively replaces np.float64 instances with float in a nested dictionary."""
    data = copy.deepcopy(data)
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = np_jsonify(value)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = np_jsonify(data[i])
    elif isinstance(data, np.float64):
        return float(data)
    return data


def make_hp_configurations(grid):
    return [{n: v for n, v in zip(grid. keys(), t)} for t in it.product(*grid.values())]


def fit_estimator(X, y, estimator, hp_conf):
    estimator.set_params(**hp_conf)
    estimator.fit(X, y)


def get_score(X_test, y_test, estimator, scorer): 
    return scorer(y_test, estimator.predict(X_test))


def check_best(minimize, score, best_score):
    return (minimize and score < best_score) or (not minimize and score > best_score)


def inner_learn(X_trainval, y_trainval, estimator, hp_conf, inner_split_method, val_scorer=metrics.root_mean_squared_error):
    conf_scores = []
            
    for train_index, val_index in inner_split_method.split(X_trainval, y_trainval):
        X_train, X_val = X_trainval[train_index], X_trainval[val_index]
        y_train, y_val = y_trainval[train_index], y_trainval[val_index]

        fit_estimator(X_train, y_train, estimator, hp_conf)
        conf_scores.append(get_score(X_val, y_val, estimator, val_scorer))
            
    return np.mean(conf_scores), hp_conf


def learn_parallel(X, y, estimator, param_grid, outer_split_method, inner_split_method, 
                   val_scorer=metrics.root_mean_squared_error, minimize_val_scorer=True, 
                   test_scorers=[metrics.root_mean_squared_error], minimize_test_scorer=True, index_test_scorer=0, 
                   n_jobs=-1):
    outer_scores = []

    best_score = np.inf if minimize_test_scorer else -np.inf
    best_conf = None

    for trainval_index, test_index in outer_split_method.split(X, y):
        X_trainval, X_test = X[trainval_index], X[test_index]
        y_trainval, y_test = y[trainval_index], y[test_index]

        inner_results = Parallel(n_jobs=n_jobs)(delayed(inner_learn)
                                            (
                                                X_trainval, 
                                                y_trainval,
                                                copy.deepcopy(estimator),
                                                hp_conf, 
                                                inner_split_method=inner_split_method,
                                                val_scorer=val_scorer
                                            )
                                       for hp_conf in make_hp_configurations(param_grid))
        
        best_inner_conf =  min(inner_results, key=lambda inner_result: inner_result[0] if minimize_val_scorer else -inner_result[0])[1]
        fit_estimator(X_trainval, y_trainval, estimator, best_inner_conf)
        outer_scores.append([get_score(X_test, y_test, estimator, test_scorer) for test_scorer in test_scorers])
        if check_best(minimize_test_scorer, outer_scores[-1][index_test_scorer], best_score):
            best_score, best_conf = outer_scores[-1][index_test_scorer], best_inner_conf

    avg = np.mean(outer_scores, axis=0)
    std = np.std(outer_scores, axis=0, ddof=1)
    fit_estimator(X, y, estimator, best_conf)
    return estimator, [{'scorer_name':test_scorer.__name__, 'avg':avg[i], 'std':std[i]} for i, test_scorer in enumerate(test_scorers)]


def learn_models(X, y, models, outer_split_method, inner_split_method,
                 test_scorers=[metrics.accuracy_score], 
                 index_test_scorer=0, 
                 minimize_test_scorer=False, 
                 replace=False):
    EXPERIMENTS_PATH = Path('experiments')
    subprocess.run(['mkdir', '-p', str(EXPERIMENTS_PATH)])

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s > %(message)s')
    
    results = []
    for model in models:
        result = None
        trained_model = None
        
        file_name = EXPERIMENTS_PATH / model['name']
        log_file = file_name.with_suffix('.log')
        model_file = file_name.with_suffix('.pickle')

        if replace:
            try:
                os.remove(log_file)
                os.remove(model_file)
            except FileNotFoundError:
                pass

        if os.path.exists(log_file) and os.path.exists(model_file):
            with open(log_file, 'r') as f:
                result = f.read()
            result = json.loads(result[result.index('>')+2:].strip())

            with open(model_file, 'rb') as f:
                trained_model = pickle.load(f)
        else:
            logger.handlers.clear()
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            pipe = Pipeline(steps=[
                ('scaler', None),
                ('classifier', model['model'])
            ])
    
            param_grid = ({'scaler': [StandardScaler(), MinMaxScaler()]} |
                         {'classifier__'+key: value for key, value in model['param_grid'].items()})

            trained_model, result = learn_parallel(X, y, pipe, param_grid,
                                                   outer_split_method,
                                                   inner_split_method,
                                                   val_scorer=metrics.accuracy_score,
                                                   minimize_val_scorer=False,
                                                   test_scorers=test_scorers,
                                                   minimize_test_scorer=minimize_test_scorer,
                                                   index_test_scorer=index_test_scorer,
                                                   n_jobs=-1)
            logger.info(json.dumps(np_jsonify(result)))
            with open(model_file, 'wb') as f:
                pickle.dump(trained_model, f)

        results.append({'model_name': model['name'], 'model': trained_model, 'result': result})
            
            
    return results