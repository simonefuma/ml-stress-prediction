# +
import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
# -

df = pd.read_csv('data.csv', index_col='ID animals', dtype={'sucrose intake': 'float64', 'NOR index': 'float64'})
# per il momento lavoro sul caso binario 'no stress' ed 'CMS (stress cronico)'
df['target'] = pd.factorize(df['target'].str.split(" - ").str[0])[0]
df

# Ottengo le coppie (X, y)
X = df.drop(columns=['target'])
y = df['target']

# Ottengo le righe che hanno campi vuoti
X[X.isnull().any(axis=1)]

# Ottengo le coppie di righe uguali
[(i, j) for i, j in list(it.combinations(X.index, 2)) if X.loc[i].equals(X.loc[j])]

# Ottengo le colonne che hanno campi vuoti
X.loc[:, X.isnull().any()]

# Ottengo le coppie di colonne uguali
[(i, j) for i, j in list(it.combinations(X.columns, 2)) if X[i].equals(X[j])]

# Ottengo le colonne costanti
X.columns[X.nunique() == 1]

# +
# Boxplot per ogni attributo del dataset
plt.figure(figsize=(12, 6))
plt.boxplot(X.values, tick_labels=X.columns, vert=False)

plt.title('Boxplot per ogni attributo')
plt.show()

# +
# Matrice di correlazione con heatmap
correlation_matrix = df.corr()

plt.figure(figsize=(9, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    square=True,
    cbar_kws={"shrink": .8},
    linewidths=0.5,
    linecolor='black',
    xticklabels=correlation_matrix.columns,
    yticklabels=correlation_matrix.index
)

plt.title('Matrice di Correlazione', fontsize=18)
plt.xticks(rotation=45, ha='right')
plt.show()


# -

def make_hp_configurations(grid):
    return [{n: v for n, v in zip(grid. keys(), t)} for t in it.product(*grid.values())]


def fit_estimator(X, y, estimator, hp_conf):
    estimator.set_params(**hp_conf)
    estimator.fit(X, y)


def get_score(X_test, y_test, estimator, scorer): 
    return scorer(y_test, estimator.predict(X_test))


def check_best(minimize, score, best_score):
    return (minimize and score < best_score) or (not minimize and score > best_score)


def learn(X, y, estimator, param_grid, outer_split_method, inner_split_method,
            val_scorer=metrics.root_mean_squared_error, minimize_val_scorer=True, 
            test_scorers=[metrics.root_mean_squared_error]):
    outer_scores = []
    best_conf = None

    for trainval_index, test_index in outer_split_method.split(X, y):
        X_trainval, X_test = X[trainval_index], X[test_index]
        y_trainval, y_test = y[trainval_index], y[test_index]

        best_inner_score = np.inf if minimize_val_scorer else -np.inf
        
        for hp_conf in make_hp_configurations(param_grid):
            conf_scores = []
            
            for train_index, val_index in inner_split_method.split(X_trainval, y_trainval):
                X_train, X_val = X_trainval[train_index], X_trainval[val_index]
                y_train, y_val = y_trainval[train_index], y_trainval[val_index]

                fit_estimator(X_train, y_train, estimator, hp_conf)
                conf_scores.append(get_score(X_val, y_val, estimator, val_scorer))
            
            conf_score = np.mean(conf_scores)
            if check_best(minimize_val_scorer, conf_score, best_inner_score):
                best_inner_score, best_conf = conf_score, hp_conf
                
        fit_estimator(X_trainval, y_trainval, estimator, best_conf)
        outer_scores.append([get_score(X_test, y_test, estimator, test_scorer) for test_scorer in test_scorers])

    avg = np.mean(outer_scores, axis=0)
    std = np.std(outer_scores, axis=0, ddof=1)
    fit_estimator(X, y, estimator, best_conf)
    return estimator, [{'scorer name':test_scorer.__name__, 'avg':avg[i], 'std':std[i]} for i, test_scorer in enumerate(test_scorers)]


def learn_models(X, y, models, test_scorers):
    results = []
    
    for model in models:
        pipe = Pipeline(steps=[
            ('scaler', _),
            ('classifier', model['model'])
        ])

        param_grid = ({'scaler': [StandardScaler(), MinMaxScaler()]} |
                     {'classifier__'+key: value for key, value in model['param_grid'].items()})

        results.append({
            'model': model['model'].__class__.__name__,
            'result': learn(X, y, pipe, param_grid,
                            StratifiedKFold(n_splits=4, shuffle=True, random_state=42),
                            StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                            val_scorer=metrics.accuracy_score,
                            minimize_val_scorer=False,
                            test_scorers=test_scorers
                           )        
        })

    return results


# +
models = [
    {
        'model': KNeighborsClassifier(),
        'param_grid': 
        {
            'n_neighbors': [k for k in range(1, 12, 2)],
            'metric': ['cityblock', 'cosine', 'euclidean', 'l2', 'l1', 'manhattan', 'nan_euclidean']
        }
    },
    {
        'model': SVC(),
        'param_grid':
        {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4, 5],
            'gamma': ['scale', 'auto', 0.001, 0.01, .1, 1]
        }
    }
]


specificity_scorer = lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, pos_label=0)
specificity_scorer.__name__ = "specificity_scorer"

test_scorers = [
    metrics.accuracy_score, 
    metrics.f1_score, 
    metrics.precision_score, 
    metrics.recall_score, 
    specificity_scorer
]
learn_models(X.values, y.values, models, test_scorers)
# -


