# +
import copy
import itertools as it
import json
import logging
import os
import pickle
import subprocess
from pathlib import Path

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
import sklearn.metrics as metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# -

df = pd.read_csv('data.csv', index_col='ID animals', dtype={'sucrose intake': 'float64', 'NOR index': 'float64'})
X = df.drop(columns=['target'])
df

# Ottengo le righe che hanno campi vuoti
df[df.isnull().any(axis=1)]

# Ottengo le coppie di righe uguali
[(i, j) for i, j in list(it.combinations(df.index, 2)) if df.loc[i].equals(df.loc[j])]

# Ottengo le colonne che hanno campi vuoti
df.loc[:, df.isnull().any()]

# Ottengo le coppie di colonne uguali
[(i, j) for i, j in list(it.combinations(df.columns, 2)) if df[i].equals(df[j])]

# Ottengo le colonne costanti
df.columns[df.nunique() == 1]

# +
# Boxplot per ogni attributo del dataset
plt.figure(figsize=(12, 6))
plt.boxplot(df.drop(columns=['target']).values, tick_labels=df.drop(columns=['target']).columns, vert=False)

plt.title('Boxplot per ogni attributo')
plt.show()


# -

def show_correlation_matrix(df, title):
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
    
    plt.title(title, fontsize=18)
    plt.xticks(rotation=45, ha='right')
    plt.show()


def show_scatter_plot(X, y, y_text, colors, title):
    plt.figure(figsize=(8, 6))
    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(colors), edgecolor='k', s=150)

    custom_lines = [plt.Line2D([0], [0], 
                               marker='o', 
                               color='w', 
                               label=y_text[i], 
                               markerfacecolor=colors[i], 
                               markersize=10) 
                    for i in range(len(y_text))]

    plt.legend(handles=custom_lines)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(handles=custom_lines)
    plt.show()


def show_cluster_plot(k, X, y, y_text, colors, title):
    kmeans = KMeans(n_clusters=k, random_state=42)
    predicts = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_;

    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    cmap = ListedColormap(colors)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.3)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    custom_lines = [plt.Line2D([0], [0], 
                               marker='o', 
                               color='w', 
                               label=y_text[i], 
                               markerfacecolor=colors[i], 
                               markersize=10) 
                    for i in range(len(y_text))]
    custom_lines.append(plt.Line2D([0], [0], 
                               marker='.', 
                               color='w', 
                               label='Centroids', 
                               markerfacecolor='k', 
                               markersize=10)
                       )
    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=150)

    plt.scatter(centroids[:, 0], centroids[:, 1], c='k', marker='.', edgecolor='k', s=150)
    plt.legend(handles=custom_lines)
    plt.title(title)
    plt.show()


def show_cluster_table(k, X, y, y_text, title):
    kmeans = KMeans(n_clusters=k, random_state=42)
    predicts = kmeans.fit_predict(X)

    table = X.drop(columns=X.columns)
    table['target'] = y_text[y]
    table['predict'] = y_text[predicts]
    
    print('\n', title)
    display(table)


# +
df_2 = copy.copy(df)
df_2['target'] = pd.factorize(df['target'].str.split(' - ').str[0])[0]
y_unique_text_2 = df['target'].str.split(' - ').str[0].unique()
y_2 = df_2['target']

df_3 = copy.copy(df)
df_3['target'] = pd.factorize(df['target'])[0]
y_3 = df_3['target']
y_unique_text_3 = df['target'].unique()
# -

show_correlation_matrix(df_2, 'Matrice di Correlazione (df_2)')
show_correlation_matrix(df_3, 'Matrice di Correlazione (df_3)')

# +
pca = PCA(n_components=13)
pca.fit(X)

explained_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.xticks(range(1, len(explained_variance) + 1))
plt.axhline(y=0.9, color='r', linestyle='--')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()
# -

pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X)

show_scatter_plot(X_2d, y_2, y_unique_text_2, ['b', 'm'], 'PCA - Scatter Plot (df_2)')
show_scatter_plot(X_2d, y_3, y_unique_text_3, ['b', 'm', 'g'], 'PCA - Scatter Plot (df_3)')

show_cluster_plot(2, X_2d, y_2, y_unique_text_2, ['b', 'm'], 'Cluster Plot (df_2)')
show_cluster_plot(3, X_2d, y_3, y_unique_text_3, ['b', 'm', 'g'], 'Cluster Plot (df_3)')

show_cluster_table(2, X, y_2, y_unique_text_2, 'df_2')
show_cluster_table(3, X, y_3, y_unique_text_3, 'df_3')


def display_table(title, data):
    print('\n' * 1)
    table = pd.DataFrame(data)
    table[['avg', 'std']] = table[['avg', 'std']].round(3)
    table.set_index('scorer_name', inplace=True)
    print(title)
    print('-' * 27)
    display(table)


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
        best_inner_conf = sorted(inner_results, key=lambda t: t[0])[0 if minimize_val_scorer else -1][1]
        
        fit_estimator(X_trainval, y_trainval, estimator, best_inner_conf)
        outer_scores.append([get_score(X_test, y_test, estimator, test_scorer) for test_scorer in test_scorers])
        if check_best(minimize_test_scorer, outer_scores[-1][index_test_scorer], best_score):
            best_score, best_conf = outer_scores[-1][index_test_scorer], best_inner_conf

    avg = np.mean(outer_scores, axis=0)
    std = np.std(outer_scores, axis=0, ddof=1)
    fit_estimator(X, y, estimator, best_conf)
    return estimator, [{'scorer_name':test_scorer.__name__, 'avg':avg[i], 'std':std[i]} for i, test_scorer in enumerate(test_scorers)]


def learn_models(X, y, models, test_scorers, minimize_test_scorer, index_test_scorer):
    EXPERIMENTS_PATH = Path('experiments')
    subprocess.run(['mkdir', '-p', str(EXPERIMENTS_PATH)])

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s > %(message)s')
    file_handler = None
    
    results = []
    for model in models:
        result = None
        trained_model = None
        
        file_name = EXPERIMENTS_PATH / (model['model'].__class__.__name__.lower())
        log_file = file_name.with_suffix('.log')
        model_file = file_name.with_suffix('.pickle')

        if os.path.exists(log_file) and os.path.exists(model_file):
            with open(log_file, 'r') as f:
                result = f.read()
            result = json.loads(result[result.index('>')+2:].strip())

            with open(model_file, 'rb') as f:
                trained_model = pickle.load(f)
        else:
            logger.removeHandler(file_handler)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            pipe = Pipeline(steps=[
                ('scaler', _),
                ('classifier', model['model'])
            ])
    
            param_grid = ({'scaler': [StandardScaler(), MinMaxScaler()]} |
                         {'classifier__'+key: value for key, value in model['param_grid'].items()})

            trained_model, result = learn_parallel(X, y, pipe, param_grid,
                                                   StratifiedKFold(n_splits=4, shuffle=True, random_state=42),
                                                   StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                                                   val_scorer=metrics.accuracy_score,
                                                   minimize_val_scorer=False,
                                                   test_scorers=test_scorers,
                                                   minimize_test_scorer=minimize_test_scorer,
                                                   index_test_scorer=index_test_scorer,
                                                   n_jobs=-1)
            logger.info(json.dumps(np_jsonify(result)))
            with open(model_file, 'wb') as f:
                pickle.dump(trained_model, f)

        results.append({'model_name': model['model'].__class__.__name__, 'model': trained_model, 'result': result})
            
            
    return results

# +
models = [
    {
        'model': KNeighborsClassifier(),
        'param_grid': 
        {
            'n_neighbors': [1, 3, 5, 7, 9, 11],
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
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
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
learned_models = learn_models(X.values, y_2.values, models, test_scorers, False, 2)
for learned_model in learned_models:
    display_table(learned_model['model_name'], learned_model['result'])
# -




