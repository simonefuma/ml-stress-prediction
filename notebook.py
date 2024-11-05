# +
import copy
import importlib
import itertools as it

import matplotlib.pyplot as plt
import ml
import numpy as np
import pandas as pd
from config import RANDOM_STATE
import visualize

import sklearn.metrics as metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

importlib.reload(ml)
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

visualize.show_correlation_matrix(df_2, 'Matrice di Correlazione (df_2)')
visualize.show_correlation_matrix(df_3, 'Matrice di Correlazione (df_3)')

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

visualize.show_scatter_plot(X_2d, y_2, y_unique_text_2, ['b', 'm'], 'PCA - Scatter Plot (df_2)')
visualize.show_scatter_plot(X_2d, y_3, y_unique_text_3, ['b', 'm', 'g'], 'PCA - Scatter Plot (df_3)')

visualize.show_cluster_plot(2, X_2d, y_2, y_unique_text_2, ['b', 'm'], 'Cluster Plot (df_2)')
visualize.show_cluster_plot(3, X_2d, y_3, y_unique_text_3, ['b', 'm', 'g'], 'Cluster Plot (df_3)')

visualize.show_cluster_table(2, X, y_2, y_unique_text_2, 'df_2')
visualize.show_cluster_table(3, X, y_3, y_unique_text_3, 'df_3')

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
    },
    {
        'model': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'param_grid':
        {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 1, 2, 3, 4, 5, 10, 15, 20],
            'min_samples_split': [0.01, 0.05, 0.1, 0.2, 0.3],
            'min_samples_leaf': [1, 0.05, 0.1, 0.2, 0.3],
        }
    },
    {
        'model': RandomForestClassifier(bootstrap=False, random_state=RANDOM_STATE),
        'param_grid':
        {
            'n_estimators': [3, 5, 7],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [None, 1, 2, 3, 4, 5, 10, 15, 20],
            'min_samples_split': [0.01, 0.05, 0.1, 0.2, 0.3],
            'min_samples_leaf': [1, 0.05, 0.1, 0.2, 0.3],
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

learned_models = ml.learn_models(X.values, 
                                 y_2.values, 
                                 models, 
                                 test_scorers=test_scorers, 
                                 index_test_scorer=0, 
                                 minimize_test_scorer=False, 
                                 replace=False)
for learned_model in learned_models:
    visualize.display_table(learned_model)
# -





