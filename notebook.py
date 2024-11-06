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

for module in [ml, visualize]:
    importlib.reload(module)
# -

df = pd.read_csv('data.csv', index_col='ID animals', dtype={'sucrose intake': 'float64', 'NOR index': 'float64'})
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

visualize.show_correlation_matrix(df_2, 'Matrice di Correlazione (df_2)')
visualize.show_correlation_matrix(df_3, 'Matrice di Correlazione (df_3)')

# +
# # %OP = OP/(OP+CL)*100
# t%OP = tOP/(tOP+tCL+tCENT)*100
# tCENT = (300-tOP-tCL)
try:
    df = df.drop(columns=['tOP', 'tCL', 'tCENT'])
except:
    pass
    
X = df.drop(columns=['target'])
df_2 = copy.copy(df)
df_2['target'] = pd.factorize(df['target'].str.split(' - ').str[0])[0]
y_unique_text_2 = df['target'].str.split(' - ').str[0].unique()
y_2 = df_2['target']

df_3 = copy.copy(df)
df_3['target'] = pd.factorize(df['target'])[0]
y_3 = df_3['target']
y_unique_text_3 = df['target'].unique()

# +
pca = PCA(n_components=len(X.columns))
pca.fit(StandardScaler().fit_transform(X))

explained_variance = np.cumsum(pca.explained_variance_ratio_)
n_min_components = np.argmax(explained_variance > 0.9)+1

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
X_2d = pca_2d.fit_transform(StandardScaler().fit_transform(X))
pd.DataFrame(pca_2d.components_, columns=X.columns, index=[f'Cmp {i+1}' for i in range(pca_2d.n_components_)])

visualize.show_scatter_plot(X_2d, y_2, y_unique_text_2, ['b', 'm'], 'PCA - Scatter Plot (df_2)')
visualize.show_scatter_plot(X_2d, y_3, y_unique_text_3, ['b', 'm', 'g'], 'PCA - Scatter Plot (df_3)')

# K-Means
visualize.show_cluster_plot(2, X_2d, y_2, y_unique_text_2, ['b', 'm'], 'Cluster Plot (df_2)')
visualize.show_cluster_plot(3, X_2d, y_3, y_unique_text_3, ['b', 'm', 'g'], 'Cluster Plot (df_3)')

# +
# SVC target 2
models = [
    {
        'model': SVC(kernel='linear'),
        'name': 'SVC_linear_X2_T2',
        'param_grid':
        {
            'C': [0.01, 0.1, 1, 10, 100],
        }
    },
    {
        'model': SVC(kernel='poly'),
        'name': 'SVC_poly_X2_T2',
        'param_grid':
        {
            'C': [0.01, 0.1, 1, 10, 100],
            'degree': [2, 3, 4, 5],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
    },
    {
        'model': SVC(kernel='rbf'),
        'name': 'SVC_rbf_X2_T2',
        'param_grid':
        {
            'C': [0.01, 0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
    }
]

learned_models = ml.learn_models(X_2d, 
                                 y_2.values, 
                                 models, 
                                 StratifiedKFold(n_splits=8, shuffle=True, random_state=RANDOM_STATE),
                                 StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
                                 test_scorers=ml.get_binary_scorers(), 
                                 index_test_scorer=0, 
                                 minimize_test_scorer=False, 
                                 replace=False)

for learned_model in learned_models:
    visualize.display_table(learned_model)
    visualize.show_svc_decision_boundary(X_2d, y_2, y_unique_text_2, learned_model['model'], ['b', 'm'], 
                                         'SVC Decision Boundary ' + learned_model['model_name'])

# +
# SVC target 3
models = [
    {
        'model': SVC(kernel='linear'),
        'name': 'SVC_linear_X2_T3',
        'param_grid':
        {
            'C': [0.01, 0.1, 1, 10, 100],
        }
    },
    {
        'model': SVC(kernel='poly'),
        'name': 'SVC_poly_X2_T3',
        'param_grid':
        {
            'C': [0.01, 0.1, 1, 10, 100],
            'degree': [2, 3, 4, 5],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
    },
    {
        'model': SVC(kernel='rbf'),
        'name': 'SVC_rbf_X2_T3',
        'param_grid':
        {
            'C': [0.01, 0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
    }
]

learned_models = ml.learn_models(X_2d, 
                                 y_3.values, 
                                 models, 
                                 StratifiedKFold(n_splits=8, shuffle=True, random_state=RANDOM_STATE),
                                 StratifiedKFold(n_splits=7, shuffle=True, random_state=RANDOM_STATE),
                                 test_scorers=ml.get_multiclass_scorers(), 
                                 index_test_scorer=0, 
                                 minimize_test_scorer=False, 
                                 replace=True)

for learned_model in learned_models:
    visualize.display_table(learned_model)
    visualize.show_svc_decision_boundary(X_2d, y_3, y_unique_text_3, learned_model['model'], ['b', 'm', 'g'], 
                                         'SVC Decision Boundary ' + learned_model['model_name'])
# -

# K-Means
visualize.show_cluster_table(2, X, y_2, y_unique_text_2, 'df_2')
visualize.show_cluster_table(3, X, y_3, y_unique_text_3, 'df_3')

# +
models = [
    {
        'model': KNeighborsClassifier(),
        'name': 'KNeighborsClassifier_X_T2',
        'param_grid': 
        {
            'n_neighbors': [1, 3, 5, 7, 9, 11],
            'metric': ['cityblock', 'cosine', 'euclidean', 'l2', 'l1', 'manhattan', 'nan_euclidean']
        }
    },
    {
        'model': SVC(),
        'name': 'SVC_X_T2',
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
        'name': 'DecisionTreeClassifier_X_T2',
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
        'name': 'RandomForestClassifier_X_T2',
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

learned_models = ml.learn_models(X.values, 
                                 y_2.values, 
                                 models,
                                 StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE),
                                 StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
                                 test_scorers=ml.get_binary_scorers(), 
                                 index_test_scorer=0, 
                                 minimize_test_scorer=False, 
                                 replace=False)

for learned_model in learned_models:
    visualize.display_table(learned_model)
# -





