# +
import copy
import importlib
import itertools as it
import matplotlib.pyplot as plt
import ml
import numpy as np
import pandas as pd
import visualize

from config import RANDOM_STATE
from sklearn.decomposition import PCA

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
min_components = np.argmax(explained_variance > 0.9)+1

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
# SVC_KERNELS_X2_T2
learned_models = ml.learn_models(X_2d, y_2.values, ml.get_svc_kernels_models('_X2_T2'), 
                                 StratifiedKFold(n_splits=8, shuffle=True, random_state=RANDOM_STATE),
                                 StratifiedKFold(n_splits=7, shuffle=True, random_state=RANDOM_STATE),
                                 test_scorers=ml.get_binary_scorers(), index_test_scorer=0, minimize_test_scorer=False, 
                                 replace=False)

visualize.display_table(learned_models)
visualize.show_svc_decision_boundary(X_2d, y_2, y_unique_text_2, learned_models, ['b', 'm'])

# +
# SVC_KERNELS_X2_T3
learned_models = ml.learn_models(X_2d, y_3.values, ml.get_svc_kernels_models('_X2_T3'), 
                                 StratifiedKFold(n_splits=8, shuffle=True, random_state=RANDOM_STATE),
                                 StratifiedKFold(n_splits=7, shuffle=True, random_state=RANDOM_STATE),
                                 test_scorers=ml.get_multiclass_scorers(), index_test_scorer=0, minimize_test_scorer=False, 
                                 replace=False)


visualize.display_table(learned_models)
visualize.show_svc_decision_boundary(X_2d, y_3, y_unique_text_3, learned_models, ['b', 'm', 'g'])
# -

# PCA con 3 componenti
pca_3d = PCA(n_components=3)
X_3d = pca_3d.fit_transform(StandardScaler().fit_transform(X))
pd.DataFrame(pca_3d.components_, columns=X.columns, index=[f'Cmp {i+1}' for i in range(pca_3d.n_components_)])

# Scatter Plot 3D dei dati trasformati da PCA con 3 componenti
visualize.show_3D_scatter_plot(X_3d, y_2, y_unique_text_2, ['b', 'm'], 'Scatter Plot (df2)')
visualize.show_3D_scatter_plot(X_3d, y_3, y_unique_text_3, ['b', 'm', 'g'], 'Scatter Plot (df3)')

# +
# SVC_KERNELS_X3_T2
learned_models = ml.learn_models(X_3d, y_2.values, ml.get_svc_kernels_models('_X3_T2'),
                                 StratifiedKFold(n_splits=8, shuffle=True, random_state=RANDOM_STATE),
                                 StratifiedKFold(n_splits=7, shuffle=True, random_state=RANDOM_STATE),
                                 test_scorers=ml.get_binary_scorers(), index_test_scorer=0, minimize_test_scorer=False, 
                                 replace=False)

visualize.display_table(learned_models)

# +
# SVC_KERNELS_X3_T3
learned_models = ml.learn_models(X_3d, y_3.values, ml.get_svc_kernels_models('_X3_T3'),
                                 StratifiedKFold(n_splits=8, shuffle=True, random_state=RANDOM_STATE),
                                 StratifiedKFold(n_splits=7, shuffle=True, random_state=RANDOM_STATE),
                                 test_scorers=ml.get_multiclass_scorers(), index_test_scorer=0, minimize_test_scorer=False, 
                                 replace=False)

visualize.display_table(learned_models)
# -

pca_min_components = PCA(n_components=min_components)
X_min_components = pca_min_components.fit_transform(StandardScaler().fit_transform(X))
pd.DataFrame(pca_min_components.components_, columns=X.columns, index=[f'Cmp {i+1}' for i in range(pca_min_components.n_components_)])

# +
# SVC_KERNELS_MIN_COMPONENTS_T2
learned_models = ml.learn_models(X_min_components, y_2.values, ml.get_svc_kernels_models('_MIN_COMPONENTS_T2'),
                                 StratifiedKFold(n_splits=8, shuffle=True, random_state=RANDOM_STATE),
                                 StratifiedKFold(n_splits=7, shuffle=True, random_state=RANDOM_STATE),
                                 test_scorers=ml.get_binary_scorers(), index_test_scorer=0, minimize_test_scorer=False, 
                                 replace=False)

visualize.display_table(learned_models)
# -

# SVC_KERNELS_MIN_COMPONENTS_T3
learned_models = ml.learn_models(X_min_components, y_3.values, ml.get_svc_kernels_models('_MIN_COMPONENTS_T3'),
                                 StratifiedKFold(n_splits=8, shuffle=True, random_state=RANDOM_STATE),
                                 StratifiedKFold(n_splits=7, shuffle=True, random_state=RANDOM_STATE),
                                 test_scorers=ml.get_multiclass_scorers(), index_test_scorer=0, minimize_test_scorer=False, 
                                 replace=False)
visualize.display_table(learned_models)

# K-Means_X_T2, K-Means_X_T3
visualize.show_cluster_table(2, StandardScaler().fit_transform(X), y_2, y_unique_text_2, 'df_2')
visualize.show_cluster_table(3, StandardScaler().fit_transform(X), y_3, y_unique_text_3, 'df_3')

# MODELS_X_T2
learned_models = ml.learn_models(X.values, y_2.values, ml.MODELS_X_T2,
                                 StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE),
                                 StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
                                 test_scorers=ml.get_binary_scorers(), index_test_scorer=0, minimize_test_scorer=False, 
                                 replace=False)
visualize.display_table(learned_models)
for learned_model in learned_models:
    if(learned_model['model'].named_steps['classifier'].__class__.__name__ == 'DecisionTreeClassifier'):
        visualize.plot_tree(X.columns, y_unique_text_2, learned_model['model'].named_steps['classifier'], learned_model['model_name'])

