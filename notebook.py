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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

for module in [ml, visualize]:
    importlib.reload(module)


# +
# TO-DO:
# topi femmina
# tutti i topi
# visualizzare i grafici svc sulle prime due componenti
# visualizzare iperparametri dei modelli allenati
# cambiare ordine delle operazioni per matrice di correlazione
# fattorizzare operazioni
# cambiare come ottenere models, per modificare iperparametri
# -

def svc_kernels(X, y, y_unique_text, colors, models, test_scorers,
                outer_split=8, inner_split=7, 
                index_test_scorer=0, minimize_test_scorer=False, 
                replace=False):
    learned_models = ml.learn_models(X, y, models,
                                 StratifiedKFold(n_splits=outer_split, shuffle=True, random_state=RANDOM_STATE),
                                 StratifiedKFold(n_splits=inner_split, shuffle=True, random_state=RANDOM_STATE),
                                 test_scorers=test_scorers, index_test_scorer=0, minimize_test_scorer=False, 
                                 replace=False)

    visualize.display_table(learned_models)
    try:
        visualize.show_svc_decision_boundary(X, y, y_unique_text, learned_models, colors)
    except:
        pass


def models(X, y, y_unique_text, models, test_scorers,
           outer_split=4, inner_split=3, 
           index_test_scorer=0, minimize_test_scorer=False, 
           replace=False):
    learned_models = ml.learn_models(X.values, y, models,
                                 StratifiedKFold(n_splits=outer_split, shuffle=True, random_state=RANDOM_STATE),
                                 StratifiedKFold(n_splits=inner_split, shuffle=True, random_state=RANDOM_STATE),
                                 test_scorers=test_scorers, index_test_scorer=index_test_scorer, minimize_test_scorer=minimize_test_scorer, 
                                 replace=False)
    visualize.display_table(learned_models)
    for learned_model in learned_models:
        if(learned_model['model'].named_steps['classifier'].__class__.__name__ == 'DecisionTreeClassifier'):
            visualize.plot_tree(X.columns, y_unique_text, learned_model['model'].named_steps['classifier'], learned_model['model_name'])


# # Mouse male

df_males = pd.read_csv('data/males.csv', index_col='ID animals', dtype={'sucrose intake': 'float64', 'NOR index': 'float64'})
df_males

# Ottengo le righe che hanno campi vuoti
df_males[df_males.isnull().any(axis=1)]

# Ottengo le coppie di righe uguali
[(i, j) for i, j in list(it.combinations(df_males.index, 2)) if df_males.loc[i].equals(df_males.loc[j])]

# Ottengo le colonne che hanno campi vuoti
df_males.loc[:, df_males.isnull().any()]

# Ottengo le coppie di colonne uguali
[(i, j) for i, j in list(it.combinations(df_males.columns, 2)) if df_males[i].equals(df_males[j])]

# Ottengo le colonne costanti
df_males.columns[df_males.nunique() == 1]

# +
# Boxplot per ogni attributo del dataset
plt.figure(figsize=(12, 6))
plt.boxplot(df_males.drop(columns=['target']).values, tick_labels=df_males.drop(columns=['target']).columns, vert=False)

plt.title('Boxplot per ogni attributo')
plt.show()

# +
# # %OP = OP/(OP+CL)*100
# t%OP = tOP/(tOP+tCL+tCENT)*100
# tCENT = (300-tOP-tCL)
try:
    df_males = df_males.drop(columns=['tOP', 'tCL', 'tCENT'])
except:
    pass
    
X_males = df_males.drop(columns=['target'])

df_males_2 = copy.copy(df_males)
y_males_unique_text_2 = df_males_2['target'].str.split(' - ').str[0].unique()
df_males_2['target'] = pd.factorize(df_males_2['target'].str.split(' - ').str[0])[0]
y_males_2 = df_males_2['target']

df_males_3 = copy.copy(df_males)
y_males_unique_text_3 = df_males_3['target'].unique()
df_males_3['target'] = pd.factorize(df_males_3['target'])[0]
y_males_3 = df_males_3['target']

df_males_stress = copy.copy(df_males)
df_males_stress = df_males_stress[df_males_stress['target'] != 'no stress']
X_males_stress = df_males_stress.drop(columns=['target'])
y_males_unique_text_stress = df_males_stress['target'].unique()
df_males_stress['target'] = pd.factorize(df_males_stress['target'])[0]
y_males_stress = df_males_stress['target']
# -

# Matrice di correlazione
visualize.show_correlation_matrix(df_males_2, 'Matrice di Correlazione (df_males_2)')
visualize.show_correlation_matrix(df_males_3, 'Matrice di Correlazione (df_males_3)')
visualize.show_correlation_matrix(df_males_stress, 'Matrice di Correlazione (df_males_stress)')

# +
# PCA_X_MALES
pca_males = PCA(n_components=len(X_males.columns))
pca_males.fit(StandardScaler().fit_transform(X_males))

explained_variance = np.cumsum(pca_males.explained_variance_ratio_)
minc_x_males = np.argmax(explained_variance > 0.9)+1

visualize.show_cumulative_explained_variance(explained_variance, 'Cumulative Explained Variance X_males')
# -

# PCA con 2 componenti
pca_males_2c = PCA(n_components=2)
X_males_2c = pca_males_2c.fit_transform(StandardScaler().fit_transform(X_males))
pd.DataFrame(pca_males_2c.components_, columns=X_males.columns, index=[f'Cmp {i+1}' for i in range(pca_males_2c.n_components_)])

# Plot PCA 2 componenti, 2 e 3 target
visualize.show_scatter_plot(X_males_2c, y_males_2, y_males_unique_text_2, ['b', 'm'], 'PCA - Scatter Plot (df_males_2)')
visualize.show_scatter_plot(X_males_2c, y_males_3, y_males_unique_text_3, ['b', 'm', 'g'], 'PCA - Scatter Plot (df_males_3)')

# K-Means
visualize.show_cluster_plot(2, X_males_2c, y_males_2, y_males_unique_text_2, ['b', 'm'], 'Cluster Plot (df_males_2)')
visualize.show_cluster_plot(3, X_males_2c, y_males_3, y_males_unique_text_3, ['b', 'm', 'g'], 'Cluster Plot (df_males_3)')

# SVC_KERNELS_X2_MALES_T2
svc_kernels(X_males_2c, y_males_2.values, y_males_unique_text_2, ['b', 'm'],
            ml.get_models(ml.SVC_KERNELS, '_X2_MALES_T2'), 
            ml.get_binary_scorers(), 
            replace=False)

# SVC_KERNELS_X2_MALES_T3
svc_kernels(X_males_2c, y_males_3.values, y_males_unique_text_3, ['b', 'm', 'g'],
            ml.get_models(ml.SVC_KERNELS, '_X2_MALES_T3'), 
            ml.get_multiclass_scorers(), 
            replace=False)

# PCA con 3 componenti
pca_males_3c = PCA(n_components=3)
X_males_3c = pca_males_3c.fit_transform(StandardScaler().fit_transform(X_males))
pd.DataFrame(pca_males_3c.components_, columns=X_males.columns, index=[f'Cmp {i+1}' for i in range(pca_males_3c.n_components_)])

# Scatter Plot 3D dei dati trasformati da PCA con 3 componenti
visualize.show_3D_scatter_plot(X_males_3c, y_males_2, y_males_unique_text_2, ['b', 'm'], 'Scatter Plot (df_males_2)')
visualize.show_3D_scatter_plot(X_males_3c, y_males_3, y_males_unique_text_3, ['b', 'm', 'g'], 'Scatter Plot (df_males_3)')

# SVC_KERNELS_X3_MALES_T2
svc_kernels(X_males_3c, y_males_2.values, y_males_unique_text_2, ['b', 'm'],
            ml.get_models(ml.SVC_KERNELS, '_X3_MALES_T2'), 
            ml.get_binary_scorers(), 
            replace=False)

# SVC_KERNELS_X3_MALES_T3
svc_kernels(X_males_3c, y_males_3.values, y_males_unique_text_3, ['b', 'm', 'g'],
            ml.get_models(ml.SVC_KERNELS, '_X3_MALES_T3'), 
            ml.get_multiclass_scorers(), 
            replace=False)

# PCA con min_components_x_males
pca_males_minc = PCA(n_components=minc_x_males)
X_males_minc = pca_males_minc.fit_transform(StandardScaler().fit_transform(X_males))
pd.DataFrame(pca_males_minc.components_, columns=X_males.columns, index=[f'Cmp {i+1}' for i in range(pca_males_minc.n_components_)])

# SVC_KERNELS_MINC_MALES_T2
svc_kernels(X_males_minc, y_males_2.values, y_males_unique_text_2, ['b', 'm'],
            ml.get_models(ml.SVC_KERNELS, '_MINC_MALES_T2'), 
            ml.get_binary_scorers(), 
            replace=False)

# SVC_KERNELS_MINC_MALES_T3
svc_kernels(X_males_minc, y_males_3.values, y_males_unique_text_3, ['b', 'm', 'g'],
            ml.get_models(ml.SVC_KERNELS, '_MINC_MALES_T3'), 
            ml.get_multiclass_scorers(), 
            replace=False)

# K-Means_X_MALES_T2, K-Means_X_MALES_T3
visualize.show_cluster_table(2, StandardScaler().fit_transform(X_males), y_males_2, y_males_unique_text_2, 'df_males_2')
visualize.show_cluster_table(3, StandardScaler().fit_transform(X_males), y_males_3, y_males_unique_text_3, 'df_males_3')

# MODELS_X_MALES_T2
models(X_males, y_males_2.values, y_males_unique_text_2, 
       ml.get_models(ml.MODELS, '_X_MALES_T2'), 
       ml.get_binary_scorers(), 
       replace=False)
# MODELS_X_MALES_T3
models(X_males, y_males_3.values, y_males_unique_text_3, 
       ml.get_models(ml.MODELS, '_X_MALES_T3'), 
       ml.get_multiclass_scorers(), 
       replace=False)

# +
# PCA_X_MALES_STRESS
pca_males_stress = PCA(n_components=len(X_males_stress.columns))
pca_males_stress.fit(StandardScaler().fit_transform(X_males_stress))

explained_variance = np.cumsum(pca_males_stress.explained_variance_ratio_)
minc_x_males_stress = np.argmax(explained_variance > 0.9)+1

visualize.show_cumulative_explained_variance(explained_variance, 'Cumulative Explained Variance X_MALES_STRESS')
# -

# PCA con due componenti
pca_males_stress_2c = PCA(n_components=2)
X_males_stress_2c = pca_males_stress_2c.fit_transform(StandardScaler().fit_transform(X_males_stress))
pd.DataFrame(pca_males_stress_2c.components_, columns=X_males_stress.columns, index=[f'Cmp {i+1}' for i in range(pca_males_stress_2c.n_components_)])

# Plot PCA 2 componenti
visualize.show_scatter_plot(X_males_stress_2c, y_males_stress, y_males_unique_text_stress, ['b', 'm'], 'PCA - Scatter Plot (df_males_stress)')

# K-Means
visualize.show_cluster_plot(2, X_males_stress_2c, y_males_stress, y_males_unique_text_stress, ['b', 'm'], 'Cluster Plot (df_males_stress)')

# SVC_KERNELS_X2_MALES_STRESS
svc_kernels(X_males_stress_2c, y_males_stress.values, y_males_unique_text_stress, ['b', 'm'],
            ml.get_models(ml.SVC_KERNELS, '_X2_MALES_STRESS'), 
            ml.get_binary_scorers(), 
            replace=False)

# PCA con 3 componenti
pca_males_stress_3c = PCA(n_components=3)
X_males_stress_3c = pca_males_stress_3c.fit_transform(StandardScaler().fit_transform(X_males_stress))
pd.DataFrame(pca_males_stress_3c.components_, columns=X_males_stress.columns, index=[f'Cmp {i+1}' for i in range(pca_males_stress_3c.n_components_)])

# Scatter Plot 3D dei dati trasformati da PCA con 3 componenti
visualize.show_3D_scatter_plot(X_males_stress_3c, y_males_stress, y_males_unique_text_stress, ['b', 'm'], 'Scatter Plot (df_males_stress)')

# SVC_KERNELS_X3_MALES_STRESS
svc_kernels(X_males_stress_3c, y_males_stress.values, y_males_unique_text_stress, ['b', 'm'],
            ml.get_models(ml.SVC_KERNELS, '_X3_MALES_STRESS'), 
            ml.get_binary_scorers(), 
            replace=False)

# PCA con minc_x_males_stress
pca_males_stress_minc = PCA(n_components=minc_x_males_stress)
X_males_stress_minc = pca_males_stress_minc.fit_transform(StandardScaler().fit_transform(X_males_stress))
pd.DataFrame(pca_males_stress_minc.components_, columns=X_males_stress.columns, index=[f'Cmp {i+1}' for i in range(pca_males_stress_minc.n_components_)])

# SVC_KERNELS_MINC_MALES_STRESS
svc_kernels(X_males_stress_minc, y_males_stress.values, y_males_unique_text_stress, ['b', 'm'],
            ml.get_models(ml.SVC_KERNELS, '_MINC_MALES_STRESS'), 
            ml.get_binary_scorers(), 
            replace=False)

# K-Means_X_MALES_STRESS
visualize.show_cluster_table(2, StandardScaler().fit_transform(X_males_stress), y_males_stress, y_males_unique_text_stress, 'df_males_stress')

# MODELS_X_MALES_STRESS
models(X_males_stress, y_males_stress.values, y_males_unique_text_stress, 
       ml.get_models(ml.MODELS, '_X_MALES_STRESS'), 
       ml.get_binary_scorers(), 
       replace=False)

# # Mouse female

df_females = pd.read_csv('data/females.csv', index_col='ID animals', dtype={'NOR index': 'float64', '% OP': 'float64', 't%OP': 'float64'})
df_females

# Ottengo le righe che hanno campi vuoti
df_females[df_females.isnull().any(axis=1)]

# Ottengo le coppie di righe uguali
[(i, j) for i, j in list(it.combinations(df_females.index, 2)) if df_females.loc[i].equals(df_females.loc[j])]

# Ottengo le colonne che hanno campi vuoti
df_females.loc[:, df_females.isnull().any()]

# Ottengo le coppie di colonne uguali
[(i, j) for i, j in list(it.combinations(df_females.columns, 2)) if df_females[i].equals(df_females[j])]

# Ottengo le colonne costanti
df_females.columns[df_females.nunique() == 1]

# +
# Boxplot per ogni attributo del dataset
plt.figure(figsize=(12, 6))
plt.boxplot(df_females.drop(columns=['target']).values, tick_labels=df_females.drop(columns=['target']).columns, vert=False)

plt.title('Boxplot per ogni attributo')
plt.show()

# +
# # %OP = OP/(OP+CL)*100
# t%OP = tOP/(tOP+tCL+tCENT)*100
# tCENT = (300-tOP-tCL)
try:
    df_females = df_females.drop(columns=['tOP', 'tCL', 'tCENT'])
except:
    pass
    
X_females = df_females.drop(columns=['target'])

df_females_2 = copy.copy(df_females)
y_females_unique_text_2 = df_females_2['target'].str.split(' - ').str[0].unique()
df_females_2['target'] = pd.factorize(df_females_2['target'].str.split(' - ').str[0])[0]
y_females_2 = df_females_2['target']

df_females_3 = copy.copy(df_females)
y_females_unique_text_3 = df_females_3['target'].unique()
df_females_3['target'] = pd.factorize(df_females_3['target'])[0]
y_females_3 = df_females_3['target']

df_females_stress = copy.copy(df_females)
df_females_stress = df_females_stress[df_females_stress['target'] != 'no stress']
X_females_stress = df_females_stress.drop(columns=['target'])
y_females_unique_text_stress = df_females_stress['target'].unique()
df_females_stress['target'] = pd.factorize(df_females_stress['target'])[0]
y_females_stress = df_females_stress['target']
# -

# Matrice di correlazione
visualize.show_correlation_matrix(df_females_2, 'Matrice di Correlazione (df_females_2)')
visualize.show_correlation_matrix(df_females_3, 'Matrice di Correlazione (df_females_3)')
visualize.show_correlation_matrix(df_females_stress, 'Matrice di Correlazione (df_females_stress)')

# +
# PCA_X_FEMALES
pca_females = PCA(n_components=len(X_females.columns))
pca_females.fit(StandardScaler().fit_transform(X_females))

explained_variance = np.cumsum(pca_females.explained_variance_ratio_)
minc_x_females = np.argmax(explained_variance > 0.9)+1

visualize.show_cumulative_explained_variance(explained_variance, 'Cumulative Explained Variance X_females')
# -

# PCA con 2 componenti
pca_females_2c = PCA(n_components=2)
X_females_2c = pca_females_2c.fit_transform(StandardScaler().fit_transform(X_females))
pd.DataFrame(pca_females_2c.components_, columns=X_females.columns, index=[f'Cmp {i+1}' for i in range(pca_females_2c.n_components_)])

# Plot PCA 2 componenti, 2 e 3 target
visualize.show_scatter_plot(X_females_2c, y_females_2, y_females_unique_text_2, ['b', 'm'], 'PCA - Scatter Plot (df_females_2)')
visualize.show_scatter_plot(X_females_2c, y_females_3, y_females_unique_text_3, ['b', 'm', 'g'], 'PCA - Scatter Plot (df_females_3)')

# K-Means
visualize.show_cluster_plot(2, X_females_2c, y_females_2, y_females_unique_text_2, ['b', 'm'], 'Cluster Plot (df_females_2)')
visualize.show_cluster_plot(3, X_females_2c, y_females_3, y_females_unique_text_3, ['b', 'm', 'g'], 'Cluster Plot (df_females_3)')

# SVC_KERNELS_X2_FEMALES_T2
svc_kernels(X_females_2c, y_females_2.values, y_females_unique_text_2, ['b', 'm'],
            ml.get_models(ml.SVC_KERNELS, '_X2_FEMALES_T2'), 
            ml.get_binary_scorers(), 
            replace=False)

# SVC_KERNELS_X2_FEMALES_T3
svc_kernels(X_females_2c, y_females_3.values, y_females_unique_text_3, ['b', 'm', 'g'],
            ml.get_models(ml.SVC_KERNELS, '_X2_FEMALES_T3'), 
            ml.get_multiclass_scorers(), 
            replace=False)

# PCA con 3 componenti
pca_females_3c = PCA(n_components=3)
X_females_3c = pca_females_3c.fit_transform(StandardScaler().fit_transform(X_females))
pd.DataFrame(pca_females_3c.components_, columns=X_females.columns, index=[f'Cmp {i+1}' for i in range(pca_females_3c.n_components_)])

# Scatter Plot 3D dei dati trasformati da PCA con 3 componenti
visualize.show_3D_scatter_plot(X_females_3c, y_females_2, y_females_unique_text_2, ['b', 'm'], 'Scatter Plot (df_females_2)')
visualize.show_3D_scatter_plot(X_females_3c, y_females_3, y_females_unique_text_3, ['b', 'm', 'g'], 'Scatter Plot (df_females_3)')

# SVC_KERNELS_X3_FEMALES_T2
svc_kernels(X_females_3c, y_females_2.values, y_females_unique_text_2, ['b', 'm'],
            ml.get_models(ml.SVC_KERNELS, '_X3_FEMALES_T2'), 
            ml.get_binary_scorers(), 
            replace=False)

# SVC_KERNELS_X3_FEMALES_T3
svc_kernels(X_females_3c, y_females_3.values, y_females_unique_text_3, ['b', 'm', 'g'],
            ml.get_models(ml.SVC_KERNELS, '_X3_FEMALES_T3'), 
            ml.get_multiclass_scorers(), 
            replace=False)

# PCA con min_components_x_females
pca_females_minc = PCA(n_components=minc_x_females)
X_females_minc = pca_females_minc.fit_transform(StandardScaler().fit_transform(X_females))
pd.DataFrame(pca_females_minc.components_, columns=X_females.columns, index=[f'Cmp {i+1}' for i in range(pca_females_minc.n_components_)])

# SVC_KERNELS_MINC_FEMALES_T2
svc_kernels(X_females_minc, y_females_2.values, y_females_unique_text_2, ['b', 'm'],
            ml.get_models(ml.SVC_KERNELS, '_MINC_FEMALES_T2'), 
            ml.get_binary_scorers(), 
            replace=False)

# SVC_KERNELS_MINC_FEMALES_T3
svc_kernels(X_females_minc, y_females_3.values, y_females_unique_text_3, ['b', 'm', 'g'],
            ml.get_models(ml.SVC_KERNELS, '_MINC_FEMALES_T3'), 
            ml.get_multiclass_scorers(), 
            replace=False)

# K-Means_X_FEMALES_T2, K-Means_X_FEMALES_T3
visualize.show_cluster_table(2, StandardScaler().fit_transform(X_females), y_females_2, y_females_unique_text_2, 'df_females_2')
visualize.show_cluster_table(3, StandardScaler().fit_transform(X_females), y_females_3, y_females_unique_text_3, 'df_females_3')

# MODELS_X_FEMALES_T2
models(X_females, y_females_2.values, y_females_unique_text_2, 
       ml.get_models(ml.MODELS, '_X_FEMALES_T2'), 
       ml.get_binary_scorers(), 
       replace=False)

# MODELS_X_FEMALES_T3
models(X_females, y_females_3.values, y_females_unique_text_3, 
       ml.get_models(ml.MODELS, '_X_FEMALES_T3'), 
       ml.get_multiclass_scorers(), 
       replace=False)

# +
# PCA_X_FEMALES_STRESS
pca_females_stress = PCA(n_components=len(X_females_stress.columns))
pca_females_stress.fit(StandardScaler().fit_transform(X_females_stress))

explained_variance = np.cumsum(pca_females_stress.explained_variance_ratio_)
minc_x_females_stress = np.argmax(explained_variance > 0.9)+1

visualize.show_cumulative_explained_variance(explained_variance, 'Cumulative Explained Variance X_FEMALES_STRESS')
# -

# PCA con due componenti
pca_females_stress_2c = PCA(n_components=2)
X_females_stress_2c = pca_females_stress_2c.fit_transform(StandardScaler().fit_transform(X_females_stress))
pd.DataFrame(pca_females_stress_2c.components_, columns=X_females_stress.columns, index=[f'Cmp {i+1}' for i in range(pca_females_stress_2c.n_components_)])

# Plot PCA 2 componenti
visualize.show_scatter_plot(X_females_stress_2c, y_females_stress, y_females_unique_text_stress, ['b', 'm'], 'PCA - Scatter Plot (df_females_stress)')

# K-Means
visualize.show_cluster_plot(2, X_females_stress_2c, y_females_stress, y_females_unique_text_stress, ['b', 'm'], 'Cluster Plot (df_females_stress)')

# SVC_KERNELS_X2_FEMALES_STRESS
svc_kernels(X_females_stress_2c, y_females_stress.values, y_females_unique_text_stress, ['b', 'm'],
            ml.get_models(ml.SVC_KERNELS, '_X2_FEMALES_STRESS'), 
            ml.get_binary_scorers(), 
            replace=False)

# PCA con 3 componenti
pca_females_stress_3c = PCA(n_components=3)
X_females_stress_3c = pca_females_stress_3c.fit_transform(StandardScaler().fit_transform(X_females_stress))
pd.DataFrame(pca_females_stress_3c.components_, columns=X_females_stress.columns, index=[f'Cmp {i+1}' for i in range(pca_females_stress_3c.n_components_)])

# Scatter Plot 3D dei dati trasformati da PCA con 3 componenti
visualize.show_3D_scatter_plot(X_females_stress_3c, y_females_stress, y_females_unique_text_stress, ['b', 'm'], 'Scatter Plot (df_females_stress)')

# SVC_KERNELS_X3_FEMALES_STRESS
svc_kernels(X_females_stress_3c, y_females_stress.values, y_females_unique_text_stress, ['b', 'm'],
            ml.get_models(ml.SVC_KERNELS, '_X3_FEMALES_STRESS'), 
            ml.get_binary_scorers(), 
            replace=False)

# PCA con minc_x_females_stress
pca_females_stress_minc = PCA(n_components=minc_x_females_stress)
X_females_stress_minc = pca_females_stress_minc.fit_transform(StandardScaler().fit_transform(X_females_stress))
pd.DataFrame(pca_females_stress_minc.components_, columns=X_females_stress.columns, index=[f'Cmp {i+1}' for i in range(pca_females_stress_minc.n_components_)])

# SVC_KERNELS_MINC_FEMALES_STRESS
svc_kernels(X_females_stress_minc, y_females_stress.values, y_females_unique_text_stress, ['b', 'm'],
            ml.get_models(ml.SVC_KERNELS, '_MINC_FEMALES_STRESS'), 
            ml.get_binary_scorers(), 
            replace=False)

# K-Means_X_FEMALES_STRESS
visualize.show_cluster_table(2, StandardScaler().fit_transform(X_females_stress), y_females_stress, y_females_unique_text_stress, 'df_females_stress')

# MODELS_X_FEMALES_STRESS
models(X_females_stress, y_females_stress.values, y_females_unique_text_stress, 
       ml.get_models(ml.MODELS, '_X_FEMALES_STRESS'), 
       ml.get_binary_scorers(), 
       replace=False)


