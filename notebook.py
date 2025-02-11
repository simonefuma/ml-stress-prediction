# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

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
    
# SCRIVERE IL PRIMO CAPITOLO
# -

# # Util

def get_dataframes(df):
    df_2 = copy.copy(df)
    y_unique_text_2 = df_2['target'].str.split(' - ').str[0].unique()
    df_2['target'] = pd.factorize(df_2['target'].str.split(' - ').str[0])[0]
    y_2 = df_2['target']
    
    df_3 = copy.copy(df)
    y_unique_text_3 = df_3['target'].unique()
    df_3['target'] = pd.factorize(df_3['target'])[0]
    y_3 = df_3['target']
    
    df_stress = copy.copy(df)
    df_stress = df_stress[df_stress['target'] != 'no stress']
    y_unique_text_stress = df_stress['target'].unique()
    df_stress['target'] = pd.factorize(df_stress['target'])[0]
    y_stress = df_stress['target']

    return ((df_2, y_2, y_unique_text_2), (df_3, y_3, y_unique_text_3), (df_stress, y_stress, y_unique_text_stress))


# da mettere in visualize
def show_linear_transform_table(n_components, X):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    display(pd.DataFrame(pca.components_, columns=X.columns, index=[f'Cmp {i+1}' for i in range(pca.n_components_)]))
    latex_code = pd.DataFrame(pca.components_, columns=X.columns, index=[f'Cmp {i+1}' for i in range(pca.n_components_)]).to_latex(float_format="%.4f")
    print(latex_code)
    return X_pca, pca.components_


def get_components(X, coefficents):
    return np.array([
        [float(sum(coefficent*row)) for coefficent in coefficents]
        for row in StandardScaler().fit_transform(X.values)
    ])


def get_modified(X, coefficents):
    coefficents_modified = np.where(
        np.abs(coefficents) >= np.mean(np.abs(coefficents), axis=1, keepdims=True)/10,
        coefficents,
        0)

    X_modified = get_components(X, coefficents_modified)

    display(pd.DataFrame(coefficents_modified, columns=X.columns, index=[f'Cmp {i+1}' for i in range(len(coefficents_modified))]))
    return X_modified, coefficents_modified


def show_scores(learned_models, X, y, scorers):
    models_name = [learned_model['model_name'] for learned_model in learned_models]
    scorers_name = [scorer.__name__ for scorer in scorers]
    scores = [
        [float(round(scorer(y, learned_model['model'].predict(X)), 3)) for scorer in scorers]  
              for learned_model in learned_models
    ]
    display(pd.DataFrame(scores, columns=scorers_name, index=models_name))

    latex_code = pd.DataFrame(scores, columns=scorers_name, index=models_name).to_latex(float_format="%.4f")
    print(latex_code)


def models(X, y, y_unique_text, colors, models, test_scorers, gender, columns,
           outer_split=4, inner_split=3, 
           index_test_scorer=0, minimize_test_scorer=False, 
           replace=False):
    learned_models = ml.learn_models(X, y, models,
                                 StratifiedKFold(n_splits=outer_split, shuffle=True, random_state=RANDOM_STATE),
                                 StratifiedKFold(n_splits=inner_split, shuffle=True, random_state=RANDOM_STATE),
                                 test_scorers=test_scorers, index_test_scorer=index_test_scorer, minimize_test_scorer=minimize_test_scorer, 
                                 replace=replace)
    visualize.display_table(learned_models)
    
    svc_learned_models = []
    for model, learned_model in zip(models, learned_models):
        visualize.display_hyperparameters(model['name'], model['param_grid'], learned_model['model'])
        if(model['name'].startswith("SVC")):
            svc_learned_models.append(learned_model)

    if X.shape[1] == 2:
        visualize.show_svc_decision_boundary(X, y, y_unique_text, svc_learned_models, colors, gender)
        
    for learned_model in learned_models:
        if(learned_model['model'].named_steps['classifier'].__class__.__name__ == 'DecisionTreeClassifier'):
            visualize.plot_tree(columns, y_unique_text, learned_model['model'].named_steps['classifier'], learned_model['model_name'])
        elif(learned_model['model'].named_steps['classifier'].__class__.__name__ == 'RandomForestClassifier'):
            visualize.plot_forest(columns, y_unique_text, learned_model['model'].named_steps['classifier'], learned_model['model_name'])

    return learned_models


# + [markdown] jp-MarkdownHeadingCollapsed=true
# # Topi Maschio
# -

# ## Analisi dei Dati

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

plt.title('Boxplot di ogni attributo')
plt.show()
# -

# Matrice di correlazione
((df_males_2, y_males_2, y_males_unique_text_2), (df_males_3, y_males_3, y_males_unique_text_3), (df_males_stress, y_males_stress, y_males_unique_text_stress)) = get_dataframes(df_males)
X_males = df_males.drop(columns=['target'])
X_males_stress = df_males_stress.drop(columns=['target'])
visualize.show_correlation_matrix(df_males_2, 'Matrice di Correlazione (df_males_2)')
visualize.show_correlation_matrix(df_males_3, 'Matrice di Correlazione (df_males_3)')
visualize.show_correlation_matrix(df_males_stress, 'Matrice di Correlazione (df_males_stress)')

# # %OP = OP/(OP+CL)*100
# t%OP = tOP/(tOP+tCL+tCENT)*100
# tCENT = (300-tOP-tCL)
try:
    X_males = X_males.drop(columns=['tOP', 'tCL', 'tCENT'])
    X_males_stress = X_males_stress.drop(columns=['tOP', 'tCL', 'tCENT'])
    df_males_2 = df_males_2.drop(columns=['tOP', 'tCL', 'tCENT'])
    df_males_3 = df_males_3.drop(columns=['tOP', 'tCL', 'tCENT'])
    df_males_stress = df_males_stress.drop(columns=['tOP', 'tCL', 'tCENT'])
except:
    pass

# +
# PCA_X_MALES
pca_males = PCA(n_components=len(X_males.columns))
pca_males.fit(StandardScaler().fit_transform(X_males))

explained_variance = np.cumsum(pca_males.explained_variance_ratio_)
minc_x_males = np.argmax(explained_variance > 0.9)+1

visualize.show_cumulative_explained_variance(explained_variance, 'Cumulative Explained Variance X_MALES')

# +
# PCA_X_MALES_STRESS
pca_males_stress = PCA(n_components=len(X_males_stress.columns))
pca_males_stress.fit(StandardScaler().fit_transform(X_males_stress))

explained_variance = np.cumsum(pca_males_stress.explained_variance_ratio_)
minc_x_males_stress = np.argmax(explained_variance > 0.9)+1

visualize.show_cumulative_explained_variance(explained_variance, 'Cumulative Explained Variance X_MALES_STRESS')
# -

# ## Studio dei Modelli

# ### No Stress/Stress

# #### PCA a 2 componenti

# PCA con 2 componenti
X_males_2c, X_males_2c_coefficents = show_linear_transform_table(2, X_males)

# Scatter Plot
visualize.show_scatter_plot(X_males_2c, y_males_2, y_males_unique_text_2, ['b', 'm'], 'Scatter Plot', np.full(len(X_males_2c), 0))

# K-Means
visualize.show_cluster_plot(2, X_males_2c, y_males_2, y_males_unique_text_2, ['b', 'm'], 'Cluster Plot', np.full(len(X_males_2c), 0))

learned_models_x2_males_t2 = models(X_males_2c, y_males_2.values, y_males_unique_text_2, ['b', 'm'],
                                    ml.get_models('_X2_MALES_T2', 13), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_males_2c), 0),
                                    [f"cmp {i}" for i in range(1, X_males_2c.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_males_2c_modified, X_males_2c_coefficents_modified = get_modified(X_males, X_males_2c_coefficents)

# #### PCA a 3 componenti

# PCA con 3 componenti
X_males_3c, X_males_3c_coefficents = show_linear_transform_table(3, X_males)

# Scatter Plot 3D
visualize.show_3D_scatter_plot(X_males_3c, y_males_2, y_males_unique_text_2, ['b', 'm'], 'Scatter Plot')

learned_models_x3_males_t2 = models(X_males_3c, y_males_2.values, y_males_unique_text_2, ['b', 'm'],
                                    ml.get_models('_X3_MALES_T2', 13), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_males_3c), 0),
                                    [f"cmp {i}" for i in range(1, X_males_3c.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_males_3c_modified, X_males_3c_coefficents_modified = get_modified(X_males, X_males_3c_coefficents)

# #### PCA con numero di componenti minimi per avere una varianza cumulata maggiore del 90%

# PCA con min_components_x_males
X_males_minc, X_males_minc_coefficents = show_linear_transform_table(minc_x_males, X_males)

learned_models_minc_males_t2 = models(X_males_minc, y_males_2.values, y_males_unique_text_2, ['b', 'm'],
                                    ml.get_models('_MINC_MALES_T2', 13), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_males_minc), 0),
                                    [f"cmp {i}" for i in range(1, X_males_minc.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_males_minc_modified, X_males_minc_coefficents_modified = get_modified(X_males, X_males_minc_coefficents)

learned_models_minc_males_t2_modified = models(X_males_minc_modified, y_males_2.values, y_males_unique_text_2, ['b', 'm'],
                                    ml.get_models('_MINC_MALES_T2_MODIFIED', 13), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_males_minc_modified), 0),
                                    [f"cmp {i}" for i in range(1, X_males_minc_modified.shape[1] + 1)],
                                    replace=False)

# #### Senza PCA

# K-Means_X_MALES_T2
visualize.show_cluster_table(2, StandardScaler().fit_transform(X_males), y_males_2, y_males_unique_text_2, 'X_MALES')

learned_models_x_males_t2 = models(X_males.values, y_males_2.values, y_males_unique_text_2, ['b', 'm'],
                                    ml.get_models('_X_MALES_T2', 13), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_males), 0),
                                    X_males.columns.values,
                                    replace=False)

# ### No Stress/Stress Resiliente/Stress Vunlerabile

# #### PCA a 2 componenti

# PCA con 2 componenti
X_males_2c, X_males_2c_coefficents = show_linear_transform_table(2, X_males)

# Scatter Plot
visualize.show_scatter_plot(X_males_2c, y_males_3, y_males_unique_text_3, ['b', 'm', 'g'], 'Scatter Plot', np.full(len(X_males_2c), 0))

# K-Means
visualize.show_cluster_plot(3, X_males_2c, y_males_3, y_males_unique_text_3, ['b', 'm', 'g'], 'Cluster Plot', np.full(len(X_males_2c), 0))

learned_models_x2_males_t3 = models(X_males_2c, y_males_3.values, y_males_unique_text_3, ['b', 'm', 'g'],
                                    ml.get_models('_X2_MALES_T3', 13), 
                                    ml.get_multiclass_scorers(),
                                    np.full(len(X_males_2c), 0),
                                    [f"cmp {i}" for i in range(1, X_males_2c.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_males_2c_modified, X_males_2c_coefficents_modified = get_modified(X_males, X_males_2c_coefficents)

# #### PCA a 3 componenti

# PCA con 3 componenti
X_males_3c, X_males_3c_coefficents = show_linear_transform_table(3, X_males)

# Scatter Plot 3D
visualize.show_3D_scatter_plot(X_males_3c, y_males_3, y_males_unique_text_3, ['b', 'm', 'g'], 'Scatter Plot')

learned_models_x3_males_t3 = models(X_males_3c, y_males_3.values, y_males_unique_text_3, ['b', 'm', 'g'],
                                    ml.get_models('_X3_MALES_T3', 13), 
                                    ml.get_multiclass_scorers(),
                                    np.full(len(X_males_3c), 0),
                                    [f"cmp {i}" for i in range(1, X_males_3c.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_males_3c_modified, X_males_3c_coefficents_modified = get_modified(X_males, X_males_3c_coefficents)

# #### PCA con numero di componenti minimi per avere una varianza cumulata maggiore del 90%

# PCA con min_components_x_males
X_males_minc, X_males_minc_coefficents = show_linear_transform_table(minc_x_males, X_males)

learned_models_minc_males_t3 = models(X_males_minc, y_males_3.values, y_males_unique_text_3, ['b', 'm', 'g'],
                                    ml.get_models('_MINC_MALES_T3', 13), 
                                    ml.get_multiclass_scorers(),
                                    np.full(len(X_males_minc), 0),
                                    [f"cmp {i}" for i in range(1, X_males_minc.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_males_minc_modified, X_males_minc_coefficents_modified = get_modified(X_males, X_males_minc_coefficents)

learned_models_minc_males_t3_modified = models(X_males_minc_modified, y_males_3.values, y_males_unique_text_3, ['b', 'm', 'g'],
                                    ml.get_models('_MINC_MALES_T3_MODIFIED', 13), 
                                    ml.get_multiclass_scorers(),
                                    np.full(len(X_males_minc_modified), 0),
                                    [f"cmp {i}" for i in range(1, X_males_minc_modified.shape[1] + 1)],
                                    replace=False)

# #### Senza PCA

# K-Means_X_MALES_T3
visualize.show_cluster_table(3, StandardScaler().fit_transform(X_males), y_males_3, y_males_unique_text_3, 'X_MALES')

learned_models_x_males_t3 = models(X_males.values, y_males_3.values, y_males_unique_text_3, ['b', 'm', 'g'],
                                    ml.get_models('_X_MALES_T3', 13), 
                                    ml.get_multiclass_scorers(),
                                    np.full(len(X_males), 0),
                                    X_males.columns.values,
                                    replace=False)

# ### Stress Resiliente/Stress Vulnerabile

# #### PCA a 2 componenti

# PCA con due componenti
X_males_stress_2c, X_males_stress_2c_coefficents = show_linear_transform_table(2, X_males_stress)

# Plot PCA 2 componenti
visualize.show_scatter_plot(X_males_stress_2c, y_males_stress, y_males_unique_text_stress, ['b', 'm'], 'Scatter Plot', np.full(len(X_males_stress_2c), 0))

# K-Means
visualize.show_cluster_plot(2, X_males_stress_2c, y_males_stress, y_males_unique_text_stress, ['b', 'm'], 'Cluster Plot', np.full(len(X_males_stress_2c), 0))

learned_models_x2_males_stress = models(X_males_stress_2c, y_males_stress.values, y_males_unique_text_stress, ['b', 'm'],
                                    ml.get_models('_X2_MALES_STRESS', 9), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_males_stress_2c), 0),
                                    [f"cmp {i}" for i in range(1, X_males_stress_2c.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_males_stress_2c_modified, X_males_stress_2c_coefficents_modified = get_modified(X_males_stress, X_males_stress_2c_coefficents)

# #### PCA a 3 componenti

# PCA con 3 componenti
X_males_stress_3c, X_males_stress_3c_coefficents = show_linear_transform_table(3, X_males_stress)

# Scatter Plot 3D
visualize.show_3D_scatter_plot(X_males_stress_3c, y_males_stress, y_males_unique_text_stress, ['b', 'm'], 'Scatter Plot')

learned_models_x3_males_stress = models(X_males_stress_3c, y_males_stress.values, y_males_unique_text_stress, ['b', 'm'],
                                    ml.get_models('_X3_MALES_STRESS', 9), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_males_stress_3c), 0),
                                    [f"cmp {i}" for i in range(1, X_males_stress_3c.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_males_stress_3c_modified, X_males_stress_3c_coefficents_modified = get_modified(X_males_stress, X_males_stress_3c_coefficents)

# #### PCA con numero di componenti minimi per avere una varianza cumulata maggiore del 90%

# PCA con minc_x_males_stress
X_males_stress_minc, X_males_stress_minc_coefficents = show_linear_transform_table(minc_x_males_stress, X_males_stress)

learned_models_minc_males_stress = models(X_males_stress_minc, y_males_stress.values, y_males_unique_text_stress, ['b', 'm'],
                                    ml.get_models('_MINC_MALES_STRESS', 9), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_males_stress_minc), 0),
                                    [f"cmp {i}" for i in range(1, X_males_stress_minc.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_males_stress_minc_modified, X_males_stress_minc_coefficents_modified = get_modified(X_males_stress, X_males_stress_minc_coefficents)

learned_models_minc_males_stress_modified = models(X_males_stress_minc_modified, y_males_stress.values, y_males_unique_text_stress, ['b', 'm'],
                                    ml.get_models('_MINC_MALES_STRESS_MODIFIED', 9), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_males_stress_minc_modified), 0),
                                    [f"cmp {i}" for i in range(1, X_males_stress_minc_modified.shape[1] + 1)],
                                    replace=False)

# #### Senza PCA

# K-Means_X_MALES_STRESS
visualize.show_cluster_table(2, StandardScaler().fit_transform(X_males_stress), y_males_stress, y_males_unique_text_stress, 'X_MALES')

learned_models_x_males_stress = models(X_males_stress.values, y_males_stress.values, y_males_unique_text_stress, ['b', 'm'],
                                    ml.get_models('_X_MALES_STRESS', 9), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_males_stress), 0),
                                    X_males_stress.columns.values,
                                    replace=False)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# # Topi Femmina
# -

# ## Analisi dei Dati

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

plt.title('Boxplot di ogni attributo')
plt.show()
# -

# Matrice di correlazione
((df_females_2, y_females_2, y_females_unique_text_2), (df_females_3, y_females_3, y_females_unique_text_3), (df_females_stress, y_females_stress, y_females_unique_text_stress)) = get_dataframes(df_females)
X_females = df_females.drop(columns=['target'])
X_females_stress = df_females_stress.drop(columns=['target'])
visualize.show_correlation_matrix(df_females_2, 'Matrice di Correlazione (df_females_2)')
visualize.show_correlation_matrix(df_females_3, 'Matrice di Correlazione (df_females_3)')
visualize.show_correlation_matrix(df_females_stress, 'Matrice di Correlazione (df_females_stress)')

# # %OP = OP/(OP+CL)*100
# t%OP = tOP/(tOP+tCL+tCENT)*100
# tCENT = (300-tOP-tCL)
try:
    X_females = X_females.drop(columns=['tOP', 'tCL', 'tCENT'])
    X_females_stress = X_females_stress.drop(columns=['tOP', 'tCL', 'tCENT'])
    df_females_2 = df_females_2.drop(columns=['tOP', 'tCL', 'tCENT'])
    df_females_3 = df_females_3.drop(columns=['tOP', 'tCL', 'tCENT'])
    df_females_stress = df_females_stress.drop(columns=['tOP', 'tCL', 'tCENT'])
except:
    pass

# +
# PCA_X_FEMALES
pca_females = PCA(n_components=len(X_females.columns))
pca_females.fit(StandardScaler().fit_transform(X_females))

explained_variance = np.cumsum(pca_females.explained_variance_ratio_)
minc_x_females = np.argmax(explained_variance > 0.9)+1

visualize.show_cumulative_explained_variance(explained_variance, 'Cumulative Explained Variance X_FEMALES')

# +
# PCA_X_FEMALES_STRESS
pca_females_stress = PCA(n_components=len(X_females_stress.columns))
pca_females_stress.fit(StandardScaler().fit_transform(X_females_stress))

explained_variance = np.cumsum(pca_females_stress.explained_variance_ratio_)
minc_x_females_stress = np.argmax(explained_variance > 0.9)+1

visualize.show_cumulative_explained_variance(explained_variance, 'Cumulative Explained Variance X_FEMALES_STRESS')
# -

# ## Studio dei Modelli

# ### No Stress/Stress

# #### PCA a 2 componenti

# PCA con 2 componenti
X_females_2c, X_females_2c_coefficents = show_linear_transform_table(2, X_females)

# Scatter Plot
visualize.show_scatter_plot(X_females_2c, y_females_2, y_females_unique_text_2, ['b', 'm'], 'Scatter Plot', np.full(len(X_females_2c), 1))

# K-Means
visualize.show_cluster_plot(2, X_females_2c, y_females_2, y_females_unique_text_2, ['b', 'm'], 'Cluster Plot', np.full(len(X_females_2c), 1))

learned_models_x2_females_t2 = models(X_females_2c, y_females_2.values, y_females_unique_text_2, ['b', 'm'],
                                    ml.get_models('_X2_FEMALES_T2', 13), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_females_2c), 1),
                                    [f"cmp {i}" for i in range(1, X_females_2c.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_females_2c_modified, X_females_2c_coefficents_modified = get_modified(X_females, X_females_2c_coefficents)

# #### PCA a 3 componenti

# PCA con 3 componenti
X_females_3c, X_females_3c_coefficents = show_linear_transform_table(3, X_females)

# Scatter Plot 3D
visualize.show_3D_scatter_plot(X_females_3c, y_females_2, y_females_unique_text_2, ['b', 'm'], 'Scatter Plot')

learned_models_x3_females_t2 = models(X_females_3c, y_females_2.values, y_females_unique_text_2, ['b', 'm'],
                                    ml.get_models('_X3_FEMALES_T2', 13), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_females_3c), 1),
                                    [f"cmp {i}" for i in range(1, X_females_3c.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_females_3c_modified, X_females_3c_coefficents_modified = get_modified(X_females, X_females_3c_coefficents)

learned_models_x3_females_t2_modified = models(X_females_3c_modified, y_females_2.values, y_females_unique_text_2, ['b', 'm'],
                                    ml.get_models('_X3_FEMALES_T2_MODIFIED', 13), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_females_3c_modified), 1),
                                    [f"cmp {i}" for i in range(1, X_females_3c_modified.shape[1] + 1)],
                                    replace=False)

# #### PCA con numero di componenti minimi per avere una varianza cumulata maggiore del 90%

# PCA con min_components_x_females
X_females_minc, X_females_minc_coefficents = show_linear_transform_table(minc_x_females, X_females)

learned_models_minc_females_t2 = models(X_females_minc, y_females_2.values, y_females_unique_text_2, ['b', 'm'],
                                    ml.get_models('_MINC_FEMALES_T2', 13), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_females_minc), 1),
                                    [f"cmp {i}" for i in range(1, X_females_minc.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_females_minc_modified, X_females_minc_coefficents_modified = get_modified(X_females, X_females_minc_coefficents)

learned_models_minc_females_t2_modified = models(X_females_minc_modified, y_females_2.values, y_females_unique_text_2, ['b', 'm'],
                                    ml.get_models('_MINC_FEMALES_T2_MODIFIED', 13), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_females_minc_modified), 1),
                                    [f"cmp {i}" for i in range(1, X_females_minc_modified.shape[1] + 1)],
                                    replace=False)

# #### Senza PCA

# K-Means_X_FEMALES_T2
visualize.show_cluster_table(2, StandardScaler().fit_transform(X_females), y_females_2, y_females_unique_text_2, 'X_FEMALES')

learned_models_x_females_t2 = models(X_females.values, y_females_2.values, y_females_unique_text_2, ['b', 'm'],
                                    ml.get_models('_X_FEMALES_T2', 13), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_females), 1),
                                    X_females.columns.values,
                                    replace=False)

# ### No Stress/Stress Resiliente/Stress Vunlerabile

# #### PCA a 2 componenti

# PCA con 2 componenti
X_females_2c, X_females_2c_coefficents = show_linear_transform_table(2, X_females)

# Scatter Plot
visualize.show_scatter_plot(X_females_2c, y_females_3, y_females_unique_text_3, ['b', 'm', 'g'], 'Scatter Plot', np.full(len(X_females_2c), 1))

# K-Means
visualize.show_cluster_plot(3, X_females_2c, y_females_3, y_females_unique_text_3, ['b', 'm', 'g'], 'Cluster Plot', np.full(len(X_females_2c), 1))

learned_models_x2_females_t3 = models(X_females_2c, y_females_3.values, y_females_unique_text_3, ['b', 'm', 'g'],
                                    ml.get_models('_X2_FEMALES_T3', 13), 
                                    ml.get_multiclass_scorers(),
                                    np.full(len(X_females_2c), 1),
                                    [f"cmp {i}" for i in range(1, X_females_2c.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_females_2c_modified, X_females_2c_coefficents_modified = get_modified(X_females, X_females_2c_coefficents)

# #### PCA a 3 componenti

# PCA con 3 componenti
X_females_3c, X_females_3c_coefficents = show_linear_transform_table(3, X_females)

# Scatter Plot 3D
visualize.show_3D_scatter_plot(X_females_3c, y_females_3, y_females_unique_text_3, ['b', 'm', 'g'], 'Scatter Plot')

learned_models_x3_females_t3 = models(X_females_3c, y_females_3.values, y_females_unique_text_3, ['b', 'm', 'g'],
                                    ml.get_models('_X3_FEMALES_T3', 13), 
                                    ml.get_multiclass_scorers(),
                                    np.full(len(X_females_3c), 1),
                                    [f"cmp {i}" for i in range(1, X_females_3c.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_females_3c_modified, X_females_3c_coefficents_modified = get_modified(X_females, X_females_3c_coefficents)

learned_models_x3_females_t3_modified = models(X_females_3c_modified, y_females_3.values, y_females_unique_text_3, ['b', 'm', 'g'],
                                    ml.get_models('_X3_FEMALES_T3_MODIFIED', 13), 
                                    ml.get_multiclass_scorers(),
                                    np.full(len(X_females_3c_modified), 1),
                                    [f"cmp {i}" for i in range(1, X_females_3c_modified.shape[1] + 1)],
                                    replace=False)

# #### PCA con numero di componenti minimi per avere una varianza cumulata maggiore del 90%

# PCA con min_components_x_females
X_females_minc, X_females_minc_coefficents = show_linear_transform_table(minc_x_females, X_females)

learned_models_minc_females_t3 = models(X_females_minc, y_females_3.values, y_females_unique_text_3, ['b', 'm', 'g'],
                                    ml.get_models('_MINC_FEMALES_T3', 13), 
                                    ml.get_multiclass_scorers(),
                                    np.full(len(X_females_minc), 1),
                                    [f"cmp {i}" for i in range(1, X_females_minc.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_females_minc_modified, X_females_minc_coefficents_modified = get_modified(X_females, X_females_minc_coefficents)

learned_models_minc_females_t3_modified = models(X_females_minc_modified, y_females_3.values, y_females_unique_text_3, ['b', 'm', 'g'],
                                    ml.get_models('_MINC_FEMALES_T3_MODIFIED', 13), 
                                    ml.get_multiclass_scorers(),
                                    np.full(len(X_females_minc_modified), 1),
                                    [f"cmp {i}" for i in range(1, X_females_minc_modified.shape[1] + 1)],
                                    replace=False)

# #### Senza PCA

# K-Means_X_FEMALES_T3
visualize.show_cluster_table(3, StandardScaler().fit_transform(X_females), y_females_3, y_females_unique_text_3, 'X_FEMALES')

learned_models_x_females_t3 = models(X_females.values, y_females_3.values, y_females_unique_text_3, ['b', 'm', 'g'],
                                    ml.get_models('_X_FEMALES_T3', 13), 
                                    ml.get_multiclass_scorers(),
                                    np.full(len(X_females), 1),
                                    X_females.columns.values,
                                    replace=False)

# ### Stress Resiliente/Stress Vulnerabile

# #### PCA a 2 componenti

# PCA con due componenti
X_females_stress_2c, X_females_stress_2c_coefficents = show_linear_transform_table(2, X_females_stress)

# Plot PCA 2 componenti
visualize.show_scatter_plot(X_females_stress_2c, y_females_stress, y_females_unique_text_stress, ['b', 'm'], 'Scatter Plot', np.full(len(X_females_stress_2c), 1))

# K-Means
visualize.show_cluster_plot(2, X_females_stress_2c, y_females_stress, y_females_unique_text_stress, ['b', 'm'], 'Cluster Plot', np.full(len(X_females_stress_2c), 1))

learned_models_x2_females_stress = models(X_females_stress_2c, y_females_stress.values, y_females_unique_text_stress, ['b', 'm'],
                                    ml.get_models('_X2_FEMALES_STRESS', 9), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_females_stress_2c), 1),
                                    [f"cmp {i}" for i in range(1, X_females_stress_2c.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_females_stress_2c_modified, X_females_stress_2c_coefficents_modified = get_modified(X_females_stress, X_females_stress_2c_coefficents)

# #### PCA a 3 componenti

# PCA con 3 componenti
X_females_stress_3c, X_females_stress_3c_coefficents = show_linear_transform_table(3, X_females_stress)

# Scatter Plot 3D
visualize.show_3D_scatter_plot(X_females_stress_3c, y_females_stress, y_females_unique_text_stress, ['b', 'm'], 'Scatter Plot')

learned_models_x3_females_stress = models(X_females_stress_3c, y_females_stress.values, y_females_unique_text_stress, ['b', 'm'],
                                    ml.get_models('_X3_FEMALES_STRESS', 9), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_females_stress_3c), 1),
                                    [f"cmp {i}" for i in range(1, X_females_stress_3c.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_females_stress_3c_modified, X_females_stress_3c_coefficents_modified = get_modified(X_females_stress, X_females_stress_3c_coefficents)

learned_models_x3_females_stress_modified = models(X_females_stress_3c_modified, y_females_stress.values, y_females_unique_text_stress, ['b', 'm'],
                                    ml.get_models('_X3_FEMALES_STRESS_MODIFIED', 9), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_females_stress_3c_modified), 1),
                                    [f"cmp {i}" for i in range(1, X_females_stress_3c_modified.shape[1] + 1)],
                                    replace=False)

# #### PCA con numero di componenti minimi per avere una varianza cumulata maggiore del 90%

# PCA con minc_x_females_stress
X_females_stress_minc, X_females_stress_minc_coefficents = show_linear_transform_table(minc_x_females_stress, X_females_stress)

learned_models_minc_females_stress = models(X_females_stress_minc, y_females_stress.values, y_females_unique_text_stress, ['b', 'm'],
                                    ml.get_models('_MINC_FEMALES_STRESS', 9), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_females_stress_minc), 1),
                                    [f"cmp {i}" for i in range(1, X_females_stress_minc.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_females_stress_minc_modified, X_females_stress_minc_coefficents_modified = get_modified(X_females_stress, X_females_stress_minc_coefficents)

learned_models_minc_females_stress_modified = models(X_females_stress_minc_modified, y_females_stress.values, y_females_unique_text_stress, ['b', 'm'],
                                    ml.get_models('_MINC_FEMALES_STRESS_MODIFIED', 9), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_females_stress_minc_modified), 1),
                                    [f"cmp {i}" for i in range(1, X_females_stress_minc_modified.shape[1] + 1)],
                                    replace=False)

# #### Senza PCA

# K-Means_X_FEMALES_STRESS
visualize.show_cluster_table(2, StandardScaler().fit_transform(X_females_stress), y_females_stress, y_females_unique_text_stress, 'X_FEMALES')

learned_models_x_females_stress = models(X_females_stress.values, y_females_stress.values, y_females_unique_text_stress, ['b', 'm'],
                                    ml.get_models('_X_FEMALES_STRESS', 9), 
                                    ml.get_binary_scorers(),
                                    np.full(len(X_females_stress), 1),
                                    X_females_stress.columns.values,
                                    replace=False)

# # Tutti i Topi

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## Analisi dei Dati

# +
df_males_copy = copy.copy(df_males)
df_males_copy['gender'] = 0
df_females_copy = copy.copy(df_females)
df_females_copy['gender'] = 1
df_females_copy.index += len(df_males_copy)

df_all = df_combined = pd.concat([df_males_copy, df_females_copy], axis=0)
df_all
# -

# Ottengo le righe che hanno campi vuoti
df_all[df_all.isnull().any(axis=1)]

# Ottengo le coppie di righe uguali
[(i, j) for i, j in list(it.combinations(df_all.index, 2)) if df_all.loc[i].equals(df_all.loc[j])]

# Ottengo le colonne che hanno campi vuoti
df_all.loc[:, df_all.isnull().any()]

# Ottengo le coppie di colonne uguali
[(i, j) for i, j in list(it.combinations(df_all.columns, 2)) if df_all[i].equals(df_all[j])]

# Ottengo le colonne costanti
df_all.columns[df_all.nunique() == 1]

# +
# Boxplot per ogni attributo del dataset
plt.figure(figsize=(12, 6))
plt.boxplot(df_all.drop(columns=['target']).values, tick_labels=df_all.drop(columns=['target']).columns, vert=False)

plt.title('Boxplot di ogni attributo')
plt.show()
# -

# Matrice di correlazione
((df_all_2, y_all_2, y_all_unique_text_2), (df_all_3, y_all_3, y_all_unique_text_3), (df_all_stress, y_all_stress, y_all_unique_text_stress)) = get_dataframes(df_all)
X_all = df_all.drop(columns=['target'])
X_all_stress = df_all_stress.drop(columns=['target'])
visualize.show_correlation_matrix(df_all_2, 'Matrice di Correlazione (df_all_2)')
visualize.show_correlation_matrix(df_all_3, 'Matrice di Correlazione (df_all_3)')
visualize.show_correlation_matrix(df_all_stress, 'Matrice di Correlazione (df_all_stress)')

# # %OP = OP/(OP+CL)*100
# t%OP = tOP/(tOP+tCL+tCENT)*100
# tCENT = (300-tOP-tCL)
try:
    X_all = X_all.drop(columns=['tOP', 'tCL', 'tCENT'])
    X_all_stress = X_all_stress.drop(columns=['tOP', 'tCL', 'tCENT'])
    df_all_2 = df_all_2.drop(columns=['tOP', 'tCL', 'tCENT'])
    df_all_3 = df_all_3.drop(columns=['tOP', 'tCL', 'tCENT'])
    df_all_stress = df_all_stress.drop(columns=['tOP', 'tCL', 'tCENT'])
except:
    pass

# +
# PCA_X_ALL
pca_all = PCA(n_components=len(X_all.columns))
pca_all.fit(StandardScaler().fit_transform(X_all))

explained_variance = np.cumsum(pca_all.explained_variance_ratio_)
minc_x_all = np.argmax(explained_variance > 0.9)+1

visualize.show_cumulative_explained_variance(explained_variance, 'Cumulative Explained Variance X_ALL')

# +
# PCA_X_ALL_STRESS
pca_all_stress = PCA(n_components=len(X_all_stress.columns))
pca_all_stress.fit(StandardScaler().fit_transform(X_all_stress))

explained_variance = np.cumsum(pca_all_stress.explained_variance_ratio_)
minc_x_all_stress = np.argmax(explained_variance > 0.9)+1

visualize.show_cumulative_explained_variance(explained_variance, 'Cumulative Explained Variance X_ALL_STRESS')
# -

# ## Studio dei Modelli

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ### No Stress/Stress

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA a 2 componenti
# -

# PCA con 2 componenti
X_all_2c, X_all_2c_coefficents = show_linear_transform_table(2, X_all)

# Scatter Plot
visualize.show_scatter_plot(X_all_2c, y_all_2, y_all_unique_text_2, ['b', 'm'], 'Scatter Plot', X_all['gender'])

# K-Means
visualize.show_cluster_plot(2, X_all_2c, y_all_2, y_all_unique_text_2, ['b', 'm'], 'Cluster Plot', X_all['gender'])

learned_models_x2_all_t2 = models(X_all_2c, y_all_2.values, y_all_unique_text_2, ['b', 'm'],
                                    ml.get_models('_X2_ALL_T2', 25), 
                                    ml.get_binary_scorers(),
                                    X_all['gender'],
                                    [f"cmp {i}" for i in range(1, X_all_2c.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_all_2c_modified, X_all_2c_coefficents_modified = get_modified(X_all, X_all_2c_coefficents)

learned_models_x2_all_t2_modified = models(X_all_2c_modified, y_all_2.values, y_all_unique_text_2, ['b', 'm'],
                                    ml.get_models('_X2_ALL_T2_MODIFIED', 25), 
                                    ml.get_binary_scorers(),
                                    X_all['gender'],
                                    [f"cmp {i}" for i in range(1, X_all_2c_modified.shape[1] + 1)],
                                    replace=False)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA a 3 componenti
# -

# PCA con 3 componenti
X_all_3c, X_all_3c_coefficents = show_linear_transform_table(3, X_all)

# Scatter Plot 3D
visualize.show_3D_scatter_plot(X_all_3c, y_all_2, y_all_unique_text_2, ['b', 'm'], 'Scatter Plot')

learned_models_x3_all_t2 = models(X_all_3c, y_all_2.values, y_all_unique_text_2, ['b', 'm'],
                                    ml.get_models('_X3_ALL_T2', 25), 
                                    ml.get_binary_scorers(),
                                    X_all['gender'],
                                    [f"cmp {i}" for i in range(1, X_all_3c.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_all_3c_modified, X_all_3c_coefficents_modified = get_modified(X_all, X_all_3c_coefficents)

learned_models_x3_all_t2_modified = models(X_all_3c_modified, y_all_2.values, y_all_unique_text_2, ['b', 'm'],
                                    ml.get_models('_X3_ALL_T2_MODIFIED', 25), 
                                    ml.get_binary_scorers(),
                                    X_all['gender'],
                                    [f"cmp {i}" for i in range(1, X_all_3c_modified.shape[1] + 1)],
                                    replace=False)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA con numero di componenti minimi per avere una varianza cumulata maggiore del 90%
# -

# PCA con min_components_x_all
X_all_minc, X_all_minc_coefficents = show_linear_transform_table(minc_x_all, X_all)

learned_models_minc_all_t2 = models(X_all_minc, y_all_2.values, y_all_unique_text_2, ['b', 'm'],
                                    ml.get_models('_MINC_ALL_T2', 25), 
                                    ml.get_binary_scorers(),
                                    X_all['gender'],
                                    [f"cmp {i}" for i in range(1, X_all_minc.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_all_minc_modified, X_all_minc_coefficents_modified = get_modified(X_all, X_all_minc_coefficents)

learned_models_minc_all_t2_modified = models(X_all_minc_modified, y_all_2.values, y_all_unique_text_2, ['b', 'm'],
                                    ml.get_models('_MINC_ALL_T2_MODIFIED', 25), 
                                    ml.get_binary_scorers(),
                                    X_all['gender'],
                                    [f"cmp {i}" for i in range(1, X_all_minc_modified.shape[1] + 1)],
                                    replace=False)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### Senza PCA
# -

# K-Means_X_ALL_T2
visualize.show_cluster_table(2, StandardScaler().fit_transform(X_all), y_all_2, y_all_unique_text_2, 'X_ALL')

learned_models_x_all_t2 = models(X_all.values, y_all_2.values, y_all_unique_text_2, ['b', 'm'],
                                    ml.get_models('_X_ALL_T2', 25), 
                                    ml.get_binary_scorers(),
                                    X_all['gender'],
                                    X_all.columns.values,
                                    replace=False)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ### No Stress/Stress Resiliente/Stress Vunlerabile

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA a 2 componenti
# -

# PCA con 2 componenti
X_all_2c, X_all_2c_coefficents = show_linear_transform_table(2, X_all)

# Scatter Plot
visualize.show_scatter_plot(X_all_2c, y_all_3, y_all_unique_text_3, ['b', 'm', 'g'], 'Scatter Plot', X_all['gender'])

# K-Means
visualize.show_cluster_plot(3, X_all_2c, y_all_3, y_all_unique_text_3, ['b', 'm', 'g'], 'Cluster Plot', X_all['gender'])

learned_models_x2_all_t3 = models(X_all_2c, y_all_3.values, y_all_unique_text_3, ['b', 'm', 'g'],
                                    ml.get_models('_X2_ALL_T3', 25), 
                                    ml.get_multiclass_scorers(),
                                    X_all['gender'],
                                    [f"cmp {i}" for i in range(1, X_all_2c.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_all_2c_modified, X_all_2c_coefficents_modified = get_modified(X_all, X_all_2c_coefficents)

learned_models_x2_all_t3_modified = models(X_all_2c_modified, y_all_3.values, y_all_unique_text_3, ['b', 'm', 'g'],
                                    ml.get_models('_X2_ALL_T3_MODIFIED', 25), 
                                    ml.get_multiclass_scorers(),
                                    X_all['gender'],
                                    [f"cmp {i}" for i in range(1, X_all_2c_modified.shape[1] + 1)],
                                    replace=False)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA a 3 componenti
# -

# PCA con 3 componenti
X_all_3c, X_all_3c_coefficents = show_linear_transform_table(3, X_all)

# Scatter Plot 3D
visualize.show_3D_scatter_plot(X_all_3c, y_all_3, y_all_unique_text_3, ['b', 'm', 'g'], 'Scatter Plot')

learned_models_x3_all_t3 = models(X_all_3c, y_all_3.values, y_all_unique_text_3, ['b', 'm', 'g'],
                                    ml.get_models('_X3_ALL_T3', 25), 
                                    ml.get_multiclass_scorers(),
                                    X_all['gender'],
                                    [f"cmp {i}" for i in range(1, X_all_3c.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_all_3c_modified, X_all_3c_coefficents_modified = get_modified(X_all, X_all_3c_coefficents)

learned_models_x3_all_t3_modified = models(X_all_3c_modified, y_all_3.values, y_all_unique_text_3, ['b', 'm', 'g'],
                                    ml.get_models('_X3_ALL_T3_MODIFIED', 25), 
                                    ml.get_multiclass_scorers(),
                                    X_all['gender'],
                                    [f"cmp {i}" for i in range(1, X_all_3c_modified.shape[1] + 1)],
                                    replace=False)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA con numero di componenti minimi per avere una varianza cumulata maggiore del 90%
# -

# PCA con min_components_x_all
X_all_minc, X_all_minc_coefficents = show_linear_transform_table(minc_x_all, X_all)

learned_models_minc_all_t3 = models(X_all_minc, y_all_3.values, y_all_unique_text_3, ['b', 'm', 'g'],
                                    ml.get_models('_MINC_ALL_T3', 25), 
                                    ml.get_multiclass_scorers(),
                                    X_all['gender'],
                                    [f"cmp {i}" for i in range(1, X_all_minc.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_all_minc_modified, X_all_minc_coefficents_modified = get_modified(X_all, X_all_minc_coefficents)

learned_models_minc_all_t3_modified = models(X_all_minc_modified, y_all_3.values, y_all_unique_text_3, ['b', 'm', 'g'],
                                    ml.get_models('_MINC_ALL_T3_MODIFIED', 25), 
                                    ml.get_multiclass_scorers(),
                                    X_all['gender'],
                                    [f"cmp {i}" for i in range(1, X_all_minc_modified.shape[1] + 1)],
                                    replace=False)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### Senza PCA
# -

# K-Means_X_ALL_T3
visualize.show_cluster_table(3, StandardScaler().fit_transform(X_all), y_all_3, y_all_unique_text_3, 'X_ALL')

learned_models_x_all_t3 = models(X_all.values, y_all_3.values, y_all_unique_text_3, ['b', 'm', 'g'],
                                    ml.get_models('_X_ALL_T3', 25), 
                                    ml.get_multiclass_scorers(),
                                    X_all['gender'],
                                    X_all.columns.values,
                                    replace=False)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ### Stress Resiliente/Stress Vulnerabile

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA a 2 componenti
# -

# PCA con due componenti
X_all_stress_2c, X_all_stress_2c_coefficents = show_linear_transform_table(2, X_all_stress)

# Plot PCA 2 componenti
visualize.show_scatter_plot(X_all_stress_2c, y_all_stress, y_all_unique_text_stress, ['b', 'm'], 'Scatter Plot', X_all_stress['gender'])

# K-Means
visualize.show_cluster_plot(2, X_all_stress_2c, y_all_stress, y_all_unique_text_stress, ['b', 'm'], 'Cluster Plot', X_all_stress['gender'])

learned_models_x2_all_stress = models(X_all_stress_2c, y_all_stress.values, y_all_unique_text_stress, ['b', 'm'],
                                    ml.get_models('_X2_ALL_STRESS', 17), 
                                    ml.get_binary_scorers(),
                                    X_all_stress['gender'],
                                    [f"cmp {i}" for i in range(1, X_all_stress_2c.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_all_stress_2c_modified, X_all_stress_2c_coefficents_modified = get_modified(X_all_stress, X_all_stress_2c_coefficents)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA a 3 componenti
# -

# PCA con 3 componenti
X_all_stress_3c, X_all_stress_3c_coefficents = show_linear_transform_table(3, X_all_stress)

# Scatter Plot 3D
visualize.show_3D_scatter_plot(X_all_stress_3c, y_all_stress, y_all_unique_text_stress, ['b', 'm'], 'Scatter Plot')

learned_models_x3_all_stress = models(X_all_stress_3c, y_all_stress.values, y_all_unique_text_stress, ['b', 'm'],
                                    ml.get_models('_X3_ALL_STRESS', 17), 
                                    ml.get_binary_scorers(),
                                    X_all_stress['gender'],
                                    [f"cmp {i}" for i in range(1, X_all_stress_3c.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_all_stress_3c_modified, X_all_stress_3c_coefficents_modified = get_modified(X_all_stress, X_all_stress_3c_coefficents)

learned_models_x3_all_stress_modified = models(X_all_stress_3c_modified, y_all_stress.values, y_all_unique_text_stress, ['b', 'm'],
                                    ml.get_models('_X3_ALL_STRESS_MODIFIED', 17), 
                                    ml.get_binary_scorers(),
                                    X_all_stress['gender'],
                                    [f"cmp {i}" for i in range(1, X_all_stress_3c_modified.shape[1] + 1)],
                                    replace=False)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA con numero di componenti minimi per avere una varianza cumulata maggiore del 90%
# -

# PCA con minc_x_all_stress
X_all_stress_minc, X_all_stress_minc_coefficents = show_linear_transform_table(minc_x_all_stress, X_all_stress)

learned_models_minc_all_stress = models(X_all_stress_minc, y_all_stress.values, y_all_unique_text_stress, ['b', 'm'],
                                    ml.get_models('_MINC_ALL_STRESS', 17), 
                                    ml.get_binary_scorers(),
                                    X_all_stress['gender'],
                                    [f"cmp {i}" for i in range(1, X_all_stress_minc.shape[1] + 1)],
                                    replace=False)

# se non ci sono differenze non vengono fatti altri esperimenti
X_all_stress_minc_modified, X_all_stress_minc_coefficents_modified = get_modified(X_all_stress, X_all_stress_minc_coefficents)

learned_models_minc_all_stress_modified = models(X_all_stress_minc_modified, y_all_stress.values, y_all_unique_text_stress, ['b', 'm'],
                                    ml.get_models('_MINC_ALL_STRESS_MODIFIED', 17), 
                                    ml.get_binary_scorers(),
                                    X_all_stress['gender'],
                                    [f"cmp {i}" for i in range(1, X_all_stress_minc_modified.shape[1] + 1)],
                                    replace=False)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### Senza PCA
# -

# K-Means_X_ALL_STRESS
visualize.show_cluster_table(2, StandardScaler().fit_transform(X_all_stress), y_all_stress, y_all_unique_text_stress, 'X_ALL')

learned_models_x_all_stress = models(X_all_stress.values, y_all_stress.values, y_all_unique_text_stress, ['b', 'm'],
                                    ml.get_models('_X_ALL_STRESS', 17), 
                                    ml.get_binary_scorers(),
                                    X_all_stress['gender'],
                                    X_all_stress.columns.values,
                                    replace=False)

# # Modelli Testati sul Sesso Opposto

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## Modelli Addestrati sui Topi Maschio, Testati sui Topi Femmina

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ### No Stress/Stress

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA a 2 componenti
# -

X_females_2c_males = get_components(X_females, X_males_2c_coefficents)
show_scores(learned_models_x2_males_t2, X_females_2c_males, y_females_2, ml.get_binary_scorers())
visualize.show_svc_decision_boundary(X_females_2c_males, y_females_2, y_females_unique_text_2, 
                                     [learned_model for learned_model in learned_models_x2_males_t2 if learned_model['model_name'].startswith('SVC')], 
                                     ['b', 'm'], np.full(len(X_females_2c_males), 1))

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA a 3 componenti
# -

X_females_3c_males = get_components(X_females, X_males_3c_coefficents)
show_scores(learned_models_x3_males_t2, X_females_3c_males, y_females_2, ml.get_binary_scorers())

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA con numero di componenti minimi per avere una varianza cumulata maggiore del 90%
# -

X_females_minc_males = get_components(X_females, X_males_minc_coefficents)
show_scores(learned_models_minc_males_t2, X_females_minc_males, y_females_2, ml.get_binary_scorers())

X_females_minc_males_modified = get_components(X_females, X_males_minc_coefficents_modified)
show_scores(learned_models_minc_males_t2_modified, X_females_minc_males_modified, y_females_2, ml.get_binary_scorers())

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### Senza PCA
# -

show_scores(learned_models_x_males_t2, X_females.values, y_females_2, ml.get_binary_scorers())

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ### No Stress/Stress Resiliente/Stress Vunlerabile

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA a 2 componenti
# -

X_females_2c_males = get_components(X_females, X_males_2c_coefficents)
show_scores(learned_models_x2_males_t3, X_females_2c_males, y_females_3, ml.get_multiclass_scorers())
visualize.show_svc_decision_boundary(X_females_2c_males, y_females_3, y_females_unique_text_3, 
                                     [learned_model for learned_model in learned_models_x2_males_t3 if learned_model['model_name'].startswith('SVC')], 
                                     ['b', 'm', 'g'], np.full(len(X_females_2c_males), 1))

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA a 3 componenti
# -

X_females_3c_males = get_components(X_females, X_males_3c_coefficents)
show_scores(learned_models_x3_males_t3, X_females_3c_males, y_females_3, ml.get_multiclass_scorers())

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA con numero di componenti minimi per avere una varianza cumulata maggiore del 90%
# -

X_females_minc_males = get_components(X_females, X_males_minc_coefficents)
show_scores(learned_models_minc_males_t3, X_females_minc_males, y_females_3, ml.get_multiclass_scorers())

X_females_minc_males_modified = get_components(X_females, X_males_minc_coefficents_modified)
show_scores(learned_models_minc_males_t3_modified, X_females_minc_males_modified, y_females_3, ml.get_multiclass_scorers())

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### Senza PCA
# -

show_scores(learned_models_x_males_t3, X_females.values, y_females_3, ml.get_multiclass_scorers())

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ### Stress Resiliente/Stress Vulnerabile

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA a 2 componenti
# -

X_females_stress_2c_males = get_components(X_females_stress, X_males_stress_2c_coefficents)
show_scores(learned_models_x2_males_stress, X_females_stress_2c_males, y_females_stress, ml.get_binary_scorers())
visualize.show_svc_decision_boundary(X_females_stress_2c_males, y_females_stress, y_females_unique_text_stress, 
                                     [learned_model for learned_model in learned_models_x2_males_stress if learned_model['model_name'].startswith('SVC')], 
                                     ['b', 'm'], np.full(len(X_females_stress_2c_males), 1))

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA a 3 componenti
# -

X_females_stress_3c_males = get_components(X_females_stress, X_males_stress_3c_coefficents)
show_scores(learned_models_x3_males_stress, X_females_stress_3c_males, y_females_stress, ml.get_binary_scorers())

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA con numero di componenti minimi per avere una varianza cumulata maggiore del 90%
# -

X_females_stress_minc_males = get_components(X_females_stress, X_males_stress_minc_coefficents)
show_scores(learned_models_minc_males_stress, X_females_stress_minc_males, y_females_stress, ml.get_binary_scorers())

X_females_stress_minc_males_modified = get_components(X_females_stress, X_males_stress_minc_coefficents_modified)
show_scores(learned_models_minc_males_stress_modified, X_females_stress_minc_males_modified, y_females_stress, ml.get_binary_scorers())

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### Senza PCA
# -

show_scores(learned_models_x_males_stress, X_females_stress.values, y_females_stress, ml.get_binary_scorers())

# ## Modelli Addestrati sui Topi Femmina, Testati sui Topi Maschio

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ### No Stress/Stress

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA a 2 componenti
# -

X_males_2c_females = get_components(X_males, X_females_2c_coefficents)
show_scores(learned_models_x2_females_t2, X_males_2c_females, y_males_2, ml.get_binary_scorers())
visualize.show_svc_decision_boundary(X_males_2c_females, y_males_2, y_males_unique_text_2, 
                                     [learned_model for learned_model in learned_models_x2_females_t2 if learned_model['model_name'].startswith('SVC')], 
                                     ['b', 'm'], np.full(len(X_males_2c_females), 0))

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA a 3 componenti
# -

X_males_3c_females = get_components(X_males, X_females_3c_coefficents)
show_scores(learned_models_x3_females_t2, X_males_3c_females, y_males_2, ml.get_binary_scorers())

X_males_3c_females_modified = get_components(X_males, X_females_3c_coefficents_modified)
show_scores(learned_models_x3_females_t2_modified, X_males_3c_females_modified, y_males_2, ml.get_binary_scorers())

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA con numero di componenti minimi per avere una varianza cumulata maggiore del 90%
# -

X_males_minc_females = get_components(X_males, X_females_minc_coefficents)
show_scores(learned_models_minc_females_t2, X_males_minc_females, y_males_2, ml.get_binary_scorers())

X_males_minc_females_modified = get_components(X_males, X_females_minc_coefficents_modified)
show_scores(learned_models_minc_females_t2_modified, X_males_minc_females_modified, y_males_2, ml.get_binary_scorers())

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### Senza PCA
# -

show_scores(learned_models_x_females_t2, X_males.values, y_males_2, ml.get_binary_scorers())

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ### No Stress/Stress Resiliente/Stress Vunlerabile

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA a 2 componenti
# -

X_males_2c_females = get_components(X_males, X_females_2c_coefficents)
show_scores(learned_models_x2_females_t3, X_males_2c_females, y_males_3, ml.get_multiclass_scorers())
visualize.show_svc_decision_boundary(X_males_2c_females, y_males_3, y_males_unique_text_3, 
                                     [learned_model for learned_model in learned_models_x2_females_t3 if learned_model['model_name'].startswith('SVC')], 
                                     ['b', 'm', 'g'], np.full(len(X_males_2c_females), 0))

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA a 3 componenti
# -

X_males_3c_females = get_components(X_males, X_females_3c_coefficents)
show_scores(learned_models_x3_females_t3, X_males_3c_females, y_males_3, ml.get_multiclass_scorers())

X_males_3c_females_modified = get_components(X_males, X_females_3c_coefficents_modified)
show_scores(learned_models_x3_females_t3_modified, X_males_3c_females_modified, y_males_3, ml.get_multiclass_scorers())

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA con numero di componenti minimi per avere una varianza cumulata maggiore del 90%
# -

X_males_minc_females = get_components(X_males, X_females_minc_coefficents)
show_scores(learned_models_minc_females_t3, X_males_minc_females, y_males_3, ml.get_multiclass_scorers())

X_males_minc_females_modified = get_components(X_males, X_females_minc_coefficents_modified)
show_scores(learned_models_minc_females_t3_modified, X_males_minc_females_modified, y_males_3, ml.get_multiclass_scorers())

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### Senza PCA
# -

show_scores(learned_models_x_females_t3, X_males.values, y_males_3, ml.get_multiclass_scorers())

# ### Stress Resiliente/Stress Vulnerabile

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA a 2 componenti
# -

X_males_stress_2c_females = get_components(X_males_stress, X_females_stress_2c_coefficents)
show_scores(learned_models_x2_females_stress, X_males_stress_2c_females, y_males_stress, ml.get_binary_scorers())
visualize.show_svc_decision_boundary(X_males_stress_2c_females, y_males_stress, y_males_unique_text_stress, 
                                     [learned_model for learned_model in learned_models_x2_females_stress if learned_model['model_name'].startswith('SVC')], 
                                     ['b', 'm'], np.full(len(X_males_stress_2c_females), 0))

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA a 3 componenti
# -

X_males_stress_3c_females = get_components(X_males_stress, X_females_stress_3c_coefficents)
show_scores(learned_models_x3_females_stress, X_males_stress_3c_females, y_males_stress, ml.get_binary_scorers())

X_males_stress_3c_females_modified = get_components(X_males_stress, X_females_stress_3c_coefficents_modified)
show_scores(learned_models_x3_females_stress_modified, X_males_stress_3c_females_modified, y_males_stress, ml.get_binary_scorers())

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### PCA con numero di componenti minimi per avere una varianza cumulata maggiore del 90%
# -

X_males_stress_minc_females = get_components(X_males_stress, X_females_stress_minc_coefficents)
show_scores(learned_models_minc_females_stress, X_males_stress_minc_females, y_males_stress, ml.get_binary_scorers())

X_males_stress_minc_females_modified = get_components(X_males_stress, X_females_stress_minc_coefficents_modified)
show_scores(learned_models_minc_females_stress_modified, X_males_stress_minc_females_modified, y_males_stress, ml.get_binary_scorers())

# #### Senza PCA

show_scores(learned_models_x_females_stress, X_males_stress.values, y_males_stress, ml.get_binary_scorers())






