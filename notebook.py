# +
import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
# -

df = pd.read_csv('data.csv', index_col='ID animals', dtype={'sucrose intake': 'float64', 'NOR index': 'float64'})
# per il momento lavoro sul caso binario 'no stress' ed 'CMS (stress cronico)'
df['target'] = pd.factorize(df['target'].str.split(" - ").str[0])[0]
df

# Ottengo le coppie (X, y)
X = df.drop(columns=['target'])
y = df['target']

# +
# Boxplot per ogni attributo del dataset
plt.figure(figsize=(12, 6))
plt.boxplot(X.values, tick_labels=X.columns, vert=False)

plt.title('Boxplot per ogni attributo')
plt.show()

# +
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
    """
    Genera tutte le combinazioni possibili di iperparametri basate su una griglia fornita.
    
    Parameters:
    ----------
    grid : dict
        Griglia degli iperparametri. Ogni coppia k-v è del tipo `nome_iperparametro : lista_possibili_valori`.
    
    Returns:
    -------
    list of dict
        Tutte le possibili configurazioni per gli iperparametri.
    
    Examples:
    --------
    >>> grid = {'n_neighbors': [1, 3], 'metric': ["nan_euclidean", "manhattan"]}
    >>> make_hp_configurations(grid)
    [{'n_neighbors': 1, 'metric': "nan_euclidean"},
     {'n_neighbors': 1, 'metric': "manhattan"},
     {'n_neighbors': 3, 'metric': "nan_euclidean"},
     {'n_neighbors': 3, 'metric': "manhattan"}]
    """
    return [{n: v for n, v in zip(grid. keys(), t)} for t in it.product(*grid.values())]


def fit_estimator(X, y, estimator, hp_conf):
    """
    Configura e addestra un modello predittivo con gli iperparametri forniti.

    Parameters:
    ----------
    X : array-like
        Osservazioni del dataset.
        
    y : array-like
        Etichette del dataset.
    
    estimator : oggetto estimator
        Modello predittivo che implementa i metodi `set_params` e `fit`.
    
    hp_conf : dict
        Dizionario contenente i nomi degli iperparametri e i rispettivi valori.
    
    Returns:
    --------
    Nessuno
        La funzione addestra l'estimatore in loco.
    
    Esempio:
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> X = [[1, 2], [3, 4], [5, 6]]
    >>> y = [0, 1, 0]
    >>> estimator = RandomForestClassifier()
    >>> hp_conf = {'n_estimators': 100, 'max_depth': 5}
    >>> fit_estimator(X, y, estimator, hp_conf)
    """
    estimator.set_params(**hp_conf)
    estimator.fit(X, y)


def get_score(X_test, y_test, estimator, scorer): 
    """
    Calcola il punteggio del modello predittivo usando il metodo di scoring specificato.
    
    Parameters:
    ----------
    X_test : array-like
        Osservazioni del dataset da classificare.
        
    y_test : array-like
        Etichette del dataset con sui confrontare le predizioni.
        
    estimator : estimator object
        Modello predittivo che implementa la funzione `predict()`.
        
    scorer : callable
        Funzione di scoring che, date le etichette e le predizioni, restituisce un punteggio.
    
    Returns:
    -------
    float
        Il punteggio calcolato in base alle previsioni fatte dal modello sui dati di test e
        valutato con la funzione di scoring.

    Example:
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.metrics import accuracy_score
    >>> X_train = [[1, 2], [3, 4], [5, 6], [7, 8]]
    >>> y_train = [0, 1, 0, 1]
    >>> X_test = [[9, 10], [11, 12]]
    >>> y_test = [1, 0]
    >>> estimator = RandomForestClassifier()
    >>> estimator.fit(X_train, y_train)
    >>> score = get_score(X_test, y_test, estimator, accuracy_score)
    >>> print(score)
    """
    
    return scorer(y_test, estimator.predict(X_test))


def check_best(minimize, score, best_score):
    """
    Verifica se il punteggio corrente è migliore rispetto al miglior punteggio finora, 
    in base alla strategia di minimizzazione o massimizzazione.
    
    Parameters:
    ----------
    minimize : bool
        Indica se la strategia di ottimizzazione è di minimizzazione (`True`) o di massimizzazione (`False`).
        
    score : float
        Punteggio corrente che si vuole confrontare con il miglior punteggio.
        
    best_score : float
        Miglior punteggio trovato finora.
    
    Returns:
    -------
    bool
        `True` se il punteggio corrente è migliore del miglior punteggio in base alla strategia 
        di ottimizzazione; `False` altrimenti.
    
    Examples:
    --------
    >>> check_best(True, 0.2, 0.5)
    True
    >>> check_best(False, 0.7, 0.5)
    True
    >>> check_best(True, 0.6, 0.5)
    False
    """
    
    return (minimize and score < best_score) or (not minimize and score > best_score)


def learn(X, y, estimator, param_grid, outer_split_method, inner_split_method,
            val_scorer=metrics.root_mean_squared_error, minimize_val_scorer=True, 
            test_scorer=metrics.root_mean_squared_error, minimize_test_scorer=True):
    """
    Addestra un modello predittivo e ottimizza i suoi iperparametri usando una procedura generica 
    di divisione dei dati (es: cross-validation, hold-out).
    
    Parameters:
    ----------
    X : array-like
        Osservazioni del dataset.
        
    y : array-like
        Etichette del dataset.
        
    estimator : estimator object
        Modello predittivo che implementa i metodi `fit()`, set_params() e `predict()`.
        
    param_grid : dict
        Griglia degli iperparametri. Ogni coppia k-v è del tipo `nome_iperparametro : lista_possibili_valori`.
        
    outer_split_method : object
        Oggetto dotato del metodo `split()`, che genera la suddivisione esterna 
        dei dati in sottoinsiemi di training/validation e test.
    
    inner_split_method : object
        Oggetto dotato del metodo `split()`, che genera la suddivisione interna 
        dei dati in sottoinsiemi di training e validation.
        
    val_scorer : callable, optional, default=metrics.root_mean_squared_error
        Funzione di scoring per valutare le prestazioni del modello sul validation set.
        
    minimize_val_scorer : bool, optional, default=True
        Se `True`, la funzione cerca di minimizzare la metrica calcolata `val_scorer`. 
        Se `False`, cerca di massimizzarlo.
        
    test_scorer : callable, optional, default=metrics.root_mean_squared_error
        Funzione di scoring per valutare le prestazioni del modello sul test set.
        
    minimize_test_scorer : bool, optional, default=True
        Se `True`, la funzione cerca di minimizzare la metrica calcolata da `test_scorer`. 
        Se `False`, cerca di massimizzarlo.
    
    Returns:
    -------
    estimator object
        Il modello addestrato sull'intero dataset con la configurazione di iperparametri ottimale.
        
    float
        Lo score finale del modello.
    """
    
    outer_scores = []

    best_score = np.inf if minimize_val_scorer else -np.inf
    best_conf = None

    for trainval_index, test_index in outer_split_method.split(X, y):
        # Ad ogni iterazione corrisponde una suddivisione diversa in trainval e test
        
        X_trainval, X_test = X[trainval_index], X[test_index]
        y_trainval, y_test = y[trainval_index], y[test_index]
        
        best_inner_score = np.inf if minimize_test_scorer else -np.inf
        best_inner_conf = None
        
        for hp_conf in make_hp_configurations(param_grid):
            # Ad ogni iterazione corrisponde una configurazione diversa degli iperparametri 
            conf_scores = []
            
            for train_index, val_index in inner_split_method.split(X_trainval, y_trainval):
                # Ad ogni iterazione corrisponde una suddivisione diversa in train e val
                
                X_train, X_val = X_trainval[train_index], X_trainval[val_index]
                y_train, y_val = y_trainval[train_index], y_trainval[val_index]

                fit_estimator(X_train, y_train, estimator, hp_conf)
                conf_scores.append(get_score(X_val, y_val, estimator, val_scorer))
            
            conf_score = np.mean(conf_scores)

            if check_best(minimize_val_scorer, conf_score, best_inner_score):
                best_inner_score, best_inner_conf = conf_score, hp_conf
        
        fit_estimator(X_trainval, y_trainval, estimator, best_inner_conf)
        outer_score = get_score(X_test, y_test, estimator, test_scorer)
        outer_scores.append(outer_score)

        if check_best(minimize_test_scorer, outer_score, best_score):
            best_score, best_conf = outer_score, best_inner_conf

    fit_estimator(X, y, estimator, best_conf)
    return estimator, np.mean(outer_scores)


# +
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

param_grid = {
    'n_neighbors': [k for k in range(1, 12, 2)], # capire quale massimo usare
    'metric': #[metric for metric in pairwise.distance_metrics()] # aclune metriche non vanno bene
    ['cityblock', 'cosine', 'euclidean', 'l2', 'l1', 'manhattan', 'nan_euclidean']
}
model = KNeighborsClassifier()

learn(X.values, y.values, model, param_grid, 
      StratifiedKFold(n_splits=4, shuffle=True, random_state=42), 
      StratifiedKFold(n_splits=3, shuffle=True, random_state=42), 
      val_scorer=metrics.accuracy_score, minimize_val_scorer=False,
      test_scorer=metrics.accuracy_score, minimize_test_scorer=False)
# -




