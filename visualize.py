import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import RANDOM_STATE
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.inspection import DecisionBoundaryDisplay

def get_custom_lines(y_text, colors):
    return [plt.Line2D([0], [0], marker='o', color='w', label=y_text[i], markerfacecolor=colors[i], markersize=10) for i in range(len(y_text))]

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

    custom_lines = get_custom_lines(y_text, colors)

    plt.legend(handles=custom_lines)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(handles=custom_lines)
    plt.show()


def show_3D_scatter_plot(X, y, y_text, colors, title):
    fig, axs = plt.subplots(1, 3, figsize=(18, 8), subplot_kw={'projection': '3d'})

    view_angles = [(45, 45), (30, 60), (90, 0)]
    
    for i, ax in enumerate(axs):
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=ListedColormap(colors), s=50, edgecolor='k', alpha=0.6)
        
        ax.view_init(elev=view_angles[i][0], azim=view_angles[i][1])
        
        ax.set_xlabel('Cmp 1')
        ax.set_ylabel('Cmp 2')
        ax.set_zlabel('Cmp 3')

        ax.set_title(f'({view_angles[i][0]}, {view_angles[i][1]})')

    custom_lines = get_custom_lines(y_text, colors)
    fig.suptitle(title, fontsize=16)
    fig.legend(handles=custom_lines, loc='upper left')
    plt.tight_layout()
    plt.show()


def show_cluster_plot(k, X, y, y_text, colors, title):
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE)
    predicts = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_;

    # Definisci i limiti del grafico
    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000))
    
    # Predici i cluster su tutto il piano e ridimensiona per visualizzare
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Disegna solo le linee di confine dei cluster
    plt.figure(figsize=(8, 6))
    plt.contour(xx, yy, Z, colors='black', linewidths=0.5)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    custom_lines = get_custom_lines(y_text, colors)
    
    custom_lines.append(plt.Line2D([0], [0], 
                               marker='.', 
                               color='w', 
                               label='Centroids', 
                               markerfacecolor='k', 
                               markersize=10))
    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(colors), edgecolor='k', s=150)

    plt.scatter(centroids[:, 0], centroids[:, 1], c='k', marker='.', edgecolor='k', s=150)
    plt.legend(handles=custom_lines)
    plt.title(title)
    plt.show()


def show_svc_decision_boundary(X, y, y_text, model, colors, title):
    plt.figure(figsize=(8, 6))

    display = DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        grid_resolution=1000,
        xlabel='Component 1',
        ylabel='Component 2',
        response_method="predict",
        cmap=ListedColormap(colors),
        alpha=0.7,
    )

    display.plot(ax=plt.gca(), cmap=plt.cm.coolwarm, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(colors), s=50, edgecolor='k', label=y_text)
    custom_lines = get_custom_lines(y_text, colors)

    plt.title(title)
    plt.legend(handles=custom_lines)

    plt.show()


def show_svc_decision_boundary_3D(X, y, y_text, model, colors, title):
    # 1. Creare una griglia tridimensionale di punti per tracciare il confine decisionale
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

    xx, yy, zz = np.meshgrid(
        np.linspace(x_min, x_max, 30),
        np.linspace(y_min, y_max, 30),
        np.linspace(z_min, z_max, 30)
    )

    # 2. Calcolare i valori di decisione per ogni punto nella griglia
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    decision_values = model.decision_function(grid_points)
    decision_values = decision_values.reshape(xx.shape)

    # 3. Creare il grafico 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")

    # 4. Visualizzare i punti dati originali
    for i, label in enumerate(np.unique(y)):
        ax.scatter(
            X[y == label, 0], X[y == label, 1], X[y == label, 2],
            color=colors[i], label=y_text[label], s=30
        )

    # 5. Aggiungere il confine decisionale
    # Il confine decisionale è il livello 0 della superficie
    ax.contourf(xx, yy, zz, decision_values, levels=[-1, 0, 1], alpha=0.3, colors=colors[:2])

    # 6. Mostrare il grafico con la legenda
    ax.legend()
    plt.show()


def show_cluster_table(k, X, y, y_text, title):
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE)
    predicts = kmeans.fit_predict(X)

    results_df = pd.DataFrame({
        title: y_text[y],
        'cluster': predicts
    })

    cluster_table = pd.crosstab(results_df['cluster'], results_df[title])

    display(cluster_table)


def display_table(learned_model):
    result_df = pd.DataFrame(learned_model['result'])
    result_df[['avg', 'std']] = result_df[['avg', 'std']].round(3)
    
    result_df.set_index('scorer_name', inplace=True)
    print('\n', learned_model['model'], '\n', learned_model['model_name'], '\n', '-' * 27)
    display(result_df)