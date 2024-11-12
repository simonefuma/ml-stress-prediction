import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import RANDOM_STATE
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import plot_tree as sklearn_plot_tree

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

    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000))
    
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
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


def show_svc_decision_boundary(X, y, y_text, models, colors):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, model in enumerate(models):
        ax = axes[i]

        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = model['model'].predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(colors))
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(colors), s=50, edgecolor='k', label=y_text)
        ax.set_title(model['model_name'])
        
        custom_lines = get_custom_lines(y_text, colors)
        ax.legend(handles=custom_lines)

    plt.tight_layout()
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


def display_table(learned_models):
    scorers = [result['scorer_name'] for result in learned_models[0]['result']]
    columns = pd.MultiIndex.from_product([scorers, ['avg', 'std']], names=['Scorer', 'Metric'])
    
    model_names = [model['model_name'] for model in learned_models]
    df = pd.DataFrame(index=model_names, columns=columns)

    for learned_model in learned_models:
        model_name = learned_model['model_name']
        for result in learned_model['result']:
            scorer = result['scorer_name']
            df.loc[model_name, (scorer, 'avg')] = round(result['avg'], 3)
            df.loc[model_name, (scorer, 'std')] = round(result['std'], 3)

    display(df)
            

def plot_tree(columns, y_text, learned_model, title):
    plt.figure(figsize=(12, 8))
    sklearn_plot_tree(learned_model, feature_names=columns, class_names=y_text, filled=True)
    plt.title(title)
    plt.show()