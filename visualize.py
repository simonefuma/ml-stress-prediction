import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import RANDOM_STATE
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans

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


def show_cluster_plot(k, X, y, y_text, colors, title):
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE)
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

    custom_lines = get_custom_lines(y_text, colors)
    
    custom_lines.append(plt.Line2D([0], [0], 
                               marker='.', 
                               color='w', 
                               label='Centroids', 
                               markerfacecolor='k', 
                               markersize=10))
    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=150)

    plt.scatter(centroids[:, 0], centroids[:, 1], c='k', marker='.', edgecolor='k', s=150)
    plt.legend(handles=custom_lines)
    plt.title(title)
    plt.show()


def show_svc_decision_boundary(X, y, y_text, model, colors, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=colors, s=50, edgecolor="k")
    
    x_min, x_max = X[:, 0].min() - 10, X[:, 0].max() + 10
    y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    
    if len(np.unique(y)) == 2:
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=[0], linestyles='-', colors='k')
    else:
        n_classes = Z.shape[1]
        Z = Z.reshape(xx.shape[0], xx.shape[1], n_classes)
        
        for i in range(n_classes):
            plt.contour(xx, yy, Z[:, :, i], levels=[0], linestyles='-', colors=colors[i])
    
    custom_lines = get_custom_lines(y_text, colors)
    
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(handles=custom_lines)
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