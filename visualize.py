import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import RANDOM_STATE
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import plot_tree as sklearn_plot_tree

from matplotlib.lines import Line2D

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


def show_cumulative_explained_variance(explained_variance, title):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.axhline(y=0.9, color='r', linestyle='--')
    plt.title(title)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid()
    plt.show()


def show_scatter_plot(X, y, y_text, colors, title, gender):
    plt.figure(figsize=(8, 6))
    
    for i, (shape, label) in enumerate(zip(['o', '^'], ['Male', 'Female'])):
        gender_mask = (gender == i)
        plt.scatter(X[gender_mask, 0], X[gender_mask, 1], 
                    c=y[gender_mask], 
                    cmap=ListedColormap(colors), 
                    edgecolor='k', 
                    s=150, 
                    marker=shape, 
                    label=label)

    custom_lines = get_custom_lines(y_text, colors)

    plt.legend(handles=custom_lines + 
               [Line2D([0], [0], color='k', lw=0, marker='o', markersize=10, label='Male'),
                Line2D([0], [0], color='k', lw=0, marker='^', markersize=10, label='Female')])
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
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


def show_cluster_plot(k, X, y, y_text, colors, title, gender):
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE)
    predicts = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000))
    
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contour(xx, yy, Z, colors='black', linewidths=0.5)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    for i, (shape, label) in enumerate(zip(['o', '^'], ['Male', 'Female'])):
        gender_mask = (gender == i)
        plt.scatter(X[gender_mask, 0], X[gender_mask, 1], 
                    c=y[gender_mask], 
                    cmap=ListedColormap(colors), 
                    edgecolor='k', 
                    s=150, 
                    marker=shape, 
                    label=label)
    
    plt.scatter(centroids[:, 0], centroids[:, 1], c='k', marker='x', s=200, label='Centroids')

    custom_lines = get_custom_lines(y_text, colors)
    custom_lines.extend([
        Line2D([0], [0], color='k', lw=0, marker='o', markersize=10, label='Male'),
        Line2D([0], [0], color='k', lw=0, marker='^', markersize=10, label='Female'),
        Line2D([0], [0], color='k', lw=0, marker='x', markersize=10, label='Centroids')
    ])
    
    plt.legend(handles=custom_lines)
    plt.title(title)
    plt.show()


def show_svc_decision_boundary(X, y, y_text, models, colors, gender):
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6))
    
    for i, model in enumerate(models):
        ax = axes[i] if len(models) > 1 else axes

        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = model['model'].predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(colors))

        for j, (shape, label) in enumerate(zip(['o', '^'], ['Male', 'Female'])):
            gender_mask = (gender == j)
            ax.scatter(X[gender_mask, 0], X[gender_mask, 1], 
                       c=y[gender_mask], 
                       cmap=ListedColormap(colors), 
                       edgecolor='k', 
                       s=50, 
                       marker=shape, 
                       label=label)

        ax.set_title(model['model_name'])
        
        custom_lines = get_custom_lines(y_text, colors)
        custom_lines.extend([
            Line2D([0], [0], color='k', lw=0, marker='o', markersize=10, label='Male'),
            Line2D([0], [0], color='k', lw=0, marker='^', markersize=10, label='Female')
        ])
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

def print_table_latex(df):
    formatted_df = df.copy()
    for col in df.columns.levels[0]:
        col_max = df[(col, "avg")].max()
        formatted_df[(col, "avg")] = df[(col, "avg")].apply(
            lambda x: f"\\textbf{{{x:.3f}}}" if x == col_max else f"{x:.3f}"
        )
        col_min = df[(col, "std")].min()
        formatted_df[(col, "std")] = df[(col, "std")].apply(
            lambda x: f"\\textbf{{{x:.3f}}}" if x == col_min else f"{x:.3f}"
        )
    
    latex_code = formatted_df.to_latex(
        index=True,
        multirow=True,
        multicolumn_format="c",
        caption="Risultati dei modelli",
        label="tab:",
        escape=False
    )
    
    print(latex_code)

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
    print_table_latex(df)
    display(df)


def display_hyperparameters(model_name, param_grid, learned_model):
    columns = ['scaler'] + list(param_grid.keys())
    values = [learned_model.get_params()['scaler']] + [
        learned_model.get_params()['classifier'].get_params()[column] for column in columns[1:]
    ]
    df = pd.DataFrame([values], index=[model_name], columns=columns)
    display(df)
            

def plot_tree(columns, y_text, learned_model, title):
    plt.figure(figsize=(12, 12))
    sklearn_plot_tree(learned_model, feature_names=columns, class_names=y_text, filled=True)
    plt.title(title)
    plt.show()

    usage = ["x" if importance > 0 else "-" for importance in learned_model.feature_importances_]
    display(pd.DataFrame([usage + [learned_model.get_depth()]], columns=list(columns) + ["Depth"], index=[title]))

    latex_code = pd.DataFrame([usage + [learned_model.get_depth()]], columns=list(columns) + ["Depth"], index=[title]).to_latex()
    print(latex_code)


def plot_forest(columns, y_text, learned_model, title):
    usages = []
    indexs = []
    for i, tree in enumerate(learned_model.estimators_):
        plt.figure(figsize=(12, 12))
        sklearn_plot_tree(tree, feature_names=columns, class_names=y_text, filled=True)
        plt.title(f'{title}_{i}')
        plt.show()

        usages.append(["x" if importance > 0 else "-" for importance in tree.feature_importances_] + [tree.get_depth()])
        indexs.append(f'{title}_{i}')
    display(pd.DataFrame(usages, columns=list(columns) + ["Depth"], index=indexs))

    latex_code = pd.DataFrame(usages, columns=list(columns) + ["Depth"], index=indexs).to_latex()
    print(latex_code)