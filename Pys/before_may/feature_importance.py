import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from removed_pys.visualize_sgm_with_accuracies import shifted_geometric_mean

def importance_bar(df, title):
    features = df['Feature']
    for col in df.columns[1:]:
        values = df[col]

        # Create the bar plot
        plt.figure(figsize=(8, 6))
        #color bars according to the branching rule they belong to
        colors = []
        for feature in features:
            if 'Mixed' in feature:
                colors.append('magenta')
            elif 'Int' in feature:
                colors.append('green')
            else:
                colors.append('orange')

        plt.bar(features, values, color=colors, alpha=0.7, edgecolor='black')
        #remove x ticks
        # Remove x-axis ticks
        plt.xticks([])
        # Add labels and title
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        #plt.title(title, fontsize=12)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', markersize=10, label='Mixed'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Int'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Both')]

        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        # Show the plot
        plt.show()
        plt.close()

def feature_importance_linreg(features, model, scaling, imputation):
    importance_df = pd.DataFrame({'Feature': features, 'Importance': model.coef_})
    #importance_bar(importance_df, "Feature Importance: LinReg imputed by "+imputation+" and scaled by "+scaling)
    return importance_df

def feature_importance_forrest(features, model, scaling, imputation):
    importance_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
    #importance_bar(importance_df, "Feature Importance: ForrestReg imputed by " + imputation + " and scaled by " + scaling)
    return importance_df

def bar(values, bar_names, title):
    plt.bar(bar_names, values, align='center')
    #plt.title(title)
    plt.xticks(range(len(bar_names)), bar_names)
    plt.show()
    plt.close()

def assign_points(df, title:str):
    """
    Assign points to each feature based on importance scores.
    Most important feature (highest score) gets 0 points,
    second most important gets 1, and so on until the least important gets 17 points.

    Parameters:
    df (pd.DataFrame): A DataFrame where the first column contains feature names
                       and the other columns contain importance scores.

    Returns:
    pd.DataFrame: A DataFrame with the same structure, but with points instead of scores.
    """
    # Copy the original DataFrame to avoid modifying it
    df_points = df.copy()

    # Iterate over all columns except the first one (Feature names)
    feature_col = df.columns[0]  # First column is assumed to contain feature names

    # Rank all importance score columns (absolute values), then subtract 1 to start from 0
    rankings = df.iloc[:, 1:].abs().rank(ascending=False, method="min") - 1

    # Compute total points across all columns
    df[title] = rankings.sum(axis=1).astype(int)

    # Return DataFrame with only features and their total points
    return df[[feature_col, title]]

def create_importance_score_df(linear_df, forest_df):
    lin_scores = assign_points(linear_df, 'Linear Scores')
    forest_scores = assign_points(forest_df, 'Forest Scores')
    combined_df = lin_scores.copy()
    combined_df['Forest Scores'] = forest_scores['Forest Scores'].copy()
    combined_df['Total Scores'] = combined_df['Linear Scores'] + combined_df['Forest Scores']
    combined_df.sort_values('Total Scores', ascending=True, inplace=True)

    return combined_df

def plot_sgm_feature_importance(df, title):
    shift = -(df.iloc[:, 1:].min().min())+1
    df["ShiftedGeometricMean"] = df.iloc[:, 1:].apply(lambda row: shifted_geometric_mean(row, shift), axis=1)
    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(df['Feature'], df['ShiftedGeometricMean'], color='skyblue')

    # Add labels and title
    plt.ylabel('SGM Feature Importance')
    #plt.title(title)
    plt.xticks([])
    # plt.ylim([0.8, 1.06])
    # Show the plot
    plt.tight_layout()
    plt.show()
