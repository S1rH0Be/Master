from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def importance_bar(df, title):
    features = df['Feature']
    values = df['Importance']

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
    plt.title(title, fontsize=12)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', markersize=10, label='Mixed'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Int'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Both')]

    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    # Show the plot
    #plt.show()

def feature_importance_linreg(features, model, scaling, imputation):
    importance_df = pd.DataFrame({'Feature': features, 'Importance': model.coef_})
    #importance_bar(importance_df, "Feature Importance: LinReg imputed by "+imputation+" and scaled by "+scaling)
    return importance_df

def feature_importance_forrest(features, model, scaling, imputation):
    importance_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
    #importance_bar(importance_df, "Feature Importance: ForrestReg imputed by " + imputation + " and scaled by " + scaling)
    return importance_df


