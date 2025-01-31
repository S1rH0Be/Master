from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tensorflow.python.ops.numpy_ops import vsplit

from visualize_sgm_with_accuracies import shifted_geometric_mean

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
        plt.title(title, fontsize=12)

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
    plt.title(title)
    plt.xticks(range(len(bar_names)), bar_names)
    plt.show()
    plt.close()

def assign_points(df):
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
    for col in df.columns[1:]:
        # Rank the features: most important (highest score) gets 0 points
        rankings = df[col].rank(ascending=False, method="min") - 1
        df_points[col] = rankings.astype(int)
        # Calculate total points for each feature
    df_points["Total Points"] = df_points.iloc[:, 1:].sum(axis=1)

    # Return a new DataFrame with feature names and their total points
    result_df = df_points[["Feature", "Total Points"]]
    return result_df

# linimp = pd.read_excel(f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/linear_importances_yeo_20_01.xlsx').drop('Feature', axis=1)
# forimp = pd.read_excel(f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/forest_importances_20_01.xlsx').drop('Feature', axis=1)
#
# # Calculate row-wise variance
# lin_row_variance = linimp.var(axis=1)
# lin_noscaling_variance = linimp.iloc[:,0::3].var(axis=1)
# lin_byhand_variance = linimp.iloc[:,1::3].var(axis=1)
# lin_yeojohn_variance = linimp.iloc[:,2::3].var(axis=1)
#
# bar(lin_row_variance.values, [str(i) for i in range(len(lin_row_variance.values))], 'Linear Variance Total')
# bar(lin_noscaling_variance.values, [str(i) for i in range(len(lin_row_variance.values))], 'Linear Variance NoScaling')
# bar(lin_byhand_variance.values, [str(i) for i in range(len(lin_row_variance.values))], 'Linear Variance byHand')
# bar(lin_yeojohn_variance.values, [str(i) for i in range(len(lin_row_variance.values))], 'Linear Variance Yeo-Johnson')



#
# # Calculate row-wise minimum
# lin_row_min = linimp.min(axis=1)
# # Calculate row-wise maximum
# lin_row_max = linimp.max(axis=1)
# # Calculate row-wise mean
# lin_row_mean = linimp.mean(axis=1)
#
#
#
# # Calculate row-wise variance
# for_row_variance = forimp.var(axis=1)
# # Calculate row-wise minimum
# for_row_min = forimp.min(axis=1)
# # Calculate row-wise maximum
# for_row_max = forimp.max(axis=1)
# # Calculate row-wise mean
# for_row_mean = forimp.mean(axis=1)
# lin_with_outlier_df = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/NoCmpFeats/Tester/Importance/Linear/unscaled_lin_impo_t18_below_100000000_hundred_seeds_28_01.xlsx')
# lin_df = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/NoCmpFeats/Tester/Importance/Linear/unscaled_lin_impo_t18_below_1000_hundred_seeds_28_01.xlsx')
# shift_outlier = -(lin_with_outlier_df.iloc[:, 1:].min().min())+1
# shift_no_outlier = -(lin_df.iloc[:, 1:].min().min())+1
# # shift = 0
# lin_df["ShiftedGeometricMean"] = lin_df.iloc[:, 1:].apply(
#     lambda row: shifted_geometric_mean(row, shift_no_outlier), axis=1)
#
# lin_with_outlier_df["ShiftedGeometricMean"] = lin_with_outlier_df.iloc[:, 1:].apply(
#     lambda row: shifted_geometric_mean(row, shift_outlier), axis=1)
#
#
# time_df = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/NoCmpFeats/base_data_24_01.xlsx')
# time_df = time_df[['Final solution time (cumulative) Int', 'Final solution time (cumulative) Mixed', 'Virtual Best']]
# values = [shifted_geometric_mean(time_df[column], 0.5) for column in time_df.columns]
# values_relative = [value/values[1] for value in values]
# names = ['Int', 'Mixed', 'Virtual Best']



def plot_sgm_feature_importance(df, title):
    shift = -(df.iloc[:, 1:].min().min())+1
    df["ShiftedGeometricMean"] = df.iloc[:, 1:].apply(lambda row: shifted_geometric_mean(row, shift), axis=1)
    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(df['Feature'], df['ShiftedGeometricMean'], color='skyblue')

    # Add labels and title
    plt.ylabel('SGM Feature Importance')
    plt.title(title)
    plt.xticks([])
    # plt.ylim([0.8, 1.06])
    # Show the plot
    plt.tight_layout()
    plt.show()

# lin_with_outlier_df[["Feature", "ShiftedGeometricMean"]].to_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/NoCmpFeats/Tester/Importance/Linear/sgm_linear_with_outlier_unscaled.xlsx', index=False)
# lin_df[["Feature", "ShiftedGeometricMean"]].to_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/NoCmpFeats/Tester/Importance/Linear/sgm_linear_no_outlier_unscaled.xlsx', index=False)
#





# forest_importance_score = assign_points(for_df)
# lin_importance = assign_points(lin_df)
# importance_df = pd.DataFrame({'Feature': forest_importance_score['Feature']})
# importance_df["Total Points Linear"] = lin_importance["Total Points"]
# importance_df["Total Points Forest"] = forest_importance_score["Total Points"]
# importance_df["Total Points"] = importance_df["Total Points Forest"]+importance_df["Total Points Linear"]
# importance_df = importance_df.sort_values(by='Total Points', ascending=True)


