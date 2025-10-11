""""This Script outputs first of all some histograms to see the distribution of the
different features. Once for Mixed and once for Int"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler


def feature_histo(df, columns: list, number_bins=10):
    """
    Create histograms for specified columns in a DataFrame, focusing only on values in (0, 1).
    Args:
        df: The DataFrame containing data.
        columns: List of column names to plot histograms for.
        number_bins: Number of bins for the histograms.
    """
    # Create a figure with subplots for each column dynamically
    fig, axs = plt.subplots(len(columns), 1, figsize=(8, 4 * len(columns)))  # n rows, 1 column

    # If there's only one column, axs is not an array, so we handle it separately
    if len(columns) == 1:
        axs = [axs]

    for i, col in enumerate(columns):
        # Filter values in (0, 1)
        # if df.values.min().min()>=0 and df.values.max().max()<=1:
        #     filtered_data = df[col][(df[col] >= 0) & (df[col] <= 1)]
        # else:
        filtered_data = df[col]
        # Plot histogram with color distinction
        color = 'red' if 'Mixed' in col else ('magenta' if 'Int' in col else 'orange')
        axs[i].hist(filtered_data, bins=number_bins, color=color, alpha=1)#, label=f'Filtered ({len(filtered_data)} points)')
        axs[i].set_title(f'{col}')

    # Adjust layout
    plt.tight_layout()
    # Show the plots once all are created
    plt.show()
    # Close the plot to free up memory
    plt.close()

# Helper function to filter columns based on existence in X_scaled
def filter_existing_columns(column_list, df):
    return [col for col in column_list if col in df.columns]

def scale_by_hand(feature):
    if feature.isna().sum().sum() > 0:
        print('Watch out! NaNs present')
        feature = feature.replace(np.nan, 0)

    cols_log_plus_one = ['#MIP nodes Mixed', '#MIP nodes Int', 'Matrix Equality Constraints', 'Matrix Quadratic Elements',
                         'Presolve Columns Mixed', 'Presolve Columns Int', '#nodes in DAG Mixed', '#nodes in DAG Int',
                         'Presolve Global Entities Mixed', 'Presolve Global Entities Int', 'Matrix NLP Formula']

    log_then_root = ['#nonlinear violations at root Mixed', '#nonlinear violations at root Int']

    cols_sqrt = ['% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG) Mixed',
                 '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG) Int']

    cols_to_4throot_scale = ['#spatial branching entities fixed (at the root) Mixed',
                             '#spatial branching entities fixed (at the root) Int',
                             '#integer violations at root Mixed', '#integer violations at root Int',
                             '% vars in DAG integer (out of vars in DAG) Mixed',
                             '% vars in DAG integer (out of vars in DAG) Int','Avg ticks for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
                             'Avg ticks for solving strong branching LPs for integer branchings (not including infeasible ones) Int']

    eighth_root = ['Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
                  'Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Int']
    tenth_root = ['Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Mixed',
                  'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Int',
                  'Avg coefficient spread for convexification cuts Mixed', 'Avg coefficient spread for convexification cuts Int',
                  'Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
                  'Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Int']

    #cols_to_log_scale sind alle ganzzahlig=> >=1 also kein problem für logarithmus
    X_scaled=feature.astype(float)

    # Filter each list of columns to include only those present in X_scaled
    log_then_root = filter_existing_columns(log_then_root, X_scaled)
    cols_log_plus_one = filter_existing_columns(cols_log_plus_one, X_scaled)
    cols_sqrt = filter_existing_columns(cols_sqrt, X_scaled)
    cols_to_4throot_scale = filter_existing_columns(cols_to_4throot_scale, X_scaled)
    eighth_root = filter_existing_columns(eighth_root, X_scaled)
    tenth_root = filter_existing_columns(tenth_root, X_scaled)

    #adding 1 before logging results in 0 being zero and the rest greater than zero, because log(1)=0
    X_scaled.loc[:, cols_log_plus_one] = X_scaled.loc[:, cols_log_plus_one] + 1
    X_scaled.loc[:, cols_log_plus_one] = X_scaled.loc[:, cols_log_plus_one].map(lambda x: np.log(x) if x>10**(-6) else 0)

    X_scaled.loc[:, log_then_root] = X_scaled.loc[:, log_then_root].map(lambda x: np.log(x) if x > 10 ** (-6) else 0)
    cols_sqrt += log_then_root
    X_scaled.loc[:, cols_sqrt] = X_scaled.loc[:, cols_sqrt].map(np.sqrt)
    X_scaled.loc[:, cols_to_4throot_scale] = X_scaled.loc[:, cols_to_4throot_scale].map(lambda x: np.power(x, 0.25))
    X_scaled.loc[:, eighth_root] = X_scaled.loc[:, eighth_root].map(lambda x: np.power(x, 0.125))
    X_scaled.loc[:, tenth_root] = X_scaled.loc[:, tenth_root].map(lambda x: np.power(x, 0.1))
    X_scaled = X_scaled.apply(lambda x: x / abs(x).max())

    return X_scaled

def scale_cmp_df(cmp_feat_df):
    log_scale = ['Matrix Equality Constraints', 'Matrix Quadratic Elements', 'Matrix NLP Formula']
    cmp_cols = [col_name for col_name in cmp_feat_df.columns if 'Cmp' in col_name]
    # Initialize PowerTransformer
    pt = PowerTransformer(method='yeo-johnson')
    standard_scaler = StandardScaler()
    # Apply transformation
    # Fit and transform only numerical columns
    # cmp_feat_df.loc[:, cmp_cols] = pt.fit_transform(cmp_feat_df.loc[:, cmp_cols])
    cmp_feat_df.loc[:, cmp_cols] = standard_scaler.fit_transform(cmp_feat_df.loc[:, cmp_cols])
    # Create a new DataFrame with transformed data
    cmp_feat_df_transformed = pd.DataFrame(cmp_feat_df, columns=cmp_feat_df.columns)

    return cmp_feat_df_transformed

# data = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/ZwischenPräsi_Januar/Data/Features/lean_data_05_01.xlsx').drop(columns='Matrix Name')
# features = data.drop(columns='Cmp Final solution time (cumulative)')
# label = data['Cmp Final solution time (cumulative)']

# scaled_data_3rd_root = data.iloc[:,1:].map(lambda x: np.cbrt(x))
# scaled_data_5th_root = data.iloc[:,1:].map(lambda x: np.power(x, 0.2))
# feature_histo(scaled_data_5th_root, data.columns[1:], number_bins=10)

# cmp_cols = [col for col in data.columns if 'Cmp' in col]
# nonneg_cols = [col for col in data.columns if 'Cmp' not in col]
#
# def yeo_johnson(df, histo=False):
#     # yeo transformer
#     pt = PowerTransformer(method='yeo-johnson')
#     yeo_data = pt.fit_transform(data)
#     yeo_df = pd.DataFrame(yeo_data, columns=data.columns)
#     yeo_df_normalized = yeo_df.apply(lambda x: x / abs(x).max())
#     if histo:
#         feature_histo(yeo_df_normalized, yeo_df.columns)
#
# def box_cox(df, histo= False):
#     make_strictly_pos_df = df+1
#
#     b_c = PowerTransformer(method='box-cox')
#     box_cox_data = b_c.fit_transform(make_strictly_pos_df)
#     box_cox_df = pd.DataFrame(box_cox_data, columns=nonneg_cols)
#     box_cox_df = box_cox_df.apply(lambda x: x / abs(x).max())
#     if histo:
#         feature_histo(box_cox_df, box_cox_df.columns)

# yeo_johnson(data[nonneg_cols], histo=True)
# box_cox(data[nonneg_cols], histo=True)






"""Detect outlier"""
# from sklearn.ensemble import IsolationForest
#
# for col in data.columns:
#     # Fit the model
#     iso = IsolationForest(contamination=0.1)  # Set contamination rate
#     data['anomaly'] = iso.fit_predict(data[[col]])
#     # print((data['anomaly'] == -1).sum())
#     # Identify outliers (label = -1)
#     outliers = data[data['anomaly'] == -1]
#     # print("Outliers using Isolation Forest:")
#     # print(outliers)
#     # Boxplot
#     plt.boxplot(data[col])
#     #plt.title(f"Boxplot of {col}")
#     plt.show()