import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer

from may_regression import shifted_geometric_mean

def log_df(df, cols_to_log):
    if cols_to_log is None:
        cols_to_log = df.columns
    for col in cols_to_log:
        column = df[col]
        value_pos = column[column >= 0]
        value_neg = column[column < 0]
        value_pos_log = np.log1p(value_pos)/np.log(10)
        value_neg_log = (np.log1p(abs(value_neg))/np.log(10)) * -1
        column = pd.concat([value_pos_log, value_neg_log]).sort_index()
        df.loc[:,col] = column
    return df

def get_top_x(impo_rank:pd.DataFrame, sort_by:str, x:int):
    impo_rank.sort_values(by=sort_by, ascending=True, inplace=True)
    top_x = impo_rank['Feature'].head(x)
    return top_x

def replace_and_impute(df, imputation_rule):
    df = df.replace([-1,-1.0], np.nan)
    imputer = SimpleImputer(missing_values=np.nan, strategy=imputation_rule)
    imputer.fit(df)
    imputed_array = imputer.transform(df)
    df_imputed = pd.DataFrame(imputed_array, columns=df.columns, index=df.index)
    return df_imputed

def create_quantile_scaled_features_zero_one(df):
    df = replace_and_impute(df, 'mean')
    transformer = QuantileTransformer(output_distribution='normal', n_quantiles=int(len(df) * 0.8))
    transformed = transformer.fit_transform(df)
    transformed = transformed.transpose()
    df = df.astype(float)
    for i, col in enumerate(df.columns):
        df.loc[:, col] = (transformed[i] / abs(transformed[i]).max())
    return df

def quantile_scaler(df):
    df = replace_and_impute(df, 'median')
    transformer = QuantileTransformer(output_distribution='normal', n_quantiles=int(len(df) * 0.8))
    transformed = transformer.fit_transform(df)
    transformed = transformed.transpose()
    df = df.astype(float)
    for i, col in enumerate(df.columns):
        df.loc[:, col] = (transformed[i])
    return df

def create_log_scaled_df(df, imputer):
    df = replace_and_impute(df, imputer)
    df_log = log_df(df)
    # make it zero one
    for i, col in enumerate(df_log.columns):
        df_log.loc[:, col] = (df_log[col] / abs(df_log[col]).max())
    return df_log

def plot_histo(data, color, number_bins, title):

    plt.hist(data, bins=number_bins, color=color, alpha=1)
    plt.title(title)

    # Adjust layout
    plt.tight_layout()
    # Show the plots once all are created
    plt.show()
    # Close the plot to free up memory
    plt.close()

def feature_histo(scip_df, fico_df, columns: list, number_bins=10):
    """
    Create histograms for specified columns in a DataFrame.
    """
    for i, col in enumerate(columns):
        color = ['violet', 'purple']
        if scip_df is not None:
            scip_data = scip_df[col]
            plot_histo(scip_data, color[0], number_bins, f'{col} SCIP scaled')
        if fico_df is not None:
            fico_data = fico_df[col]
            plot_histo(fico_data, color[1], number_bins, f'{col} FICO scaled')

def comp_scip_and_fico(scip_feature, fico_feature):
    scip_feat_names = [feat_name+' SCIP' for feat_name in scip_feature.columns]
    fico_feat_names = [feat_name + ' FICO' for feat_name in fico_feature.columns]
    all_features = sorted(scip_feat_names+fico_feat_names)
    parameters = ['Min', 'Max', 'Median', 'Mean', 'SGM']

    comp_df = pd.DataFrame(index=parameters, columns=all_features, data=0.0)
    for col in fico_feature.columns:
        fico_feature.loc[:, col] = fico_feature[col].replace(-1.0, np.nan)
        mean = fico_feature[col].mean()
        fico_feature.loc[:, col] = fico_feature[col].replace(np.nan, mean)

    for col in scip_feature.columns:
        scip_feature.loc[:, col] = scip_feature[col].replace(-1.0, np.nan)
        mean = scip_feature[col].mean()
        scip_feature.loc[:, col] = scip_feature[col].replace(np.nan, mean)

    for col in all_features:
        col_name_fico = col.replace(' FICO', '')
        col_name_scip = col.replace(' SCIP', '')

        if col_name_fico in fico_feature.columns:
            comp_df.loc['Min', col] = fico_feature[col_name_fico].min()
            comp_df.loc['Max', col] = fico_feature[col_name_fico].max()
            comp_df.loc['Mean', col] = fico_feature[col_name_fico].mean()
            comp_df.loc['Median', col] = fico_feature[col_name_fico].median()
            comp_df.loc['SGM', col] = shifted_geometric_mean(fico_feature[col_name_fico], 2)

        if col_name_scip in scip_feature.columns:
            comp_df.loc['Min', col] = scip_feature[col_name_scip].min()
            comp_df.loc['Max', col] = scip_feature[col_name_scip].max()
            comp_df.loc['Mean', col] = scip_feature[col_name_scip].mean()
            comp_df.loc['Median', col] = scip_feature[col_name_scip].median()
            comp_df.loc['SGM', col] = shifted_geometric_mean(scip_feature[col_name_scip], 2)


    comp_df.to_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/comp_fico_and_scip_feats.csv')

def plot_top_x_distribution(features:pd.DataFrame, impo_rank:pd.DataFrame, sort_by:str, number_of_feats:int, scip_or_fico:str):
    # TODO: Add to decide if scip or fico
    top_feats = get_top_x(impo_rank, sort_by, number_of_feats).values.tolist()
    feature_histo(None, features, top_feats, )

# scip_feats = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/cleaned_scip/scip_default_clean_feats.csv')
# fico_feats = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/918/base_feats_no_cmp_918_24_01.xlsx', index_col=0)
# common_cols = list(set(fico_feats.columns).intersection(scip_feats.columns))

# rel_quant_log_fico_feats = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/June/Bases/FICO/Scaled/relative_logged_quantile_fico_feats.csv',
#                                        index_col=0)
# impo_ranking = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/June/Iteration2/RelativeLoggedQuantileFico/ScaledLabel/Importance/fico_importance_ranking.csv')
# plot_top_x_distribution(rel_quant_log_fico_feats, impo_ranking, sort_by='Linear Score', number_of_feats=5)
# plot_top_x_distribution(rel_quant_log_fico_feats, impo_ranking, sort_by='Forest Score', number_of_feats=5)
# plot_top_x_distribution(rel_quant_log_fico_feats, impo_ranking, sort_by='Combined', number_of_feats=5)

# Scaled features
# scip_scaled = create_quantile_scaled_features_zero_one(scip_feats)
# fico_scaled = create_quantile_scaled_features_zero_one(fico_feats)
#
# scip_scaled.to_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/scaled_feats/scip_quantile_feats.csv',
#                    index=False)
# fico_scaled.to_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/scaled_feats/fico_quantile_feats.csv',
#                    index=False)

# feature_histo(scip_scaled, fico_scaled, common_cols)


# scip_logged = create_log_scaled_df(scip_feats, 'mean')
# scip_logged.to_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/scaled_feats/scip_logged_feats.csv',
#                    index=False)
# fico_logged = create_log_scaled_df(fico_feats, 'mean')
# fico_logged.to_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/scaled_feats/fico_logged_feats.csv',
#                    index=False)

# feature_histo(scip_logged, fico_logged, common_cols)

# fico_relative = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/June/Bases/FICO/Scaled/relative_fico_feats.csv')
fico_relative_logged_quantile = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/June/Bases/FICO/Scaled/relative_logged_quantile_fico_feats.csv',
                                   index_col=0)
log_cols = ['Avg work for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
'Avg work for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Mixed',
'Avg coefficient spread for convexification cuts Mixed']


# feature_histo(None, fico_relative_logged_quantile, fico_relative_logged_quantile.columns)