import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import QuantileTransformer

from may_regression import shifted_geometric_mean

def create_quantile_scaled_features_zero_one(df):
    transformer = QuantileTransformer(output_distribution='normal', n_quantiles=int(len(df) * 0.8))
    transformed = transformer.fit_transform(df)
    transformed = transformed.transpose()
    df = df.astype(float)
    for i, col in enumerate(df.columns):
        df.loc[:, col] = (transformed[i] / abs(transformed[i]).max())
    return df

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
        scip_data = scip_df[col]
        fico_data = fico_df[col]
        # Plot histogram with color distinction
        color =  ['violet', 'purple']
        plot_histo(scip_data, color[0], number_bins, f'{col} SCIP scaled')
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


scip_feats = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/cleaned_scip/scip_default_clean_feats.csv')
fico_feats = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/918/base_feats_no_cmp_918_24_01.xlsx', index_col=0)
common_cols = list(set(fico_feats.columns).intersection(scip_feats.columns))



# Scaled features
scip_scaled = create_quantile_scaled_features_zero_one(scip_feats)
fico_scaled = create_quantile_scaled_features_zero_one(fico_feats)

scip_scaled.to_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/scaled_feats/scip_quantile_feats.csv')
fico_scaled.to_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/scaled_feats/fico_quantile_feats.csv')



# feature_histo(scip_scaled, fico_scaled, common_cols)
