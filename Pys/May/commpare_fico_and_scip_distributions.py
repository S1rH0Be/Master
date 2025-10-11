import pandas as pd
import seaborn as sns
import numpy as np
import re
import matplotlib.pyplot as plt

def get_common_cols(df1, df2):
    common_cols = list(set(df1.columns).intersection(df2.columns))
    return common_cols

def plot_column_distributions(df1, df2, labels=("FICO", "SCIP"), df2_repeat=3):
    """
    Plot overlaid distribution plots for each numerical column in df1 and df2.

    Parameters:
        df1 (pd.DataFrame): First DataFrame
        df2 (pd.DataFrame): Second DataFrame
        labels (tuple): Labels for the DataFrames in the legend
    """
    # Ensure both DataFrames have the same columns
    assert list(df1.columns) == list(df2.columns), "DataFrames must have the same columns"
    df2_repeated = pd.concat([df2] * df2_repeat, ignore_index=True)
    # Loop through each column
    for col in df1.select_dtypes(include='number').columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df1[col], label=labels[0], fill=True, common_norm=False, alpha=0.5, bins=10)
        sns.histplot(df2_repeated[col], label=labels[1], fill=True, common_norm=False, alpha=0.5, bins=10)
        #plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # for shared bins
    # for col in df1.select_dtypes(include='number').columns:
    #     # Combine both datasets for shared binning
    #     all_data = pd.concat([df1[col], df2_repeated[col]])
    #
    #     # Create consistent bin edges
    #     bin_edges = np.histogram_bin_edges(all_data.dropna(), bins=10)
    #
    #     # Plot with shared bins
    #     plt.figure(figsize=(8, 4))
    #     sns.histplot(df1[col], bins=bin_edges, label=labels[0], fill=True, common_norm=False, alpha=0.5)
    #     sns.histplot(df2_repeated[col], bins=bin_edges, label=labels[1], fill=True, common_norm=False, alpha=0.5)
    #     #plt.title(f'Distribution of {col}')
    #     plt.xlabel(col)
    #     plt.ylabel('Density')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

def plot_median_imputed():
    imputed_fico_feats = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/fico/median_imputed_fico_feats.csv')
    imputed_scip_feats = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/cleaned_scip/median_imputed_scip_feats.csv')

    c_c = get_common_cols(imputed_fico_feats, imputed_scip_feats)
    plot_column_distributions(imputed_fico_feats[c_c], imputed_scip_feats[c_c])

def plot_median_logged():
    imp_logged_fico = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/fico/median_logged_fico_feats.csv')
    imp_logged_scip = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/cleaned_scip/median_logged_scip_feats.csv')

    com_cols = get_common_cols(imp_logged_fico, imp_logged_scip)
    plot_column_distributions(imp_logged_fico[com_cols], imp_logged_scip[com_cols])

def plot_quantile():
    quantile_logged_fico = pd.read_csv(
        '/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/fico/median_imputed_quantile_logged_fico_feats.csv')
    quantile_logged_scip = pd.read_csv(
        '/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/cleaned_scip/median_imputed_quantile_logged_scip_feats.csv')

    com_cols = get_common_cols(quantile_logged_fico, quantile_logged_scip)
    plot_column_distributions(quantile_logged_fico[com_cols], quantile_logged_scip[com_cols])

def get_schnitt_di_effs():
    schnitt_df = pd.read_csv("/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/fico_schnitt.csv")
    schnitt_instances = list(set(schnitt_df['Matrix Name'].tolist()))
    schnitt_instances.remove('crudeoil_lee4_08')

    fico_df = pd.read_csv("/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/fico/fico_clean_data.csv")
    scip_df = pd.read_csv(
        "/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/cleaned_scip/scip_default_clean_data.csv")

    fico_indices = []
    names = []
    scip_indices = []

    for index, row in fico_df.iterrows():
        base_name = re.sub(r'_dup\d+$', '', str(row['Matrix Name']))
        if base_name in schnitt_instances:
            fico_indices.append(index)
            names.append(base_name)

    for index, row in scip_df.iterrows():
        base_name = re.sub(r'_dup\d+$', '', str(row['Matrix Name']))
        if base_name in schnitt_instances:
            scip_indices.append(index)
        else:
            if base_name in names:
                print(base_name)

    common_cols = get_common_cols(fico_df, scip_df)
    print(len(common_cols))
    fico_schnitt = fico_df.loc[fico_indices, common_cols]
    scip_schnitt = scip_df.loc[scip_indices, common_cols]


    plot_column_distributions(fico_schnitt, scip_schnitt, df2_repeat=1)


get_schnitt_di_effs()