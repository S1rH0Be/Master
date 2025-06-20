import sys
import pandas as pd
import numpy as np
from typing import Union

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer


def replace_and_impute(df, imputation_rule):
    df = df.replace([-1,-1.0], np.nan)
    imputer = SimpleImputer(missing_values=np.nan, strategy=imputation_rule)
    imputer.fit(df)
    imputed_array = imputer.transform(df)
    df_imputed = pd.DataFrame(imputed_array, columns=df.columns, index=df.index).astype(float)
    return df_imputed

def get_log_cols(feature_df):
    """
    Log all features with min <10 and max>10 => gets everything in range [0,15]
    """
    log_columns = []
    for feature in feature_df.columns:
        if feature_df[feature].max() > 10:
            log_columns.append(feature)
    return log_columns

def log_col(column:pd.Series):
    value_pos = column[column >= 0]
    value_neg = column[column < 0]
    value_pos_log = np.log1p(value_pos)
    value_neg_log = np.log1p(abs(value_neg)) * -1
    column = pd.concat([value_pos_log, value_neg_log]).sort_index()
    return column

def log_feature(data:pd.Series or pd.DataFrame):
    if isinstance(data, pd.Series):
        logged_data = log_col(data)
    elif isinstance(data, pd.DataFrame):
        logged_data = data.apply(log_col)
    else:
        print("Not a valid format for values. Use pd.Series or pd.DataFrame")
        sys.exit(1)
    return logged_data

def make_feature_relative_again(data:Union[pd.Series, pd.DataFrame]):
    if isinstance(data, pd.Series):
        max_value = data.max()
        if max_value > 0:
            relative_data = data/max_value
        else:
            print(f"Max value equals {max_value}. This shouldn't be. Check data cleaning.")
            sys.exit(1)
    elif isinstance(data, pd.DataFrame):
        if (data.max() <= 0).any():
            for column in data.columns:
                if data[column].max() <= 0:
                    print(f"Column {column} has max value {data[column].max()}. This shouldn't be. Check data cleaning.")
            sys.exit(1)
        relative_data = data.apply(lambda col: col/col.max() if col.max() != 0 else col)
    else:
        print("Not a valid format for values. Use pd.Series or pd.DataFrame")
        sys.exit(1)
    return relative_data

def print_min_max(df):
    for feature in df.columns:
        print(feature)
        print(df[feature].min(), df[feature].max())
        # print('Range: ', df[feature].max()-df[feature].min())

def standard_scaling(df):
    """
    Standardize a pandas DataFrame (zero mean, unit variance).

    Parameters:
        df (pd.DataFrame): Input DataFrame with numerical features.

    Returns:
        pd.DataFrame: Standard-scaled DataFrame with same columns and index.
    """
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
    return scaled_df

def min_max_scaling(df):
    """
    MinMax a pandas DataFrame (zero mean, unit variance).

    Parameters:
        df (pd.DataFrame): Input DataFrame with numerical features.

    Returns:
        pd.DataFrame: MinMax-scaled DataFrame with same columns and index.
    """
    scaler = MinMaxScaler()
    scaled_array = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
    return scaled_df

def quantile_scaling(df):
    """
        Quantile scale a pandas DataFrame (zero mean, unit variance).

        Parameters:
            df (pd.DataFrame): Input DataFrame with numerical features.

        Returns:
            pd.DataFrame: Quantile-scaled DataFrame with same columns and index.
        """
    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=int(len(df) * 0.8))
    scaled_array = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
    return scaled_df

def create_imputed_df(imputation:str):
    fico_feats = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/fico/base_feats_no_cmp_918_24_01.xlsx',
                               index_col=0).astype(float)
    scip_feats = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/cleaned_scip/scip_default_clean_feats.csv').astype(float)

    imputed_fico = replace_and_impute(fico_feats.astype(float), imputation)
    imputed_scip = replace_and_impute(scip_feats.astype(float), imputation)

    imputed_fico.to_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/fico/{imputation}_imputed_fico_feats.csv')
    imputed_scip.to_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/cleaned_scip/{imputation}_imputed_scip_feats.csv')

def main(df: pd.DataFrame, data_set:str, imputation:str, scaler:str, relative=False, log=False):
    """
    Input: Unscaled Data as csv or xlsx
    Returns: Scaled Data as csv
    """
    working_df = df.copy().astype(float)

    if log:
        log_columns = get_log_cols(working_df)
        if len(log_columns)>0:
            working_df.loc[:, log_columns] = log_feature(working_df[log_columns])
    if relative:
        rel_cols = get_log_cols(working_df)
        working_df.loc[:, rel_cols] = make_feature_relative_again(working_df[rel_cols])
        working_df.to_csv(f"/Users/fritz/Downloads/ZIB/Master/NewTry/DataSets/cleaned/ready_for_ml/{data_set}_relative_ready_for_ml.csv")
        print(working_df.min())
    if scaler == 'Standard':
        working_df = standard_scaling(working_df)
    elif scaler == 'MinMax':
        working_df = min_max_scaling(working_df)
    elif scaler == 'Quantile':
        working_df = quantile_scaling(working_df)
    return working_df



create_imputed_df('median')

imputed_fico_feats = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/fico/median_imputed_fico_feats.csv',
                                 index_col=0).astype(float)
imputed_scip_feats = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/cleaned_scip/median_imputed_scip_feats.csv')
relative_fico = main(imputed_fico_feats, data_set='fico', imputation='median', scaler='None', relative=True)#.astype(float)

# logged_fico = main(imputed_fico_feats, data_set='fico', imputation='median', scaler='None')
# logged_scip = main(imputed_scip_feats, data_set='scip', imputation='median', scaler='None')
# logged_fico.to_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/fico/median_logged_fico_feats.csv',
#                    index=False)
#
# logged_scip.to_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/cleaned_scip/median_logged_scip_feats.csv',
#                    index=False)




