import pandas as pd

import column_interplay_july
import ugly_fico_maybe_scip_july
import numpy as np



pd.options.mode.copy_on_write = True

def remove_outliers(df, label:str, threshold):
    indices_to_keep = df[df[label].abs() <= threshold].index
    df = df.loc[indices_to_keep]
    return df

def drop_trivial(df):
    dropped_cols = []
    for column in df.columns:
        if df[column].nunique()==1:
            dropped_cols.append(column)
            df.drop(column, axis=1, inplace=True)
    return df, dropped_cols

def each_name_thrice(data_frame:pd.DataFrame, column_name:str):
    broken_inst = []

    matrix_names = data_frame[column_name].unique()
    for matrix_name in matrix_names:
        if len(data_frame[data_frame[column_name]==matrix_name])!=3:
            broken_inst.append((matrix_name, 'Not 3 times'))

    if len(broken_inst)>0:
        print("Each Name Thrice:", broken_inst)
    return broken_inst
# sets everything to zero which is smaller than 10**(-6)
def set_small_to_zero(df):
    numeric_cols = df.select_dtypes(include=['number']).columns  # Get numeric columns
    clean = df.copy()
    # replace values close to 0 with zero
    clean[numeric_cols] = clean[numeric_cols].where(abs(clean[numeric_cols]) >= 10 ** -6, 0) # where condition is not met replace with zero
    return clean

def pointfive_is_zero(df):
    pointfive_is_zero_df = df.copy()
    # divide label by 100 to get factor instead of percent
    pointfive_is_zero_df['Cmp Final solution time (cumulative)'] = pointfive_is_zero_df['Cmp Final solution time (cumulative)'] / 100
    for index, row in pointfive_is_zero_df.iterrows():
        # see time difference of half a second, abs(timemixed-timeint)<=0.5, as 0.0
        if abs(row['Final solution time (cumulative) Mixed'] - row['Final solution time (cumulative) Int']) <= 0.5:
            pointfive_is_zero_df.loc[index, 'Cmp Final solution time (cumulative)'] = 0.0
            pointfive_is_zero_df.loc[index, 'Pot Time Save'] = 0.0
        # if run time differs only by 1% set label to 0.0
        if abs(row['Cmp Final solution time (cumulative)']) <= 0.01:
            pointfive_is_zero_df.loc[index, 'Cmp Final solution time (cumulative)'] = 0.0
            pointfive_is_zero_df.loc[index, 'Pot Time Save'] = 0.0
    return pointfive_is_zero_df

# deletes all instances with erroneuos data, and adds potential time save column
def delete_bad_instances(df):
    clean = df.copy()
    # delete all 'bad' instances
    clean = clean[clean['Deletion Reason'] == ''].drop('Deletion Reason', axis=1)
    clean.loc[:, 'Virtual Best'] = [
        min(row['Final solution time (cumulative) Mixed'],
            row['Final solution time (cumulative) Int']) for index, row in clean.iterrows()]
    return clean

def replace_large_with_inf(df):
    numeric_df = df.select_dtypes(include=['number'])
    quasi_inf = 1 * np.e ** 39
    numeric_df[numeric_df > quasi_inf] = np.inf
    numeric_df[numeric_df < -quasi_inf] = -np.inf
    df.loc[:, numeric_df.columns] = numeric_df
    return df

def no_dag(df):
    no_dag_df = df.copy()
    no_dag_inst = [no_dag_df['Matrix Name'].loc[index] for index, row in no_dag_df.iterrows()
                   if no_dag_df['NodesInDAG'].loc[index]==0]
    return no_dag_inst

def no_int(df):
    no_int_df = df.copy()
    no_int_inst = [no_int_df['Matrix Name'].loc[index] for index, row in no_int_df.iterrows()
                   if no_int_df['IntVarsPostPre'].loc[index]==0]
    return no_int_inst

def nonlinearity_features(df:pd.DataFrame, scip):
    if scip:
        df['QuadrElements'] = df['QuadrElements']/(df['NodesInDAG']+df['Matrix non-zeros'])
        df['NodesInDAG'] = df['NodesInDAG']/(df['NodesInDAG'] + df['Matrix non-zeros'])
        df['EqCons'] = df['EqCons']/df['#Constraints']
        df['NonlinCons'] = df['NonlinCons']/df['#Constraints']

    if not scip:
        df['QuadrElements'] = df['QuadrElements'] / (
                    df['NodesInDAG'] + df['Presolve Elements'])
        df['NodesInDAG'] = df['NodesInDAG'] / (df['NodesInDAG'] + df['Presolve Elements'])
        df['EqCons'] = df['EqCons'] / df['Matrix Rows']
        df['NonlinCons'] = df['NonlinCons']/df['Matrix Rows']

    df.insert(9, 'IntVarsPostPre', df['Presolve Global Entities'] / df['Presolve Columns'])
    df.drop(['Presolve Global Entities', 'Presolve Columns'], axis=1, inplace=True)
    return df

def data_cleaning(data:str, requirement_data_frame:pd.DataFrame, scip=False, fico=False, DEBUG=False):
    fico_columns_integer = ['SpatBranchEntFixed',
                            'IntBranchEntFixed', 'NodesInDAG',
                            '#IntViols', '#NonlinViols']
    fico_columns_double = [
        '%VarsDAG',
        '%VarsDAGUnbnd',
        '%VarsDAGInt',
        '%QuadrNodesDAG',
        'AvgWorkPropa',
        'AvgWorkSBLPSpat',
        'AvgWorkSBLPInt',
        'AvgRelBndChngSBLPSpat',
        'AvgRelBndChngSBLPInt',
        'AvgCoeffSpreadConvCuts']
    # 6-9 are set to -1 if they did not happen

    # data is the name of the csv file as a string
    # for fico data i need to rename columns
    if fico:
        to_be_cleaned_df = ugly_fico_maybe_scip_july.read_and_rename(data, fico_columns_integer, fico_columns_double)
    # scip file is already named properly
    else:
        to_be_cleaned_df=pd.read_csv(data)

    # replace large values by infintiy
    to_be_cleaned_df = replace_large_with_inf(to_be_cleaned_df)
    if DEBUG:
        print("Instances ToBeCleaned:", len(to_be_cleaned_df))
    # in deleted_instances we store all matrix names of instances which have a erroneuos entry
    deleted_instances = []
    # store all columns which get deleted due to reasons
    deleted_columns = []
    clean_df, deleted_cols = drop_trivial(to_be_cleaned_df)
    deleted_columns += deleted_cols

    del_instances = each_name_thrice(clean_df, 'Matrix Name')
    del_instances = [instance[0] for instance in del_instances]
    pre_append = deleted_instances.copy()
    deleted_instances += del_instances
    clean_df = clean_df[~clean_df['Matrix Name'].isin(deleted_instances)]
    if DEBUG:
        print("Instances Left: Thrice", len(clean_df), len(set(del_instances)-set(pre_append)))
    # try to convert each column to the right datatype, if not possible find instances which are broken
    clean_df, del_instances = ugly_fico_maybe_scip_july.datatype_converter(clean_df) # calls ugly_fico_maybe_scip_july.check_datatype
    pre_append = deleted_instances.copy()
    deleted_instances += del_instances
    clean_df = clean_df[~clean_df['Matrix Name'].isin(deleted_instances)]
    if DEBUG:
        print("Instances Left: DataTypeConverter", len(clean_df), len(set(del_instances)-set(pre_append)))
    # deletes all NodesInDAG zeros
    clean_df, deleted_inst = ugly_fico_maybe_scip_july.check_col_consistency(clean_df, requirement_data_frame, scip=scip)
    pre_append = deleted_instances.copy()
    deleted_instances += deleted_inst
    clean_df = clean_df[~clean_df['Matrix Name'].isin(deleted_instances)]
    if DEBUG:
        print("Instances Left: Consis", len(clean_df), len(set(deleted_inst)-set(pre_append)))
    # add column with reason why instance got delete
    clean_df, deleted_instances_col_interplay = column_interplay_july.column_interplay(clean_df, DEBUG=False, fico=fico)
    pre_append = deleted_instances.copy()
    deleted_instances += list(deleted_instances_col_interplay)
    # create df with deleted instances for checking
    deleted_instances_df = to_be_cleaned_df[to_be_cleaned_df['Matrix Name'].isin(deleted_instances)]
    # delete_bad_instances also adds vbs column.
    clean_df = delete_bad_instances(clean_df) # just replaces column Del reason with VBS
    clean_df = clean_df[~clean_df['Matrix Name'].isin(deleted_instances)]
    if DEBUG:
        print("Instances Left: ColInterplay", len(clean_df), len(set(list(deleted_instances_col_interplay))-set(pre_append)))
    # add an absolute timesave potential column
    x = np.round(clean_df['Final solution time (cumulative) Mixed']-clean_df['Final solution time (cumulative) Int'], 2).abs()
    clean_df.loc[:, 'Pot Time Save'] = x
    # very large values get handled as inf
    clean_df = replace_large_with_inf(clean_df)
    if DEBUG:
        print("Instances Left: Round ", len(clean_df))
    # everything close to zero gets set to zero
    clean_df = set_small_to_zero(clean_df)
    if fico:
        clean_df = pointfive_is_zero(clean_df)
    if DEBUG:
        print("Instances Left: SmallZero", len(clean_df))
    clean_df.columns = clean_df.columns.str.replace(' Mixed', '', regex=False)
    del_inst = no_dag(clean_df)
    pre_append = deleted_instances.copy()
    deleted_instances += del_inst
    clean_df = clean_df[~clean_df['Matrix Name'].isin(del_inst)]
    if DEBUG:
        print("Instances Left: No Dag", len(clean_df), len(set(list(del_inst))-set(pre_append)))
    clean_df = nonlinearity_features(clean_df, scip)
    # clean_df = clean_df[~clean_df['Matrix Name'].isin(deleted_instances)]
    if DEBUG:
        print("Instances Left: NonLin", len(clean_df))
    del_inst = no_int(clean_df)
    pre_append = deleted_instances.copy().copy()

    deleted_instances += del_inst
    clean_df = clean_df[~clean_df['Matrix Name'].isin(del_inst)]
    if DEBUG:
        print("Instances Left: No Int", len(clean_df), len(set(list(del_inst))-set(pre_append)))
    # For safety check if everything is there 3 times
    del_instances = each_name_thrice(clean_df, 'Matrix Name')
    del_instances = [instance[0] for instance in del_instances]
    pre_append = deleted_instances.copy()
    deleted_instances += del_instances
    clean_df = clean_df[~clean_df['Matrix Name'].isin(deleted_instances)]
    if DEBUG:
        print("Instances Left: Thrice", len(clean_df), len(set(list(del_instances))-set(pre_append)))
    return clean_df, deleted_instances_df, deleted_columns

def create_combined_requirements(fico_reqs:pd.DataFrame, scip_reqs:pd.DataFrame):
    for col in scip_reqs.columns:
        if col not in fico_reqs.columns:
            fico_reqs[col.strip()] = scip_reqs[col]
    fico_reqs.dropna(axis=1, inplace=True)
    return fico_reqs

def log_cols(df):
    features_to_log = ['#IntViols',
                       '#NonlinViols',
                       'SpatBranchEntFixed',
                       'AvgWorkSBLPSpat',
                       'AvgRelBndChngSBLPSpat',
                       'AvgWorkSBLPInt',
                       'AvgRelBndChngSBLPInt',
                       'AvgCoeffSpreadConvCuts']
    for col in features_to_log:
        if col in df.columns:
            mask = df[col].notna() & (df[col] != -1)
            df.loc[mask, col] = np.log10(df.loc[mask, col] + 1)

    return df

def logger(df):
    for col in df.columns:
        if df[col].max() > 10:
            df[col] = np.log10(df[col] + 1)
    return df

def main(requirement_df:str, raw_data_csv_path:str, save_clean_data_path:str, scip=False, fico=True, debuggen=False):

    combined_requirements = pd.read_csv(requirement_df)

    if fico:
        cleaned_df, deleted_df, columns_deleted = data_cleaning(raw_data_csv_path,
                                                                combined_requirements, scip=False, fico=True,
                                                                DEBUG=debuggen)
        cleaned_df = log_cols(cleaned_df)
        cleaned_df.dropna(axis=0)
        fico_feats = ['EqCons',
                      'QuadrElements',
                      'NonlinCons',
                      'IntVarsPostPre',
                      'NodesInDAG',
                      '#IntViols',
                      '#NonlinViols',
                      '%VarsDAG',
                      '%VarsDAGUnbnd',
                      '%VarsDAGInt',
                      '%QuadrNodesDAG',
                      'AvgWorkSBLPSpat',
                      'AvgWorkSBLPInt',
                      'AvgRelBndChngSBLPSpat',
                      'AvgRelBndChngSBLPInt',
                      'SpatBranchEntFixed',
                      'AvgCoeffSpreadConvCuts']
        cleaned_df.to_csv(f'{save_clean_data_path}.csv',
                          index=False)

        feature_df = cleaned_df[fico_feats]

        feature_df.to_csv(f'{save_clean_data_path}_features.csv', index=False)

        deleted_df.to_csv(f'{save_clean_data_path}_deletion.csv',
                          index=False)

    if scip:
        scip_feature = ['#IntViols', 'NodesInDAG',
                        'AvgCoeffSpreadConvCuts',
                        '#NonlinViols', 'IntVarsPostPre',
                        'SBWork',
                        'EqCons', 'NonlinCons',
                        'AvgRelBndChngSBLPInt',
                        '%VarsDAG',
                        '%VarsDAGInt',
                        '%VarsDAGUnbnd',
                        '%QuadrNodesDAG',
                        'QuadrElements']

        cleaned_df, deleted_df, columns_deleted = data_cleaning(raw_data_csv_path, combined_requirements,
                                                                scip=True, fico=False, DEBUG=debuggen)
        cleaned_df.to_csv(f'{save_clean_data_path}.csv',
                          index=False)

        deleted_df.to_csv(f'{save_clean_data_path}_deletion_.csv',
                          index=False)


        surviving_feats = [feat for feat in scip_feature if feat in cleaned_df.columns]
        scip_feature_df = cleaned_df[surviving_feats]
        scip_feature_df =logger(scip_feature_df)
        scip_feature_df.to_csv(f'{save_clean_data_path}_features.csv', index=False)







