import pandas as pd

import column_interplay_july
import ugly_fico_maybe_scip_july
import numpy as np
from datetime import datetime

pd.options.mode.copy_on_write = True
# Get the current date for saving files
current_date = datetime.now()
# Format it as a string
date_string = current_date.strftime("%d_%m")


def drop_trivial(df):
    # Input: DataFrame
    # Output: DataFrame without trivial columns, list of column names which got dropped
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

# TODO: Check if correct
def replace_large_with_inf(df):
    numeric_df = df.select_dtypes(include=['number'])
    quasi_inf = 1 * np.e ** 39
    numeric_df[numeric_df > quasi_inf] = np.inf
    numeric_df[numeric_df < -quasi_inf] = -np.inf
    # inf_count = np.isinf(numeric_df).sum().sum()
    df.loc[:, numeric_df.columns] = numeric_df
    return df

# TODO: Check if correct
def data_cleaning(data:str, requirement_data_frame:pd.DataFrame, scip=False, fico=False, debuggen=False):
    # fico columns
    fico_columns_integer = ['#spatial branching entities fixed (at the root)',
                            '#non-spatial branch entities fixed (at the root)', '#nodes in DAG',
                            '#integer violations at root', '#nonlinear violations at root']
    fico_columns_double = [
        '% vars in DAG (out of all vars)',
        '% vars in DAG unbounded (out of vars in DAG)',
        '% vars in DAG integer (out of vars in DAG)',
        '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)',
        'Avg work for propagation + cutting / entity / rootLPticks',
        'Avg work for solving strong branching LPs for spatial branching (not including infeasible ones)',
        'Avg work for solving strong branching LPs for integer branchings (not including infeasible ones)',
        'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones)',
        'Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones)',
        'Avg coefficient spread for convexification cuts']
    # 6-9 are set to -1 if they did not happen

    # data is the name of the csv file as a string
    # for fico data i need to rename columns
    if fico:
        to_be_cleaned_df = ugly_fico_maybe_scip_july.read_and_rename(data, fico_columns_integer, fico_columns_double)
    # scip file is already named properly
    else:
        to_be_cleaned_df=pd.read_csv(data)


    # replace large values by infintiy
    # TODO maybe pnly do it later?
    to_be_cleaned_df = replace_large_with_inf(to_be_cleaned_df)

    # in deleted_instances we store all matrix names of instances which have a erroneuos entry
    deleted_instances = []
    # store all columns which get deleted due to reasons
    deleted_columns = []

    clean_df, deleted_cols = drop_trivial(to_be_cleaned_df)

    deleted_columns += deleted_cols

    del_instances = each_name_thrice(clean_df, 'Matrix Name')
    del_instances = [instance[0] for instance in del_instances]
    deleted_instances += del_instances
    clean_df = clean_df[~clean_df['Matrix Name'].isin(deleted_instances)]

    # try to convert each column to the right datatype, if not possible find instances which are broken
    clean_df, del_instances = ugly_fico_maybe_scip_july.datatype_converter(clean_df) # calls ugly_fico_maybe_scip_july.check_datatype
    deleted_instances += del_instances

    # deletes all #nodes in dag zeros
    clean_df, deleted_inst = ugly_fico_maybe_scip_july.check_col_consistency(clean_df, requirement_data_frame, scip=scip)
    deleted_inst = [del_inst[0] for del_inst in deleted_inst]
    deleted_instances += deleted_inst
    clean_df = clean_df[~clean_df['Matrix Name'].isin(deleted_instances)]

    # add column with reason why instance got delete
    # TODO: Check column_interplay
    clean_df, deleted_instances_col_interplay = column_interplay_july.column_interplay(clean_df, DEBUG=debuggen, fico=fico)
    deleted_instances += list(deleted_instances_col_interplay)
    # create df with deleted instances for checking
    deleted_instances_df = to_be_cleaned_df[to_be_cleaned_df['Matrix Name'].isin(deleted_instances)]
    # delete_bad_instances also adds vbs column.

    clean_df = delete_bad_instances(clean_df) # just replaces column Del reason with VBS
    # add an absolute timesave potential column
    x = np.round(clean_df['Final solution time (cumulative) Mixed']-clean_df['Final solution time (cumulative) Int'], 2).abs()
    clean_df.loc[:, 'Pot Time Save'] = x
    # very large values get handled as inf
    # TODO: Gets called a second time here reasonable?
    clean_df = replace_large_with_inf(clean_df)
    # everything close to zero gets set to zero
    clean_df = set_small_to_zero(clean_df)
    if fico:
        clean_df = pointfive_is_zero(clean_df)
    return clean_df, deleted_instances_df, deleted_columns

def create_combined_requirements(fico_reqs:pd.DataFrame, scip_reqs:pd.DataFrame):
    for col in scip_reqs.columns:
        if col not in fico_reqs.columns:
            fico_reqs[col.strip()] = scip_reqs[col]
    fico_reqs.dropna(axis=1, inplace=True)
    return fico_reqs

def main(scip=False, fico=True, scip_experiment=False):
    fico_requirement_df = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/JulyTry/Bases/Requirements/clean_requirements.csv')
    scip_requirements = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/JulyTry/Bases/Requirements/scip_requirements.xlsx')

    combined_requirements = create_combined_requirements(fico_requirement_df, scip_requirements)
    combined_requirements.to_csv('/Users/fritz/Downloads/ZIB/Master/JulyTry/Bases/Requirements/all_requirements.csv',
                                 index=False)

    if fico:
        print("FIIIIIICOOOOOOOOO")
        cleaned_df, deleted_df, columns_deleted = data_cleaning('/Users/fritz/Downloads/ZIB/Master/JulyTry/Bases/FICO/Raw/fico_data_18-07-25.csv',
                                                                combined_requirements, scip=False, fico=fico, debuggen=False)

        cleaned_df.to_csv('/Users/fritz/Downloads/ZIB/Master/JulyTry/Bases/FICO/Cleaned/fico_clean_data_753.csv',
                          index=False)

        deleted_df.to_csv('/Users/fritz/Downloads/ZIB/Master/JulyTry/Bases/FICO/Cleaned/fico_deletion.csv',
                          index=False)


        fico_feats = ['Matrix Equality Constraints', 'Matrix Quadratic Elements',
       'Matrix NLP Formula', 'Presolve Columns', 'Presolve Global Entities',
       '#nodes in DAG', '#integer violations at root',
       '#nonlinear violations at root', '% vars in DAG (out of all vars)',
       '% vars in DAG unbounded (out of vars in DAG)',
       '% vars in DAG integer (out of vars in DAG)',
       '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)',
       'Avg work for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
       'Avg work for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
       'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Mixed',
       'Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
       '#spatial branching entities fixed (at the root) Mixed',
       'Avg coefficient spread for convexification cuts Mixed']

        feature_df = cleaned_df[fico_feats]
        feature_df.to_csv('/Users/fritz/Downloads/ZIB/Master/JulyTry/Bases/FICO/Cleaned/fico_features_753.csv', index=False)

    if scip:
        print("SCIPIIIIIIIIIIIIIIII")
        cleaned_df, deleted_df, columns_deleted = data_cleaning(
            '/Users/fritz/Downloads/ZIB/Master/JulyTry/Bases/SCIP/NamedFeatures/Complete/scip_default_named_columns.csv',
            combined_requirements, scip=scip, fico=False, debuggen=False)

        cleaned_df.to_csv('/Users/fritz/Downloads/ZIB/Master/JulyTry/Bases/SCIP/Cleaned/scip_default_clean_data.csv',
                          index=False)

        deleted_df.to_csv('/Users/fritz/Downloads/ZIB/Master/JulyTry/Bases/SCIP/Cleaned/scip_default_deletion.csv',
                          index=False)

        scip_feature = ['#integer violations at root', '#nodes in DAG',
                        'Avg coefficient spread for convexification cuts Mixed',
                        'Presolve Global Entities', 'Presolve Columns',
                        '#nonlinear violations at root',
                        'Avg strong branching iterations in root',
                        'Matrix Equality Constraints', 'Matrix NLP Formula',
                        'Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
                        '% vars in DAG (out of all vars)',
                        '% vars in DAG integer (out of vars in DAG)',
                        '% vars in DAG unbounded (out of vars in DAG)',
                        '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)',
                        'Matrix Quadratic Elements']
        surviving_feats = [feat for feat in scip_feature if feat in cleaned_df.columns]

        scip_feature_df = cleaned_df[surviving_feats]
        scip_feature_df.to_csv('/Users/fritz/Downloads/ZIB/Master/JulyTry/Bases/SCIP/Cleaned/scip_default_features_cleaned.csv',
                               index=False)

    if scip_experiment:
        print("SCIPIIIIIIIIIIIIIIII EXPERIMENTIIIIII")
        cleaned_df, deleted_df, columns_deleted = data_cleaning(
            '/Users/fritz/Downloads/ZIB/Master/JulyTry/Bases/SCIP/NamedFeatures/Complete/scip_012345_named_columns.csv',
            combined_requirements, scip=scip, fico=False, debuggen=False)

        cleaned_df.to_csv('/Users/fritz/Downloads/ZIB/Master/JulyTry/Bases/SCIP/Cleaned/scip_experiment_clean_data.csv',
                          index=False)

        deleted_df.to_csv('/Users/fritz/Downloads/ZIB/Master/JulyTry/Bases/SCIP/Cleaned/scip_experiment_deletion.csv',
                          index=False)

        scip_feature = ['#integer violations at root', '#nodes in DAG',
                        'Avg coefficient spread for convexification cuts Mixed',
                        'Presolve Global Entities', 'Presolve Columns',
                        '#nonlinear violations at root',
                        'Avg strong branching iterations in root',
                        'Matrix Equality Constraints', 'Matrix NLP Formula',
                        'Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
                        '% vars in DAG (out of all vars)',
                        '% vars in DAG integer (out of vars in DAG)',
                        '% vars in DAG unbounded (out of vars in DAG)',
                        '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)',
                        'Matrix Quadratic Elements']
        surviving_feats = [feat for feat in scip_feature if feat in cleaned_df.columns]

        scip_feature_df = cleaned_df[surviving_feats]
        scip_feature_df.to_csv(
            '/Users/fritz/Downloads/ZIB/Master/JulyTry/Bases/SCIP/Cleaned/scip_experiment_features_cleaned.csv',
            index=False)

# main(scip=False, fico=True)
# main(scip=True, fico=False)
main(scip=True, fico=False, scip_experiment=False)


"""
    5. Scale columns to same magnitude
        5.1 "inf" to ???
"""
