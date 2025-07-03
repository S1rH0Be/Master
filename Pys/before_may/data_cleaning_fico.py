import pandas as pd

import column_interplay
import ugly_fico_maybe_scip
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

# sets everything to zero which is smaller than 10**(-6)
def set_small_to_zero(df):
    numeric_cols = df.select_dtypes(include=['number']).columns  # Get numeric columns
    clean = df.copy()
    # replace values close to 0 with zero
    clean[numeric_cols] = clean[numeric_cols].where(abs(clean[numeric_cols]) >= 10 ** -6, 0) # where condition is not met replace with zero

    pointfive_is_zero_df = clean
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
    inf_count = np.isinf(numeric_df).sum().sum()
    df.loc[:, numeric_df.columns] = numeric_df
    return df
# TODO: Check if correct
def data_cleaning(data:str, requirement_data_frame:pd.DataFrame, scip=False):
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
    to_be_cleaned_df = ugly_fico_maybe_scip.read_and_rename(data, fico_columns_integer, fico_columns_double)
    # replace large values by infintiy
    to_be_cleaned_df = replace_large_with_inf(to_be_cleaned_df)


    # in deleted_instances we store all matrix names of instances which have a erroneuos entry
    deleted_instances = []
    # store all columns which get deleted due to reasons
    deleted_columns = []

    #need complete_df to get deleted instance names
    complete_df = to_be_cleaned_df.copy()

    clean_df, deleted_cols = drop_trivial(to_be_cleaned_df)
    deleted_columns = deleted_columns + deleted_cols

    # try to convert each column to the right datatype, if not possible find instances which are broken
    clean_df, broken_cols = ugly_fico_maybe_scip.datatype_converter(clean_df)
    if len(broken_cols)>0:
        deleted_instances += ugly_fico_maybe_scip.check_datatype(clean_df)

    clean_df, deleted_inst = ugly_fico_maybe_scip.check_col_consistency(clean_df, requirement_data_frame, SCIP=scip)
    deleted_instances += deleted_inst
    # add column with reason why instance got deleted
    # TODO: Check column_interplay
    # TODO: If i want deleted_df i need to rework column_interplay to return instance with deletion reason
    clean_df, deleted_instances_col_interplay = column_interplay.column_interplay(clean_df)
    deleted_instances.append(deleted_instances_col_interplay)

    # create df with deleted instances for checking
    deleted_instances_df = complete_df[complete_df['Matrix Name'].isin(deleted_instances)]
    #add a absolute timesave potential column
    x = np.round(clean_df['Final solution time (cumulative) Mixed']-clean_df['Final solution time (cumulative) Int'], 2).abs()
    clean_df.loc[:, 'Pot Time Save'] = x
    # delete_bad_instances also adds vbs column.
    clean_df = delete_bad_instances(clean_df)
    # very large values get handled as inf
    # TODO: Gets called a second time here reasonable?
    clean_df = replace_large_with_inf(clean_df)
    # everything close to zero gets set to zero
    clean_df = set_small_to_zero(clean_df)

    return clean_df, deleted_instances_df, deleted_columns

def main(scip=False):
    july = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/JuneTry/Bases/FICO/Raw/fico_data_18-07-25.csv')
    requirement_df = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/JuneTry/Bases/requirements.csv', delimiter=';')

    cleaned_df, deleted_df, columns_deleted = data_cleaning('/Users/fritz/Downloads/ZIB/Master/JuneTry/Bases/FICO/Raw/fico_data_18-07-25.csv',
                                                            requirement_df, scip=scip)


main()
#
# deleted_df.to_excel(
#         f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/JuneTry/BaseCSVs/deleted_data_final.xlsx',
#         index=False)
#
# clean_df.to_excel(
#         f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/wgwqgq3g3qg43qgqw34g314_{date_string}.xlsx', index=False)
# to_comp = pd.read_excel(
#         f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/wgwqgq3g3qg43qgqw34g314_{date_string}.xlsx')
#
# based = pd.read_excel(
#         f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/918/clean_data_final_06_03.xlsx')
#
# bool_df = to_comp==based
#
# unequal_cols = [col_name for col_name in bool_df.columns if (len(set(bool_df[col_name]))>1) ]
# for col in unequal_cols:
#     for ind in bool_df[col].index:
#         if not bool_df[col].loc[ind]:
#             print(col, ind, to_comp[col].loc[ind], based[col].loc[ind])

#
# feature_pointfive_is_zero_df = pointfive_is_zero_df[['Matrix Equality Constraints', 'Matrix Quadratic Elements',
#        'Matrix NLP Formula', 'Presolve Columns', 'Presolve Global Entities',
#        '#nodes in DAG', '#integer violations at root',
#        '#nonlinear violations at root', '% vars in DAG (out of all vars)',
#        '% vars in DAG unbounded (out of vars in DAG)',
#        '% vars in DAG integer (out of vars in DAG)',
#        '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)',
#        'Avg work for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
#        'Avg work for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
#        'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Mixed',
#        'Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
#        '#spatial branching entities fixed (at the root) Mixed',
#        'Avg coefficient spread for convexification cuts Mixed']].copy()
# feature_pointfive_is_zero_df.to_excel(
#         f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/clean_features_final_{date_string}.xlsx',
#         index=False)

"""
    5. Scale columns to same magnitude
        5.1 "inf" to ???
"""
