import pandas as pd

import ugly #all functions where i need a lot of long text, like rename columns
import column_interplay
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
    for col in df.columns:
        if df[col].nunique()==1:
            dropped_cols.append(col)
            df.drop(col, axis=1, inplace=True)

    if len(dropped_cols)>0:
        print("Trivial ", dropped_cols)

    return df, dropped_cols

def set_small_to_zero(df):
    numeric_cols = df.select_dtypes(include=['number']).columns  # Get numeric columns
    clean = df.copy()
    # replace values close to 0 with zero
    clean[numeric_cols] = clean[numeric_cols].where(abs(clean[numeric_cols]) >= 10 ** -6, 0)
    # betrachte abs(timemixed-timeint)<=0.5 as 0
    pointfive_is_zero_df = clean.copy()
    # divide label by 100 to get factor instead of percent
    pointfive_is_zero_df['Cmp Final solution time (cumulative)'] = pointfive_is_zero_df['Cmp Final solution time (cumulative)'] / 100

    for index, row in pointfive_is_zero_df.iterrows():
        if abs(row['Final solution time (cumulative) Mixed'] - row['Final solution time (cumulative) Int']) <= 0.5:
            pointfive_is_zero_df.loc[index, 'Cmp Final solution time (cumulative)'] = 0.0
            pointfive_is_zero_df.loc[index, 'Pot Time Save'] = 0.0

        if abs(row['Cmp Final solution time (cumulative)']) <= 0.01:
            # 1% doesnt matter as well
            pointfive_is_zero_df.loc[index, 'Cmp Final solution time (cumulative)'] = 0.0
            pointfive_is_zero_df.loc[index, 'Pot Time Save'] = 0.0

    return pointfive_is_zero_df

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
    inf_count = np.isinf(numeric_df).sum()  # .sum()
    df.loc[:, numeric_df.columns] = numeric_df
    return df

def data_cleaning(data):
    #data is the name of the csv file as a string
    to_be_cleaned_df = ugly.read_and_rename(data)
    to_be_cleaned_df = replace_large_with_inf(to_be_cleaned_df)
    # in deleted_instances we store all matrix names of instances which have a erronous entry
    deleted_instances = []

    #need complete_df to get deleted instances names
    complete_df = to_be_cleaned_df.copy()

    clean_df, deleted_cols = drop_trivial(to_be_cleaned_df)
    clean_df, broken_cols = ugly.datatype_converter(clean_df)
    if len(broken_cols)>0:
        deleted_instances += ugly.check_datatype(clean_df)

    # add column with reason why instance got deleted
    clean_df, deleted_instances = column_interplay.column_interplay(clean_df)

    # create df with deleted instances for checking
    deleted_instances_df = complete_df[complete_df['Matrix Name'].isin(deleted_instances)]
    #add a absolute timesave potential column
    x = np.round(clean_df['Final solution time (cumulative) Mixed']-clean_df['Final solution time (cumulative) Int'], 2).abs()
    clean_df.loc[:, 'Pot Time Save'] = x

    clean_df = delete_bad_instances(clean_df)
    # very large values get handled as inf
    clean_df = replace_large_with_inf(clean_df)
    # everything close to zero gets set to zero
    clean_df = set_small_to_zero(clean_df)

    return clean_df, deleted_instances_df

requirement_df = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/requirements.csv', delimiter=';')
clean_df, deleted_df = data_cleaning('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/data_raw_august.csv')

deleted_df.to_excel(
        f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/deleted_data_final.xlsx',
        index=False)

clean_df.to_excel(
        f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/wgwqgq3g3qg43qgqw34g314_{date_string}.xlsx', index=False)
to_comp = pd.read_excel(
        f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/wgwqgq3g3qg43qgqw34g314_{date_string}.xlsx')

based = pd.read_excel(
        f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/918/clean_data_final_06_03.xlsx')

bool_df = to_comp==based

unequal_cols = [col_name for col_name in bool_df.columns if (len(set(bool_df[col_name]))>1) ]
for col in unequal_cols:
    for ind in bool_df[col].index:
        if not bool_df[col].loc[ind]:
            print(col, ind, to_comp[col].loc[ind], based[col].loc[ind])

#
# feature_pointfive_is_zero_df = pointfive_is_zero_df[['Matrix Equality Constraints', 'Matrix Quadratic Elements',
#        'Matrix NLP Formula', 'Presolve Columns', 'Presolve Global Entities',
#        '#nodes in DAG', '#integer violations at root',
#        '#nonlinear violations at root', '% vars in DAG (out of all vars)',
#        '% vars in DAG unbounded (out of vars in DAG)',
#        '% vars in DAG integer (out of vars in DAG)',
#        '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)',
#        'Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
#        'Avg ticks for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
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
