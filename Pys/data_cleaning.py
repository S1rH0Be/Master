import pandas as pd

import ugly #all functions where i need a lot of long text, like rename columns
import column_interplay
import numpy as np
from datetime import datetime

pd.options.mode.copy_on_write = True

def drop_trivial(df):
    #input: DataFrame
    #Output: DataFrame without trivial columns, list of column names which got dropped 
    #Theoretisch: wenn cmp column is zero then delete int and mixed columns as well
    dropped_cols = []
    for i in df.columns:
        if df[i].nunique()==1:
            dropped_cols.append(i)
            df.drop(i, axis=1, inplace=True)
    #if cmp in dropped_cols:....
    if len(dropped_cols)>0:
        print("Trivial ", dropped_cols)
    return df, dropped_cols
    
def data_cleaning(data):
    #data is the name of the csv file as a string
    df = ugly.read_and_rename(data)
    numeric_df = df.select_dtypes(include=['number'])
    quasi_inf = 1*np.e**39
    numeric_df[numeric_df>quasi_inf] = np.inf
    numeric_df[numeric_df<-quasi_inf] = -np.inf
    inf_count = np.isinf(numeric_df).sum().sum()
    df.loc[:, numeric_df.columns] = numeric_df

    #need complete_df to get deleted instances names
    complete_df = df
    df, dropped_triv_lst = drop_trivial(df)

    df, dropped_wrong_type = ugly.datatype_converter(df)
    
    df, not_consistent_cols = ugly.check_col_consistency(df, requirement_df) #hier doppel gemoppelt mit float/int checker
    # add column with reason why instance got deleted
    df, different_opt_val = column_interplay.column_interplay(df)
    deleted_cols = []
    #deleted_cols += dropped_triv_lst
    deleted_cols += not_consistent_cols
    deleted_cols += dropped_wrong_type
    deleted_instances = different_opt_val
    #create df with deleted instances for checking
    #now ugly later beautiful
    deleted_instances_df = complete_df[complete_df['Matrix Name'].isin(deleted_instances)]

    #add a absolute timesave potential column
    x = np.round(df['Final solution time (cumulative) Mixed']-df['Final solution time (cumulative) Int'], 6).abs()
    df.loc[:, 'Pot Time Save'] = x
    return df, deleted_instances_df

requirement_df = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/requirements.csv', delimiter=';')
clean_df, deleted_df = data_cleaning('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/data_raw_august.csv')

# Assuming df is your existing DataFrame
# Replace absolute values smaller than 10**-6 with 0 in numeric columns
numeric_cols = clean_df.select_dtypes(include=['number']).columns  # Get numeric columns
# replace values close to 0 with zero
clean_df[numeric_cols] = clean_df[numeric_cols].where(abs(clean_df[numeric_cols]) >= 10**-6, 0)

#delete all 'bad' instances
clean_df = clean_df[clean_df['Deletion Reason']==''].drop('Deletion Reason', axis=1)
clean_df.loc[:, 'Virtual Best'] = [min(row['Final solution time (cumulative) Mixed'], row['Final solution time (cumulative) Int']) for index,row in clean_df.iterrows()]

#betrachte abs(timemixed-timeint)<=0.5 as 0
pointfive_is_zero_df = clean_df.copy()
# divide label by 100 to get factor instead of percent
pointfive_is_zero_df['Cmp Final solution time (cumulative)'] = pointfive_is_zero_df['Cmp Final solution time (cumulative)']/100
for index, row in pointfive_is_zero_df.iterrows():
    if abs(row['Final solution time (cumulative) Mixed']-row['Final solution time (cumulative) Int'])<=0.5:
        pointfive_is_zero_df.loc[index, 'Cmp Final solution time (cumulative)'] = 0.0
        pointfive_is_zero_df.loc[index, 'Pot Time Save'] = 0.0

    if abs(row['Cmp Final solution time (cumulative)'])<=0.01:
        #1% doesnt matter as well
        pointfive_is_zero_df.loc[index, 'Cmp Final solution time (cumulative)'] = 0.0
        pointfive_is_zero_df.loc[index, 'Pot Time Save'] = 0.0

# Get the current date
current_date = datetime.now()
# Format it as a string
date_string = current_date.strftime("%d_%m")

pointfive_is_zero_df.to_excel(
        f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/clean_data_final_{date_string}.xlsx',
        index=False)

# print(feature_candidates.columns)
features = ['Matrix Equality Constraints', 'Matrix Quadratic Elements',
       'Matrix NLP Formula', 'Presolve Columns', 'Presolve Global Entities',
       '#nodes in DAG', '#integer violations at root',
       '#nonlinear violations at root', '% vars in DAG (out of all vars)',
       '% vars in DAG unbounded (out of vars in DAG)',
       '% vars in DAG integer (out of vars in DAG)',
       '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)',
       'Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
       'Avg ticks for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
       'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Mixed',
       'Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
       'Cmp #spatial branching entities fixed (at the root)',
       'Cmp Avg coefficient spread for convexification cuts']

feature_df = pointfive_is_zero_df[features]

feature_df.to_excel(
        f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/final_features_{date_string}.xlsx',
        index=False)

"""
    2. Add reason column to deleted_df
    3. Scaling: #Aka auf alle Cmprelevanten Spalten +100 oder so
        3.3 Was mach ich mit denen hier: ['#non-spatial branch entities fixed (at the root) Mixed', '#non-spatial branch entities fixed (at the root) Int']
    5. Scale columns to same magnitude
        5.1 "inf" to ???
    6. Modell drauf werfen
    
    """
