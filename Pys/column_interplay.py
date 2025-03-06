from typing import List
import string
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def delete_instances(df, instances, reason):
    """"Input: df pandas dataframe, instances: set of strings containing names of instances which should be deleted
    Output: df pandas dataframe with deleted instances removed and reason why added to column Deletion Reason"""

    if not ('Deletion Reason' in df.columns):
        df.loc[:, 'Deletion Reason'] = ''

    df.loc[df['Matrix Name'].isin(instances), 'Deletion Reason'] = reason

    #df = df[~df['Matrix Name'].isin(instances)]
    return df

def opt_opt(df):
    """Input: df
    Output: List of instances names which have different entries where they should be equal"""
    broken_instances:List['string'] = []
    opt_opt_df = df[(df['Status Mixed'] == 'Optimal') & (df['Status Int'] == 'Optimal')]

    for index, row in opt_opt_df.iterrows():
        if abs(row['Final Objective Mixed'] - row['Final Objective Int']) > 10**(-3):
            broken_instances.append(row['Matrix Name'])

    return df, broken_instances

def timeout_time(df):
    """
    if status is timeout the time should be >=3600
    Input: df pandas dataframe
    output: instances which terminated too early
    """
    too_early = []

    mixed_timeout_df = df[df['Status Mixed']=='Timeout'][['Matrix Name', 'Final solution time (cumulative) Mixed']]
    int_timeout_df = df[df['Status Int']=='Timeout'][['Matrix Name', 'Final solution time (cumulative) Int']]

    too_early += mixed_timeout_df[mixed_timeout_df['Final solution time (cumulative) Mixed'] < 3600]['Matrix Name'].tolist()
    too_early += int_timeout_df[int_timeout_df['Final solution time (cumulative) Int'] < 3600]['Matrix Name'].tolist()

    return too_early

def entities_vs_vars():
    """Again need to check what happens here"""
#only constraint on ticks is, that they are either nonnegative or equal to -1
def tickst_du_richtig(df):
    negative_ticks = []
    return df, negative_ticks

def permutations(df):
    """columns where each instance needs to be equal to its permutations"""
    broken_cols = []
    bad_instances = []
    equal_cols = [('Presolve Columns Mixed', 'Presolve Columns Int'), ('Presolve Global Entities Mixed', 'Presolve Global Entities Int'),
                  ('permutation seed Mixed', 'permutation seed Int')]
    equal_col_entries_permutation = ['Matrix Equality Constraints', 'Matrix Quadratic Elements', 'Matrix NLP Formula']
    def check_equality(df, col_tupel_lst):
        unequal_cols = []
        for tupel in col_tupel_lst:
            if df[tupel[0]].equals(df[tupel[1]]):
                continue
            else:
                unequal_cols.append(tupel)
        return unequal_cols

    def check_same_entries_for_permutations(df, column):
        bad_perm = []
        for i in range(0, len(df), 3):
            # Prüfen, ob die Gruppe von drei Werten gleich ist
            if not (df[column].iloc[i] == df[column].iloc[i + 1] == df[column].iloc[i + 2]):
                bad_perm.append(df['Matrix Name'].iloc[i])
                return bad_perm
        return []


    for col in equal_col_entries_permutation:
        x =  check_same_entries_for_permutations(df, col)
        bad_instances += x


    broken_cols += check_equality(df, equal_cols)

    if len(broken_cols) > 0:
        for tupel in broken_cols:
            bad_instances += df['Matrix Name'].loc[df[tupel[0]] != df[tupel[1]],:].to_list()
    return bad_instances

def perm_consistent(df):
    eq_cols = df[['Matrix Name', 'Matrix Equality Constraints',
                  'Matrix Quadratic Elements', 'Matrix NLP Formula']]
    not_eq = []
    eq_cols = eq_cols.sort_values(by='Matrix Name')
    drop_reasons = {}

    # Collect inconsistent permutations
    for i in range(0, len(eq_cols), 3):
        if not (eq_cols.iloc[i].equals(eq_cols.iloc[i+1]) and eq_cols.iloc[i].equals(eq_cols.iloc[i+2])):
            instance = eq_cols['Matrix Name'].iloc[i]
            not_eq.append(instance)
            drop_reasons.setdefault(
                'inconsistent permutations', []).append(instance)

    # Drop inconsistent permutations
    for i in not_eq:
        df.drop(df[df['Matrix Name'] == i].index, inplace=True)

    eq_opt = df[['Matrix Name', 'Status Int', 'Status Mixed',
                 'Final Objective Int', 'Final Objective Mixed', 'Cmp Final Objective']]
    #eq_opt = eq_opt.loc[find_optopt_index(
    #    eq_opt).to_list()].sort_values(by='Matrix Name')
    eq_opt_int = eq_opt[eq_opt['Status Int'] == 'Optimal']
    eq_opt_mixed = eq_opt[eq_opt['Status Mixed'] == 'Optimal']
    number_status_opt_int = eq_opt_int['Matrix Name'].value_counts()
    number_status_opt_mixed = eq_opt_mixed['Matrix Name'].value_counts()

    unique_opt = number_status_opt_int[number_status_opt_int == 1].index.to_list()+number_status_opt_mixed[number_status_opt_mixed == 1].index.to_list()

    # Drop unique optimals
    for i in unique_opt:
        eq_opt.drop(eq_opt[eq_opt['Matrix Name'] == i].index[0], inplace=True)

    pair_opt = number_status_opt_int[number_status_opt_int == 2].index.to_list()+number_status_opt_mixed[number_status_opt_mixed == 2].index.to_list()
    index_pairs = [eq_opt[eq_opt['Matrix Name'] == i].index for i in pair_opt]
    cmp_opt_pairs = [(abs(eq_opt['Final Objective Int'].loc[i[0]]) -
                      abs(eq_opt['Final Objective Mixed'].loc[i[1]])) for i in index_pairs]
    not_same_opt = [x for x in cmp_opt_pairs if abs(x) > 10**(-3)]

    for i in index_pairs:
        eq_opt.drop(i[0], inplace=True)
        eq_opt.drop(i[1], inplace=True)

    triple_opt = number_status_opt_int[number_status_opt_int == 3].index.to_list()+number_status_opt_mixed[number_status_opt_mixed == 3].index.to_list()
    for i in triple_opt:
        all_perms = eq_opt[eq_opt['Matrix Name'] == i]
        max_opt_int = all_perms['Final Objective Int'].max()
        min_opt_int = all_perms['Final Objective Int'].min()
        wurm = pd.Series((max_opt_int - min_opt_int), index=all_perms.index)
        if wurm.max() > 0.001:
            not_same_opt.append(i)

    # Drop inconsistent optimal values across permutations
    for i in not_same_opt:
        df.drop(df[df['Matrix Name'] == i].index, inplace=True)
        drop_reasons.setdefault(
            'Inconsistent optimal values across permutations', []).append(i)

    return drop_reasons

def equal_cols_to_static(dataframe):
    eq_col_names = {}
    static_df = dataframe

    for i in range(0, len(dataframe.columns)-1):
        if (dataframe.iloc[:, i] == dataframe.iloc[:, i + 1]).all():
            eq_col_names[dataframe.columns[i]] = dataframe.columns[i].replace(' Mixed', '')
            static_df = static_df.drop(dataframe.columns[i+1], axis=1)

    static_df.rename(columns=eq_col_names, inplace=True)

    return static_df

def too_long(df):
    bad_instances = []
    for index, row in df.iterrows():
        if df['Final solution time (cumulative) Mixed'].loc[index]>3636:
            bad_instances.append(index)
        if df['Final solution time (cumulative) Int'].loc[index]>3636:
            bad_instances.append(index)
    return bad_instances

def column_interplay(df):
    #important: first timeout_time checken, then scale columns with +100
    deleted_instances = []

    cleaner_df, del_instances = opt_opt(df)
    cleaner_df = delete_instances(cleaner_df, del_instances, 'Optimal Value too different')
    deleted_instances += del_instances
    deleted_instances += timeout_time(cleaner_df)
    cleaner_df = delete_instances(cleaner_df, timeout_time(cleaner_df), 'Timeout too soon')
    deleted_instances += permutations(cleaner_df)
    cleaner_df = delete_instances(cleaner_df, permutations(cleaner_df), 'Permutations not consistent')
    deleted_instances += too_long(cleaner_df)
    cleaner_df = delete_instances(cleaner_df, too_long(cleaner_df), 'Timeout too long')
    cleaner_df, too_many_ticks = tickst_du_richtig(cleaner_df)
    cleaner_df = delete_instances(cleaner_df, too_many_ticks, 'Ticks > 100')
    deleted_instances += too_many_ticks
    cleaner_df = equal_cols_to_static(cleaner_df)
    deleted_instances = set(deleted_instances)
    return cleaner_df, deleted_instances


opt_opt_same_lst =['Final Objective Mixed',	'Final Objective Int']

scaling_plus_hundred = ['#MIP nodes Mixed',	'#MIP nodes Int', 'Final solution time (cumulative) Mixed',	'Final solution time (cumulative) Int',
                        '#spatial branching entities fixed (at the root) Mixed', '#spatial branching entities fixed (at the root) Int',
                        'Avg coefficient spread for convexification cuts Mixed', 'Avg coefficient spread for convexification cuts Int',
                        'Final Objective Mixed', 'Final Objective Int']

cmp_tupel = [('#MIP nodes Mixed', '#MIP nodes Int'), ('Final solution time (cumulative) Mixed',	'Final solution time (cumulative) Int'),
             ('#spatial branching entities fixed (at the root) Mixed', '#spatial branching entities fixed (at the root) Int'),
             ('Avg coefficient spread for convexification cuts Mixed', 'Avg coefficient spread for convexification cuts Int'),
             ('Final Objective Mixed', 'Final Objective Int')]

