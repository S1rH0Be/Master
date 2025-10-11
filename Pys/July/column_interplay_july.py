from typing import List
import string
import pandas as pd

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

def too_long(df):
    bad_instances = []
    for index, row in df.iterrows():
        if df['Final solution time (cumulative) Mixed'].loc[index]>3636:
            bad_instances.append(df['Matrix Name'].loc[index])
        if df['Final solution time (cumulative) Int'].loc[index]>3636:
            bad_instances.append(df['Matrix Name'].loc[index])
    return bad_instances

# TODO: Write this function. Think about what this should do
def entities_vs_vars():
    """Again need to check what happens here"""

# only constraint on ticks is, that they are either nonnegative or equal to -1
# TODO: Ask Timo if <100 correct
def tickst_du_richtig(df):
    negative_ticks = []
    return df, negative_ticks
# TODO: Get columns which should be equal in one permutation
def permutations(df, fico):
    """columns where each instance needs to be equal to its permutations"""
    broken_cols = []
    bad_instances = []
    equal_cols = [('Presolve Columns Mixed', 'Presolve Columns Int'), ('Presolve Global Entities Mixed', 'Presolve Global Entities Int'),
                  ('permutation seed Mixed', 'permutation seed Int')]
    equal_col_entries_permutation = ['EqCons', 'QuadrElements', 'NonlinCons']

    def check_equality(data_frame, col_tupel_lst):
        unequal_cols = []
        for int_mixed_pair in col_tupel_lst:
            if data_frame[int_mixed_pair[0]].equals(data_frame[int_mixed_pair[1]]):
                continue
            else:
                unequal_cols.append(int_mixed_pair)
        return unequal_cols

    def check_same_entries_for_permutations(data_frame, column):
        bad_perm = []
        for i in range(0, len(data_frame), 3):
            # Prüfen, ob die Gruppe von drei Werten gleich ist
            if not (data_frame[column].iloc[i] == data_frame[column].iloc[i + 1] == data_frame[column].iloc[i + 2]):
                bad_perm.append(data_frame['Matrix Name'].iloc[i])
        return bad_perm


    for col in equal_col_entries_permutation:
        x = check_same_entries_for_permutations(df, col)
        bad_instances += x

    if fico:
        broken_cols += check_equality(df, equal_cols)

    if len(broken_cols) > 0:
        for tupel in broken_cols:
            bad_instances += df['Matrix Name'].loc[df[tupel[0]] != df[tupel[1]],:].to_list()
    return bad_instances

def perm_consistent(df, fico:bool, DEBUG=False):
    eq_cols = df[['Matrix Name', 'EqCons',
                  'QuadrElements', 'NonlinCons']]
    not_eq = []
    eq_cols = eq_cols.sort_values(by='Matrix Name')
    drop_instances = []

    # Collect inconsistent permutations
    for i in range(0, len(eq_cols), 3):
        if not (eq_cols.iloc[i].equals(eq_cols.iloc[i+1]) and eq_cols.iloc[i].equals(eq_cols.iloc[i+2])):
            instance = eq_cols['Matrix Name'].iloc[i]
            not_eq.append(instance)
            drop_instances.append(instance)
    if DEBUG:
        print(f"PERM CONSISTENT STATIC FEATS: {len(set(drop_instances))}")
    # Drop inconsistent permutations
    df = df[~df['Matrix Name'].isin(not_eq)]

    # only fico data has objective value, scip doesnt
    if fico:
        eq_opt = df[['Matrix Name', 'Status Int', 'Status Mixed',
                     'Final Objective Int', 'Final Objective Mixed', 'Cmp Final Objective']]

        eq_opt_int = eq_opt[eq_opt['Status Int'] == 'Optimal']
        eq_opt_mixed = eq_opt[eq_opt['Status Mixed'] == 'Optimal']

        eq_opt_int_indices = eq_opt_int.index
        eq_opt_mixed_indices = eq_opt_mixed.index

        number_status_opt_int = eq_opt_int['Matrix Name'].value_counts() # pandas series, index is matrix name
        number_status_opt_mixed = eq_opt_mixed['Matrix Name'].value_counts() # pandas series, index is matrix name

        unique_opt = set(number_status_opt_int[number_status_opt_int == 1].index.to_list()+number_status_opt_mixed[number_status_opt_mixed == 1].index.to_list())
        unique_combined_opt = []
        for matrix in unique_opt:
            matrix_indices = df[df['Matrix Name'] == matrix].index
            count_of_index_appearances_int = [0,0,0]
            count_of_index_appearances_mixed = [0, 0, 0]
            current_count = 0
            for index in matrix_indices:
                if index in eq_opt_int_indices:
                    count_of_index_appearances_int[current_count] += 1
                if index in eq_opt_mixed_indices:
                    count_of_index_appearances_mixed[current_count] += 1
                current_count += 1
            count_of_index_appearances = count_of_index_appearances_int+count_of_index_appearances_mixed
            if count_of_index_appearances.count(0)>=5:
                unique_combined_opt.append(matrix)
                drop_instances.append(matrix)
        if DEBUG:
            print("SolvedOnlyOnce", len(set(unique_combined_opt)))
        # Drop unique optimals
        for i in unique_combined_opt:
            eq_opt.drop(eq_opt[eq_opt['Matrix Name'] == i].index, inplace=True)
        # TODO resolve if i should use absolute or relative threshold. i guess relative
        # TODO Propably abs(1-(int/mixed))
        # not_same_opt stores all matrix names which should be deleted later
        not_same_opt = []


        # check if two permutations are solved to optimality if the optimal value is equal
        pair_opt_int = number_status_opt_int[number_status_opt_int == 2].index.to_list()
        pair_opt_mixed = number_status_opt_mixed[number_status_opt_mixed == 2].index.to_list()
        triple_opt_int = number_status_opt_int[number_status_opt_int == 3].index.to_list()
        triple_opt_mixed = number_status_opt_mixed[number_status_opt_mixed == 3].index.to_list()


        for matrix in triple_opt_int:
            if matrix in pair_opt_mixed:
                pair_opt_mixed.remove(matrix)
            if matrix in pair_opt_int:
                pair_opt_int.remove(matrix)
        for matrix in triple_opt_mixed:
            if matrix in pair_opt_mixed:
                pair_opt_mixed.remove(matrix)
            if matrix in pair_opt_int:
                pair_opt_int.remove(matrix)

        pair_opt_int_df = eq_opt_int[eq_opt_int['Matrix Name'].isin(pair_opt_int)]
        pair_opt_mixed_df = eq_opt_mixed[eq_opt_mixed['Matrix Name'].isin(pair_opt_mixed)]


        index_pairs_int = [pair_opt_int_df[pair_opt_int_df['Matrix Name']==matrix].index for matrix in pair_opt_int]
        index_pairs_mixed = [pair_opt_mixed_df[pair_opt_mixed_df['Matrix Name']==matrix].index for matrix in pair_opt_mixed]

        cmp_opt_pairs_int = [(abs(eq_opt['Final Objective Int'].loc[i[0]]) -
                              abs(eq_opt['Final Objective Int'].loc[i[1]])) for i in index_pairs_int]

        cmp_opt_pairs_mixed = [(abs(eq_opt['Final Objective Mixed'].loc[i[0]]) -
                              abs(eq_opt['Final Objective Mixed'].loc[i[1]])) for i in index_pairs_mixed]

        not_same_opt_int = [x for x in cmp_opt_pairs_int if abs(x) > 10**(-3)]
        not_same_opt_mixed = [x for x in cmp_opt_pairs_mixed if abs(x) > 10 ** (-3)]
        not_same_opt_index_int = [index_pairs_int[i] for i in range(len(not_same_opt_int)) if abs(not_same_opt_int[i]) > 10**(-3)]
        not_same_opt_index_mixed = [index_pairs_mixed[i] for i in range(len(not_same_opt_mixed)) if
                              abs(not_same_opt_mixed[i]) > 10 ** (-3)]

        all_indices_which_should_be_deleted = not_same_opt_index_int+not_same_opt_index_mixed
        set_indices = []
        for i in all_indices_which_should_be_deleted:
            if i[0] not in set_indices:
                set_indices.append(i[0])
            if i[1] not in set_indices:
                set_indices.append(i[1])

        matrix_names_which_should_de_deleted = [df['Matrix Name'].loc[index] for index in set_indices]
        not_same_opt+=matrix_names_which_should_de_deleted

        for i in triple_opt_int:
            all_perms = eq_opt[eq_opt['Matrix Name'] == i]
            max_opt_int = all_perms['Final Objective Int'].max()
            min_opt_int = all_perms['Final Objective Int'].min()
            wurm = pd.Series((max_opt_int - min_opt_int), index=all_perms.index)
            if wurm.max() > 0.001:
                not_same_opt.append(i)

        for i in triple_opt_mixed:
            all_perms = eq_opt[eq_opt['Matrix Name'] == i]
            max_opt_mixed = all_perms['Final Objective Mixed'].max()
            min_opt_mixed = all_perms['Final Objective Mixed'].min()
            min_max_diff = max_opt_mixed - min_opt_mixed
            if min_max_diff > 0.001:
                if i not in not_same_opt_mixed:
                    not_same_opt.append(i)
        if DEBUG:
            print("NotSameOpt:", len(set(not_same_opt)))
        # Drop inconsistent optimal values across permutations
        for instance_name in not_same_opt:
            df.drop(df[df['Matrix Name'] == instance_name].index, inplace=True)
            drop_instances.append(instance_name)
    if fico:
        return df, drop_instances, not_same_opt, unique_combined_opt
    else:
        return df, drop_instances, [], []

def equal_cols_to_static(dataframe):
    eq_col_names = {}
    static_df = dataframe

    # HIIIIIIIIIIAAAAAAAAHHHHHHHHHHHHHHH
    for i in range(0, len(dataframe.columns)-1):
        if (dataframe.iloc[:, i] == dataframe.iloc[:, i + 1]).all():
            eq_col_names[dataframe.columns[i]] = dataframe.columns[i].replace(' Mixed', '')
            static_df = static_df.drop(dataframe.columns[i+1], axis=1)

    static_df.rename(columns=eq_col_names, inplace=True)

    return static_df

def column_interplay(df:pd.DataFrame, fico:bool, DEBUG=True):
    #important: first timeout_time checken, then scale columns with +100
    deleted_instances = []
    # check if optimal val is equal if both rules solved instance to optimality
    cleaner_df, del_instances = opt_opt(df)

    cleaner_df = delete_instances(cleaner_df, del_instances, 'Optimal Value too different')
    pre_append = deleted_instances.copy()
    deleted_instances += del_instances
    if DEBUG:
        print('opt_opt', len(set(del_instances)), len(set(del_instances)-set(pre_append)))
    # check if solver didn't stop too early when status == timeout
    del_instances = timeout_time(cleaner_df)
    pre_append = deleted_instances.copy()
    deleted_instances += timeout_time(cleaner_df)
    cleaner_df = delete_instances(cleaner_df, timeout_time(cleaner_df), 'Timeout too soon')
    if DEBUG:
        print('too_early', len(set(del_instances)), len(set(del_instances)-set(pre_append)))
    # check if solver stopped in time when reaching timeout limit
    del_instances = too_long(cleaner_df)
    pre_append = deleted_instances.copy()
    deleted_instances += too_long(cleaner_df)
    cleaner_df = delete_instances(cleaner_df, too_long(cleaner_df), 'Timeout too long')
    if DEBUG:
        print('too_long', len(set(del_instances)), len(set(del_instances)-set(pre_append)))
    # check if values across permutations are equal where there should be
    del_instances = permutations(cleaner_df, fico=fico)
    cleaner_df = delete_instances(cleaner_df, del_instances, 'Permutations not consistent')
    pre_append = deleted_instances.copy()
    deleted_instances += del_instances
    if DEBUG:
        print('perms', len(set(del_instances)), len(set(del_instances)-set(pre_append)))
    # check if values across permutations are equal where there should be
    cleaner_df, del_instances, diff_opt_perm, solved_once = perm_consistent(cleaner_df, fico=fico, DEBUG=DEBUG)
    pre_append = deleted_instances.copy()
    deleted_instances += del_instances
    if DEBUG:
        print('DiffOptPerms', len(set(diff_opt_perm)), len(set(diff_opt_perm)-set(pre_append)))
        print('SolvedOnce', len(set(solved_once)), len(set(solved_once)-set(pre_append)))

    # TODO: Ask timo if work<=100, if yes rework. Right now this does nothing
    cleaner_df, del_instances = tickst_du_richtig(cleaner_df)
    cleaner_df = delete_instances(cleaner_df, del_instances, 'Ticks > 100')
    pre_append = deleted_instances.copy()
    deleted_instances += del_instances
    if DEBUG:
        print('too_many_works', len(set(del_instances)), len(set(del_instances)-set(pre_append)))
    # if feature has the exact same entries for both rules, replace feature with static feature
    cleaner_df = equal_cols_to_static(cleaner_df)
    deleted_instances = set(deleted_instances)
    if DEBUG:
        print('Number of instances to be deleted:', len(deleted_instances), len(set(deleted_instances)))
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

