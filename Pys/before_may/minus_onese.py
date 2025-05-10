import pandas as pd
import matplotlib.pyplot as plt
from flatbuffers.compat import import_numpy
import numpy as np

df = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/Master_Excel/ready_to_ml.xlsx')
#print(df['Final solution time (cumulative) Mixed'].sum()-df['Final solution time (cumulative) Int'].sum())

def find_minus_nonneg_instances(df):
    #columns with 'ticks' or 'bound change' in their name are candidates for having -1 as entries for action not happening
    #ticks for propagation need to be recalculated-> they are deleted
    #Cmp columns also are irrelavant
    relevant_columns = df.columns[df.columns.str.contains('ticks|bound change')& ~df.columns.str.contains('Cmp|propagation')]

    minus_onese_df = df[relevant_columns.insert(0, 'Matrix Name')]
    """Neighbouring entries of relevant_columns form tupels, (1,2),(3,4),(5,6)..., where i want to find the instances with one entry
    of tupel is -1 and the other is not"""
    relevant_indices = []
    mixed_rule_negative = []
    int_rule_negative = []
    sb_sb_indices = [] #contains all ticks and bound change -1/not-1 pair indices for strong branching/spatial branching
    sb_ib_indices = [] #contains all ticks and bound change -1/not-1 pair indices for strong branching/integer branching

    #first column is matrix name, so we start with second column
    for i in range(1,len(relevant_columns),2):
        for index, row in minus_onese_df.iloc[:,i:i+2].iterrows():
            if (row.iloc[0]==-1) ^ (row.iloc[1]==-1):
                if index not in relevant_indices:
                    relevant_indices.append(index)
                if ('spatial' in row.index[0]):
                    if index not in sb_sb_indices:
                        sb_sb_indices.append(index)
                else:
                    if index not in sb_ib_indices:
                        sb_ib_indices.append(index)

    df['Absolute Time'] = df['Final solution time (cumulative) Mixed'] - df['Final solution time (cumulative) Int']
    columns_for_time_comparison_all = ['Matrix Name']+list(df.columns[df.columns.str.contains('ticks|bound change')& ~df.columns.str.contains('Cmp|propagation')]) +['Final solution time (cumulative) Mixed', 'Final solution time (cumulative) Int',
                                       'Cmp Final solution time (cumulative)', 'Absolute Time']
    columns_for_time_comparison_spatial_branch = [entry for entry in columns_for_time_comparison_all if "integer" not in entry]
    columns_for_time_comparison_integer_branch = [entry for entry in columns_for_time_comparison_all if "spatial" not in entry]

    # df[columns_for_time_comparison_all].iloc[relevant_indices].to_excel('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/Minus_Onese/time_neg_nonneg_all.xlsx', index=False)
    # df[columns_for_time_comparison_spatial_branch].iloc[sb_sb_indices].to_excel('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/Minus_Onese/time_neg_nonneg_spatial_branch.xlsx', index=False)
    # df[columns_for_time_comparison_integer_branch].iloc[sb_ib_indices].to_excel('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/Minus_Onese/time_neg_nonneg_integer_branch.xlsx', index=False)
    # #df[columns_for_time_comparison].to_excel('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/Minus_Onese/neg_nonneg_sb_sb.xlsx')
    # #ticks_sb_ib_df.to_excel('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/Minus_Onese/neg_nonneg__sb_ib.xlsx')

    interesting_minuses_df = df[columns_for_time_comparison_all].iloc[relevant_indices]

    interesting_minuses_df['Which rule is negative?'] = ''
    for i in sb_sb_indices:
        interesting_minuses_df.loc[i,'Which rule is negative?'] = 'IntFirst'
    interesting_minuses_df['Which rule is negative?'] = interesting_minuses_df['Which rule is negative?'].replace('','Mixed')


    interesting_minuses_df['Status'] = ''
    for i in relevant_indices:
        if df['Status Mixed'].loc[i] == 'Optimal':
            if df['Status Int'].loc[i] == 'Optimal':
                interesting_minuses_df.loc[i, 'Status'] = 'OptOpt'
            else:
                interesting_minuses_df.loc[i, 'Status'] = 'Mixed Opt/Int Time'
        else:
            if df['Status Int'].loc[i] == 'Optimal':
                interesting_minuses_df.loc[i, 'Status'] = 'Mixed Time/Int Opt'
            else:
                interesting_minuses_df.loc[i, 'Status'] = 'TimeTime'

    # interesting_minuses_df.to_excel('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/Minus_Onese/interesting_minuses.xlsx')
    return interesting_minuses_df, sb_sb_indices, sb_ib_indices

def histo_neg_nonneg(df, columns: list, number_bins=10):
    # Create a figure with subplots for each column dynamically
    fig, axs = plt.subplots(len(columns), 1, figsize=(8, 4 * len(columns)))  # n rows, 1 column

    # If there's only one column, axs is not an array, so we handle it separately
    if len(columns) == 1:
        axs = [axs]

    for i, col in enumerate(columns):
        between_zero_one = 0
        if 'Mixed' in col:
            axs[i].hist(df[col], bins=number_bins, color='green', alpha=1, label='Scaled')
            axs[i].set_title(col)
        else:

            axs[i].hist(df[col], bins=number_bins, color='blue', alpha=1, label='Scaled')
            axs[i].set_title(col)
        # Add legend to each subplot
        axs[i].legend()
    # Adjust layout
    plt.tight_layout()

    # Show the plots once all are created
    plt.show()

    # Close the plot to free up memory
    plt.close()

def scatter_neg_nonneg(interesting_minuses_df):
    tuple_names = ['Ticks strong branching for spatial branching', 'Ticks strong branching for integer branching', 'Bound Change strong branching for spatial branching', 'Bound Change strong branching for integer branching']
    # Define your tuples where each tuple consists of two columns from the DataFrame
    tuple1 = (interesting_minuses_df['Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed'], interesting_minuses_df['Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Int'])  # (x, y) pairs
    tuple2 = (interesting_minuses_df['Avg ticks for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed'], interesting_minuses_df['Avg ticks for solving strong branching LPs for integer branchings (not including infeasible ones) Int'])
    tuple3 = (interesting_minuses_df['Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Mixed'], interesting_minuses_df['Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Int'])
    tuple4 = (interesting_minuses_df['Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed'], interesting_minuses_df['Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Int'])

    # Function to filter points where at least one entry is negative
    def filter_negative_points(x, y):
        mask = (x < 0) | (y < 0)  # Get a mask for points where either x or y is negative
        return x[mask], y[mask]

    # Define a list of colors for each tuple
    colors = ['green', 'orange','magenta', 'red']
    # Plot each tuple with filtering and count
    for idx, (x, y) in enumerate([tuple1, tuple2, tuple3, tuple4], start=1):
        x_filtered, y_filtered = filter_negative_points(x, y)
        num_points = len(x_filtered)  # Count the number of plotted points

        plt.scatter(x_filtered, y_filtered, label=f'{tuple_names[idx-1]} (Count: {num_points})', color=colors[idx-1])

    # Set custom axis labels
    plt.xlabel('Mixed')
    plt.ylabel('Prefer Int')
    # Add a legend to indicate which color corresponds to which tuple
    plt.legend(fontsize='small')

    # Show the plot
    plt.show()

def time_comp(df, mixed_indices, intfirst_indices):
    total_time_mixed = df.loc[mixed_indices, 'Absolute Time'].sum()
    total_time_intfirst = df.loc[intfirst_indices, 'Absolute Time'].sum()
    negative = 0
    nonnegative = 0
    for index in intfirst_indices:
        if df.loc[index, 'Absolute Time'] < 0:
            negative += df.loc[index, 'Absolute Time']
        else:
            nonnegative += df.loc[index, 'Absolute Time']

inter_minuses_df, int_indices, mixed_indices = find_minus_nonneg_instances(df)
print(len(inter_minuses_df))
time_comp(inter_minuses_df, mixed_indices, int_indices)
#histo_neg_nonneg(inter_minuses_df, inter_minuses_df.columns[-1:] )
scatter_neg_nonneg(inter_minuses_df)

















"""Create a spreadsheet with interesting properties of dataset"""

total_number_instances = len(df)
number_neg_nonneg_instances = len(inter_minuses_df)
cmp_time_around_zero_total = ((df['Cmp Final solution time (cumulative)'] >= -0.5) & (df['Cmp Final solution time (cumulative)'] <= 0.5)).sum()
cmp_time_around_zero_neg_nonneg = ((inter_minuses_df['Cmp Final solution time (cumulative)'] >= -0.5) & (inter_minuses_df['Cmp Final solution time (cumulative)'] <= 0.5)).sum()

observation_df = pd.DataFrame(index=range(4), columns=['Value', 'Share'])
observation_df[['Value', 'Share']] = observation_df[['Value', 'Share']].astype(float)

observation_df.index = ['Total number Instances', 'Instances Cmp Time in [-0.5, 0.5] ', 'Number neg/nonneg Instances', 'Instances neg/nonneg Cmp Time in [-0.5, 0.5]']
observation_df.iloc[0,0] = total_number_instances
observation_df.iloc[1,0] = cmp_time_around_zero_total
observation_df.iloc[1,1] = np.round((cmp_time_around_zero_total/total_number_instances)*100, 2)
observation_df.iloc[2,0] = number_neg_nonneg_instances
observation_df.iloc[3,0] = cmp_time_around_zero_neg_nonneg
observation_df.iloc[3,1] = np.round((cmp_time_around_zero_neg_nonneg/number_neg_nonneg_instances)*100, 2)
observation_df = observation_df.replace(np.nan, 1)

# observation_df.to_excel('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/Minus_Onese/observations_minus_onese.xlsx')

print(df.columns)