import pandas as pd
import numpy as np

from typing import List, Union
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def create_time_df(df):
    time_data = df[['Final solution time (cumulative) Int', 'Final solution time (cumulative) Mixed', 'Cmp Final solution time (cumulative) Predicted']].copy()
    time_data.loc[:, 'Virtual Best'] = 0.0
    for ind in time_data.index:
        time_data.loc[ind, 'Virtual Best'] = min(df.loc[ind, 'Final solution time (cumulative) Mixed'], time_data.loc[ind, 'Final solution time (cumulative) Int'])
    return time_data

def bar_plot_time(tuples:List[tuple], title: str) -> None:
    #tuples: (Name of column, float)
    labels = []
    values = []

    for tup in tuples:
        if 'Mixed' in tup[0]:
            labels.append('Always Mixed')
        elif 'Int' in tup[0]:
            labels.append('Always Int')
        elif 'Predicted' in tup[0]:
            labels.append('Predicted')
        else:
            labels.append(tup[0])

        values.append(tup[1])

    plt.figure(figsize=(8, 5))
    # Sort the data by values in descending order
    sorted_data = sorted(zip(values, labels), reverse=True)
    sorted_values, sorted_labels = zip(*sorted_data)

    num_bars = len(sorted_labels)
    colors = cm.get_cmap('viridis', num_bars)(range(num_bars))

    #bars = plt.bar(sorted_labels, sorted_values, colors)
    bars = plt.bar(sorted_labels, sorted_values, color=['blue', 'green', 'turquoise', 'red'])
    plt.title(title)
    plt.ylim(min(0, min(values))*1.1, max(values)*1.1)  # Set y-axis limits for visibility
    plt.xticks(rotation=45, fontsize=6)
    # Create custom legend entries with value annotations
    legend_labels = [f"{label}: {value}" for label, value in zip(sorted_labels, sorted_values)]
    plt.legend(bars, legend_labels, title="Values")
    # Display the plot
    plt.show()
    plt.close()

def shifted_geometric_mean(values, shift):
    """
    Calculate the shifted geometric mean of a list or array of values.

    Parameters:
    values (array-like): The list or array of values for which to calculate the shifted geometric mean.
    shift (float): The constant to add to each value (shift) before calculating the geometric mean.

    Returns:
    numpy.float64: The shifted geometric mean.
    """

    # Convert values to a NumPy array for compatibility
    values = np.array(values)
    # Shift the values by the constant and check for any negative values after shifting
    shifted_values = values + shift
    if shifted_values.dtype == 'object':
        # Attempt to convert to float
        shifted_values = shifted_values.astype(float)

    shifted_values_log = np.log(shifted_values)  # Step 1: Log of each element in shifted_values
    log_mean = np.mean(shifted_values_log)  # Step 2: Compute the mean of the log values
    geo_mean = np.exp(log_mean) - shift
    geo_mean = np.round(geo_mean, 2)
    return geo_mean

def relative_to(df, reference:str, other_values:List[str]):
    """returns a list of tuples with label name and corresponding relative value
    reference is in ['Mixed','Int','VBS']"""
    tuples = []
    ref_val = np.round(df[reference].sum(), 2)
    # Check if result is a scalar or a DataFrame/Series
    if isinstance(ref_val, (pd.DataFrame, pd.Series)):
        ref_val = float(ref_val.iloc[0])  # Convert the first element to float
    else:
        ref_val = float(ref_val)  # Directly convert to float if it's already a scalar

    for i in range(len(other_values)):
        value = df[other_values[i]].sum()
        tuples.append((other_values[i], np.round(value/ref_val, 2)))
    return tuples

def abs_time(df, values:List[str]):
    """Input: columns names to compare the total sum of each
    Return: a list of tuples with label name and corresponding relative value"""
    tuples = []
    ref_val = np.round(df[reference].sum(), 2)
    # Check if result is a scalar or a DataFrame/Series
    if isinstance(ref_val, (pd.DataFrame, pd.Series)):
        ref_val = float(ref_val.iloc[0])  # Convert the first element to float
    else:
        ref_val = float(ref_val)  # Directly convert to float if it's already a scalar

    for i in range(len(values)):
            value = df[other_values[i]].sum()
            tuples.append((other_values[i], np.round(value, 2)))
    return tuples

def abs_diff_to(df, reference:str, other_values:List[str]):
    """Input: columns names to compare the total sum of each
    Return: a list of tuples with label name and corresponding relative value"""
    tuples = []
    ref_val = np.round(df[reference].sum(), 2)
    # Check if result is a scalar or a DataFrame/Series
    if isinstance(ref_val, (pd.DataFrame, pd.Series)):
        ref_val = float(ref_val.iloc[0])  # Convert the first element to float
    else:
        ref_val = float(ref_val)  # Directly convert to float if it's already a scalar


    for i in range(len(other_values)):
            value = df[other_values[i]].sum()
            tuples.append((other_values[i], np.round(value-ref_val, 2)))
    return tuples

def geo_mean(df, values:List[str], shift):
    """returns a list of tuples with label name and corresponding relative value"""
    tuples = []
    for i in range(len(values)):
        value = shifted_geometric_mean(df[values[i]], shift)
        tuples.append((values[i], np.round(value, 2)))

    return tuples

def geo_mean_relative_to(reference:str, geo_means_tuple):
    # create a df for shifted geometric means to be able to calculate relative values
    # Extract column names and values from geo_means
    column_names = [tup[0] for tup in geo_means_tuple]
    values = [tup[1] for tup in geo_means_tuple]
    # Create a DataFrame with one row using these columns and values
    geo_mean_df = pd.DataFrame([values], columns=column_names)
    relative_geo_mean = relative_to(geo_mean_df, reference, column_names)
    return relative_geo_mean

def compare_time(df, reference:str, plot=True, plot_all=True, title_add_on=''):
    """This Function evaluates the whole prediction.
    y_pred_test is a list of tuple containing the predicted value and actual value of Model """

    possible_references = ['Mixed', 'Int', 'Predicted', 'Virtual Best']
    if reference not in possible_references:
        raise ValueError(f"Reference {reference} is not in {possible_references}")

    #time_df = create_time_df(df)
    time_df = df
    #defines column which should be compared against
    reference_col = time_df.filter(regex=r'\b'+reference+r'\b', axis=1).columns

    relative_tuples = relative_to(time_df, reference_col, time_df.columns)
    diff_tuples = abs_diff_to(time_df, reference_col, time_df.columns)
    geo_means = geo_mean(time_df, time_df.columns, 0.5)
    relative_geo_mean = geo_mean_relative_to(reference_col, geo_means)
    #all values in the following dict are empty Series
    to_be_plotted = {'Difference to Mixed':diff_tuples, 'Relative GeoMean to Mixed(+0.5)':relative_geo_mean}

    if plot == True:
        if plot_all==False:
            bar_plot_time(to_be_plotted['Relative GeoMean to Mixed(+0.5)'], title="Relative ShiftGeoMeanTime to Mixed "+ title_add_on)
        else:
            for name, lst in to_be_plotted.items():
                bar_plot_time(lst, name+" "+ title_add_on)

    return to_be_plotted

# df = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/Master_Excel/ready_to_ml.xlsx')
df = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/data_raw_august.xlsx')
# df['Final solution time (cumulative) Predicted'] = 0.0
# cmp_time = compare_time(df, 'Mixed')
fin_time_df = df[['Final_solution_time_(cumulative)_Mixed', 'Final_solution_time_(cumulative)_Int']]
shifted_times = []
for i in ['Final_solution_time_(cumulative)_Mixed', 'Final_solution_time_(cumulative)_Int']:
    shifted_times.append(shifted_geometric_mean(df[i], 0.5))
