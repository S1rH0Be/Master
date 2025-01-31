import os
from scipy.stats import gmean
import pandas as pd
from pandas import Series
import numpy as np
# plotting
import matplotlib.pyplot as plt

def shifted_geometric_mean(values, shift):
    values = np.array(values)
    # Shift the values by the constant and check for any negative values after shifting
    shifted_values = values + shift
    if shifted_values.dtype == 'object':
        # Attempt to convert to float
        shifted_values = shifted_values.astype(float)

    shifted_values_log = np.log(shifted_values)  # Step 1: Log of each element in shifted_values
    log_mean = np.mean(shifted_values_log)  # Step 2: Compute the mean of the log values
    geo_mean = np.exp(log_mean) - shift
    geo_mean = np.round(geo_mean, 6)
    return geo_mean

def plot_column_gmeans(df, model_name:str):
    """
    Calculates the variance of each column in a DataFrame and plots it in a bar graph.

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    None
    """
    # Calculate geometric mean for each column
    geometric_means = df.apply(lambda col: gmean(col[col > 0]), axis=0)

    # Plot the variance as a bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(geometric_means.index, geometric_means.values, color='skyblue')

    # Add labels and title
    plt.xlabel('Runs')
    plt.ylabel('Variance')
    plt.title(f'GeoMean of Feature Importance for each {model_name} run')
    plt.tight_layout()
    # Remove x-axis ticks
    plt.xticks([])
    # Show the plot
    plt.show()

def compute_potential_time_saves(df):
    bins = [0, 0.5, 2, 4, 10, 100, 1000, float('inf')]  # Define bin edges
    intervals = ["(0,0.5)", "[0.5,2)", "[2,4)", "[4,10)", "[10,100)", "[100,1000)", "[1000, inf)"]  # Corresponding labels

    df_filtered = df[df['Cmp Final solution time (cumulative)'] != 0].copy()

    abs_cmp_time = abs(df_filtered['Cmp Final solution time (cumulative)'])
    df_filtered['interval'] = pd.cut(abs_cmp_time, bins=bins, labels=intervals, right=False)  # Categorize into bins
    # Compute sum of 'Pot Time Save' for each interval
    grouped_sums = df_filtered.groupby('interval')['Pot Time Save'].sum()
    # Ensure all bins are represented (fill missing ones with 0)
    interval_sums = grouped_sums.reindex(intervals, fill_value=0).tolist()

    total_pot_save = df_filtered['Pot Time Save'].sum()  # Total potential save

    instance_counts = df_filtered['interval'].value_counts().reindex(intervals, fill_value=0).tolist()
    total_instances = len(df_filtered)
    instances = [total_instances] + instance_counts

    values = [total_pot_save] + interval_sums  # Combine total with bin-wise sums
    intervals.insert(0, 'Total')
    return values, intervals, instances

def plot_time_save(data):
    values, names, number_of_instances_in_intervall = compute_potential_time_saves(data)
    plt.figure(figsize=(10, 7))
    bars = plt.bar(names, values, color=['magenta', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue'])
    # Add instance count above each bar
    for bar, num_instances in zip(bars, number_of_instances_in_intervall):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(num_instances),
                 ha='center', va='bottom', fontsize=12, color='black')
    plt.xticks(names, rotation=45, ha='right')
    plt.title('Potential Time Save per Intervall')
    plt.show()

def plot_sgm_feature_importance(df, title):
    shift = -(df.iloc[:, 1:].min().min()) + 1
    df["ShiftedGeometricMean"] = df.iloc[:, 1:].apply(lambda row: shifted_geometric_mean(row, shift), axis=1)
    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(df['Feature'], df['ShiftedGeometricMean'], color='skyblue')

    # Add labels and title
    plt.ylabel('SGM Feature Importance')
    plt.title(title)
    plt.xticks([])
    # plt.ylim([0.8, 1.06])
    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_sgm_accuracy(accuracy_df, title):
    shift=0
    lin_mean = shifted_geometric_mean(accuracy_df['Accuracy'][accuracy_df['Model'] == 'LinearRegression'], shift)
    for_mean = shifted_geometric_mean(accuracy_df['Accuracy'][accuracy_df['Model'] == 'RandomForest'], shift)
    lin_ex_mean = shifted_geometric_mean(accuracy_df['Extreme Accuracy'][accuracy_df['Model'] == 'LinearRegression'], shift)
    for_ex_mean = shifted_geometric_mean(accuracy_df['Extreme Accuracy'][accuracy_df['Model'] == 'RandomForest'], shift)
    values_seperated = [lin_mean, lin_ex_mean, for_mean, for_ex_mean]

    values_seperated_names = ['LinReg Total', 'LinReg Extreme', 'RandomForest Total', 'RandomForest Extreme']
    colors = ['orange', 'orange', 'limegreen', 'limegreen']
    plt.figure(figsize=(10, 6))
    plt.bar(values_seperated_names, values_seperated, color=colors)
    plt.ylim([35,  90])
    # Add labels and title
    plt.ylabel('SGM Accuracy')
    plt.title(title)
    plt.tight_layout()
    # Remove x-axis ticks
    plt.xticks(values_seperated_names)
    # Show the plot
    plt.show()

def box_plot(values:Series, title:str):
    plt.boxplot(values, vert=True, patch_artist=True)
    plt.title(title)
    plt.ylabel("Values")
    plt.show()
    plt.close()

def plot_sgm(df, title:str):
    lin_df = df[[col for col in df.columns if 'Linear' in col]]
    for_df = df[[col for col in df.columns if 'Forest' in col]]

    sgm_lin = lin_df.iloc[:, 1:].apply(lambda row: shifted_geometric_mean(row, 0), axis=1)
    sgm_for = for_df.iloc[:, 1:].apply(lambda row: shifted_geometric_mean(row, 0), axis=1)
    sgm_values = [sgm_lin.iloc[0], sgm_lin.iloc[1], sgm_for.iloc[1], sgm_for.iloc[2]]

    bar_names = ['Mixed', 'Linear', 'Forest', 'Virtual Best']

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(bar_names, sgm_values, color=['lightblue', 'orange', 'limegreen', 'lightblue'])
    # Add labels and title
    plt.ylabel('Shifted Geometric Mean', fontsize=12)
    plt.ylim(0.5,1.1)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    # Show the plot
    plt.show()

def plot_sgm_single_df(df, title:str):
    bar_names = df.iloc[:, 0]
    # Calculate the mean of each row (excluding the first column)
    values = df.iloc[:, 1]
    # print(values[1], values[5])
    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(bar_names, values, color=['lightblue', 'orange', '#FF8C00', 'limegreen', 'lightgreen', 'lightblue'])
    # Add labels and title
    plt.ylabel('Shifted Geometric Mean', fontsize=12)
    plt.ylim(0.5,1.1)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    # Show the plot
    plt.show()

def plot_sgm_relative_to_mixed(df, title:str):
    lin_df = df[[col for col in df.columns if 'Linear' in col]]
    for_df = df[[col for col in df.columns if 'Forest' in col]]

    sgm_lin = lin_df.iloc[:, 1:].apply(lambda row: shifted_geometric_mean(row, 0), axis=1)
    sgm_for = for_df.iloc[:, 1:].apply(lambda row: shifted_geometric_mean(row, 0), axis=1)
    sgm_mixed = sgm_lin.iloc[0]

    relative_values = [1.0, np.round(sgm_lin.iloc[1]/sgm_mixed, 2), np.round(sgm_for.iloc[1]/sgm_mixed, 2), np.round(sgm_for.iloc[2]/sgm_mixed, 2)]
    print(relative_values)
    bar_names = ['Mixed', 'Linear', 'Forest', 'Virtual Best']

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(bar_names, relative_values, color=['lightblue', 'orange', 'limegreen', 'lightblue'])
    # Add labels and title
    plt.ylabel('Shifted Geometric Mean', fontsize=12)
    plt.ylim(0.5, 1.1)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    # Show the plot
    plt.show()

def plot_sgm_run_time_single_df(sgm_df1, sgm_df2,  title):

    def calc_sgm(sgm_df):
        lin_df = sgm_df[[col for col in sgm_df.columns if 'Linear' in col]]
        for_df = sgm_df[[col for col in sgm_df.columns if 'Forest' in col]]

        sgm_lin = lin_df.iloc[:, 1:].apply(lambda row: shifted_geometric_mean(row, 0), axis=1)
        sgm_for = for_df.iloc[:, 1:].apply(lambda row: shifted_geometric_mean(row, 0), axis=1)
        sgm_values = [sgm_lin.iloc[1], sgm_for.iloc[1], sgm_for.iloc[2]]
        return sgm_values

    sgm1_values = calc_sgm(sgm_df1)
    sgm2_values = calc_sgm(sgm_df2)

    sgm_combined_values = [1, sgm1_values[0], sgm2_values[0], sgm1_values[1], sgm2_values[1], sgm1_values[2]]
    names = ['Mixed', 'Linear All', 'Linear Top3', 'Random Forest All', 'Forest Top3', 'Virtual Best']

    sgm_combined_df = pd.DataFrame({'Regressor': names, 'Shifted Geometric Mean': sgm_combined_values})
    plot_sgm_single_df(sgm_combined_df, title)

def create_run_time_bars(df=None, title=None):
    if df is None:
        all_sgm_df = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/PräsiTristan/Präsi/CSV/all_unscaled_sgm_1000.xlsx')
        t3_sgm_df = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/PräsiTristan/Präsi/CSV/t3_unscaled_sgm_1000.xlsx')
        plot_sgm_run_time_single_df(all_sgm_df, t3_sgm_df, 'SGM Run Time Comparison on unscaled Label')
        all_logged = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/NoCmpFeats/Tester/SGM/sgm_logged_t18_both_below_1000_hundred_seeds_28_01.xlsx')
        t3_logged = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/NoCmpFeats/Tester/SGM/sgm_logged_t3_both_below_1000_hundred_seeds_29_01.xlsx')
        plot_sgm_run_time_single_df(all_logged, t3_logged, 'SGM Run Time Comparison on logged Label')
    else:
        plot_sgm_run_time_single_df(df, df, title)

def create_accuracy_bars(df=None, title=None):
    if df is None:
        # unscaled label
        acc_t18_unscaled_label = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/NoCmpFeats/Tester/Accuracy/unscaled/unscaled_t18_both_below_1000_hundred_seeds_28_01.xlsx')
        plot_sgm_accuracy(acc_t18_unscaled_label, 'Accuracy on All Features and unscaled Label')
        acc_t3_unscaled_label = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/NoCmpFeats/Tester/Accuracy/unscaled/unscaled_t3_both_below_1000_hundred_seeds_28_01.xlsx')
        plot_sgm_accuracy(acc_t3_unscaled_label, 'Accuracy on Top3 Features and unscaled Label')
        # logged label
        acc_t18_logged_label = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/NoCmpFeats/Tester/Accuracy/logged/logged_t18_both_below_1000_hundred_seeds_28_01.xlsx')
        plot_sgm_accuracy(acc_t18_logged_label, 'Accuracy on All Features and logged Label')
        acc_t3_logged_label = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/NoCmpFeats/Tester/Accuracy/logged/logged_t3_both_below_1000_hundred_seeds_29_01.xlsx')
        plot_sgm_accuracy(acc_t3_logged_label, 'Accuracy on Top3 Features and logged Label')
    else:
        plot_sgm_accuracy(df, title)

def process_excel_files(directory, function1):
    """
    Reads all .xlsx files in a given directory and applies the assign_points function.

    Parameters:
    directory (str): Path to the directory containing .xlsx files.

    Returns:
    dict: A dictionary where keys are filenames and values are DataFrames with assigned points.
    """
    results = {}  # Store processed DataFrames

    for file in os.listdir(directory):
        if file.endswith(".xlsx"):  # Process only .xlsx files
            file_path = os.path.join(directory, file)

            try:
                df = pd.read_excel(file_path)  # Read the Excel file
                processed_df = function1(df)  # Apply your function
                results[file] = processed_df  # Store the result
                print(f"Processed: {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")

    return results

def list_of_dataframes(directory):
    dataframes = []
    for file in os.listdir(directory):
        if file.endswith(".xlsx"):  # Process only .xlsx files
            file_path = os.path.join(directory, file)
            dataframes.append(pd.read_excel(file_path))
    return dataframes

accuracy_df = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/Logged/Accuracy/logged_t18_lin_optimized_both_below_1000_hundred_seeds_1_31_01.xlsx')
print(len(accuracy_df))
run_time_df = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/Logged/SGM/sgm_logged_t18_lin_optimized_both_below_1000_hundred_seeds_1_31_01.xlsx')

