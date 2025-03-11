import os
from scipy.stats import gmean
import pandas as pd
from pandas import Series
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
# custim functions
from das_ist_die_richtige_regression import scale_label
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
    bars = plt.bar(values_seperated_names, values_seperated, color=colors)
    plt.ylim([35,  105])
    # Add labels and title
    plt.ylabel('SGM Accuracy')
    plt.title(title)
    plt.tight_layout()
    # Remove x-axis ticks
    plt.xticks(values_seperated_names)
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}', ha='center', fontsize=12)
    # Show the plot
    plt.show()

def box_plot(values:Series, title:str):
    plt.boxplot(values, vert=True, patch_artist=True)
    plt.title(title)
    plt.ylabel("Values")
    plt.show()
    plt.close()

def comp_box_plot(values, title):
    for value in values:
        plt.boxplot(values, vert=True, patch_artist=True)
    plt.title(title)
    plt.ylabel("Values")
    plt.show()
    plt.close()

def plot_sgm_relative_to_mixed(df, title:str):
    lin_df = df[[col for col in df.columns if 'Linear' in col]]
    for_df = df[[col for col in df.columns if 'Forest' in col]]

    sgm_lin = lin_df.iloc[:, 1:].apply(lambda row: shifted_geometric_mean(row, 0), axis=1)
    sgm_for = for_df.iloc[:, 1:].apply(lambda row: shifted_geometric_mean(row, 0), axis=1)
    sgm_mixed = sgm_lin.iloc[0]

    relative_values = [1.0, np.round(sgm_lin.iloc[1]/sgm_mixed, 6), np.round(sgm_for.iloc[1]/sgm_mixed, 6), np.round(sgm_for.iloc[2]/sgm_mixed, 6)]

    bar_names = ['Mixed', 'Linear', 'Forest', 'Virtual Best']
    colors = ['lightblue', 'orange', 'limegreen', 'lightblue']

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(bar_names, relative_values, color=colors)
    # Add labels and title
    plt.ylabel('Shifted Geometric Mean', fontsize=12)
    plt.ylim(0.55, 1.05)
    plt.title(title, fontsize=14)
    plt.xticks(bar_names)
    plt.tight_layout()
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}', ha='center', fontsize=12)
    # Show the plot
    plt.show()

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

def feature_histo(df, columns: list, number_bins=10):
    """
    Create histograms for specified columns in a DataFrame, focusing only on values in (0, 1).
    Args:
        df: The DataFrame containing data.
        columns: List of column names to plot histograms for.
        number_bins: Number of bins for the histograms.
    """
    # Create a figure with subplots for each column dynamically
    fig, axs = plt.subplots(len(columns), 1, figsize=(8, 4 * len(columns)))  # n rows, 1 column

    # If there's only one column, axs is not an array, so we handle it separately
    if len(columns) == 1:
        axs = [axs]

    for i, col in enumerate(columns):
        # Filter values in (0, 1)
        # if df.values.min().min()>=0 and df.values.max().max()<=1:
        #     filtered_data = df[col][(df[col] >= 0) & (df[col] <= 1)]
        # else:
        filtered_data = df[col]
        # Plot histogram with color distinction
        color = 'orange' if 'Mixed' in col else 'lightblue'
        axs[i].hist(filtered_data, bins=number_bins, color=color, alpha=1)#, label=f'Filtered ({len(filtered_data)} points)')
        axs[i].set_title(f'{col}')

    # Adjust layout
    plt.tight_layout()
    # Show the plots once all are created
    plt.show()
    # Close the plot to free up memory
    plt.close()

def label_histo(label_series, number_bins=10):

    plt.hist(label_series, bins=number_bins, color='magenta')
    plt.title('Cmp Final solution time (cumulative)')

    # Adjust layout
    plt.tight_layout()
    # Show the plots once all are created
    plt.show()
    # Close the plot to free up memory
    plt.close()

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

def imputation(df):
    imputated_features = df.copy()
    imputated_features = imputated_features.replace(-1, np.nan)
    # Initialize the imputer with the median strategy
    imputer = SimpleImputer(strategy="median")
    # Fit and transform the data
    imputated_features = pd.DataFrame(imputer.fit_transform(imputated_features), columns=imputated_features.columns)
    return imputated_features

def scaling(feat_df, label_series):
    qt = QuantileTransformer(n_quantiles=100, output_distribution="normal", random_state=42)
    # Fit and transform the data
    feature_df_transformed = qt.fit_transform(feat_df)
    feature_df_transformed = pd.DataFrame(feature_df_transformed, columns=feature_df.columns)
    logged_label = scale_label(label_series)
    return feature_df_transformed, logged_label

data = pd.read_excel("/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/base_data_24_01.xlsx")
feature_df = pd.read_excel("/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/base_feats_no_cmp_24_01.xlsx")
label = data['Cmp Final solution time (cumulative)'].copy()

imputed_feature = imputation(feature_df)
scaled_feature, logged_label = scaling(imputed_feature, label)

imputed_feature.to_excel("/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/base_feats_no_cmp_imputed.xlsx", index=False)
scaled_feature.to_excel("/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/base_feats_no_cmp_imputed_Scaled.xlsx", index=False)
logged_label.to_excel("/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/label_scaled.xlsx", index=False)

for col in feature_df.columns:
    box_plot(scaled_feature[col], col)