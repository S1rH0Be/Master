from scipy.stats import gmean
from typing import List
import pandas as pd
from pandas import Series
import numpy as np
# plotting
import matplotlib.pyplot as plt



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

def plot_column_variance(df, model_name:str):
    """
    Calculates the variance of each column in a DataFrame and plots it in a bar graph.

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    None
    """
    # Calculate geometric mean for each column
    variance = df.var()

    # Plot the variance as a bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(variance.index, variance.values, color='skyblue')

    # Add labels and title
    plt.xlabel('Runs')
    plt.ylabel('Variance')
    plt.title(f'Variance of Feature Importance for each {model_name} run')
    plt.tight_layout()
    # Remove x-axis ticks
    plt.xticks([])
    # Show the plot
    plt.show()

def plot_mean_accuracy(df, title):
    values = [df['Accuracy'].mean(), df['Accuracy'].var(), df['Extreme Accuracy'].mean(), df['Extreme Accuracy'].var(), df['r2'].mean()*100, df['r2'].median()*100]
    value_names = ['Mean Accuracy', 'Accuracy Variance', 'Mean Extreme Accuracy', 'Extreme Accuracy Variance', 'r2 Mean', 'r2 Median']

    colors = ['magenta', 'blue', 'green', 'red', 'cyan', 'turquoise']
    plt.figure(figsize=(10, 6))
    plt.bar(value_names, values, color=colors[:len(values)])
    plt.ylim([min(50, min(values))*0.9,  max(values)*1.1])
    # Add labels and title
    plt.xlabel('Nada')
    plt.ylabel('Values in Percent')
    plt.title(title)
    plt.tight_layout()
    # Remove x-axis ticks
    plt.xticks(value_names)
    # Show the plot
    plt.show()

def plot_r2_accuracy(regressors:List[str]):
    acc_total = pd.read_excel(f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/TestCombinations/label_logged_below_1000_twenty_seeds_24_01.xlsx')
    acc_names = ['Total', 'None', 'Standard', 'Power']
    for regressor in regressors:
        if regressor == 'LinearRegression':
            lin_acc_total = acc_total[acc_total['Model'] == 'LinearRegression']
            lin_acc_none = lin_acc_total[lin_acc_total['Scaling'].isna()==True]
            lin_acc_standard = lin_acc_total[lin_acc_total['Scaling'] == 'StandardScaler']
            lin_acc_power = lin_acc_total[lin_acc_total['Scaling'] == 'PowerTransformer']

            lin_accs_by_scaling_list = [lin_acc_total, lin_acc_none, lin_acc_standard, lin_acc_power]
            count = 0
            for results in lin_accs_by_scaling_list:
                plot_mean_accuracy(results, acc_names[count])
                count += 1

        elif regressor == 'RandomForest':
            for_acc_total = acc_total[acc_total['Model'] == 'RandomForest']
            for_acc_none = for_acc_total[for_acc_total['Scaling'].isna()==True]
            for_acc_standard = for_acc_total[for_acc_total['Scaling'] == 'StandardScaler']
            for_acc_power = for_acc_total[for_acc_total['Scaling'] == 'PowerTransformer']

            for_accs_by_scaling_list = [for_acc_total, for_acc_none, for_acc_standard, for_acc_power]

            count = 0
            for results in for_accs_by_scaling_list:
                plot_mean_accuracy(results, acc_names[count])
                count+=1
        else:
            print(f'{regressor} is an invalid regressor! Try string LinearRegression or RandomForest')

def box_plot(values:Series, title:str):
    plt.boxplot(values, vert=True, patch_artist=True)
    plt.title(title)
    plt.ylabel("Values")
    plt.show()
    plt.close()


data = pd.read_excel(f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/NoCmpFeats/base_data_24_01.xlsx').drop(columns='Matrix Name')

time_save_and_label = data[['Cmp Final solution time (cumulative)', 'Pot Time Save']]
time_save_and_label_only_relevant = time_save_and_label[time_save_and_label['Cmp Final solution time (cumulative)']!=0]
time_save_and_label_only_relevant = time_save_and_label_only_relevant.sort_values(
                                    by='Cmp Final solution time (cumulative)', key=lambda x: abs(x))

total_pot_save = time_save_and_label_only_relevant['Pot Time Save'].sum()
irrelavent_range = time_save_and_label_only_relevant['Pot Time Save'][abs(time_save_and_label_only_relevant['Cmp Final solution time (cumulative)'])<0.5].sum()
irrelevant_til_two = time_save_and_label_only_relevant['Pot Time Save'][(abs(time_save_and_label_only_relevant['Cmp Final solution time (cumulative)'])>=0.5)&(abs(time_save_and_label_only_relevant['Cmp Final solution time (cumulative)'])<2)].sum()
two_til_four = time_save_and_label_only_relevant['Pot Time Save'][(abs(time_save_and_label_only_relevant['Cmp Final solution time (cumulative)'])>=2)&(abs(time_save_and_label_only_relevant['Cmp Final solution time (cumulative)'])<4)].sum()
four_til_ten = time_save_and_label_only_relevant['Pot Time Save'][(abs(time_save_and_label_only_relevant['Cmp Final solution time (cumulative)'])>=4)&(abs(time_save_and_label_only_relevant['Cmp Final solution time (cumulative)'])<10)].sum()
ten_til_hundred = time_save_and_label_only_relevant['Pot Time Save'][(abs(time_save_and_label_only_relevant['Cmp Final solution time (cumulative)'])>=10)&(abs(time_save_and_label_only_relevant['Cmp Final solution time (cumulative)'])<100)].sum()
hundred_til_thousand = time_save_and_label_only_relevant['Pot Time Save'][(abs(time_save_and_label_only_relevant['Cmp Final solution time (cumulative)'])>=100)&(abs(time_save_and_label_only_relevant['Cmp Final solution time (cumulative)'])<1000)].sum()
above_thousand = time_save_and_label_only_relevant['Pot Time Save'][abs(time_save_and_label_only_relevant['Cmp Final solution time (cumulative)'])>=1000]

between = irrelevant_til_two+two_til_four
four_til_hundred = four_til_ten+ten_til_hundred+hundred_til_thousand+above_thousand
number_relevant = len(time_save_and_label_only_relevant['Pot Time Save'][abs(time_save_and_label_only_relevant['Cmp Final solution time (cumulative)'])>=10])

# values = [total_pot_save, irrelavent_range, irrelevant_til_two, two_til_four, four_til_ten, ten_til_hundred, hundred_til_thousand, above_thousand]#, between, four_til_hundred]
# names =  ['total', '(0,0.5)', '[0.5,2)', '[2,4)', '[4,10)', '\n[10,100)', '[100,1000)', '\n[1000, inf)']#, '[0,4)\n247', '[4, inf)\n51']
# number_of_instances_in_intervall =  [298, 140, 78, 10, 19, 29, 21, 1]
# plt.bar(names, values, color=['magenta', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue'])
#
# plt.title('Potential Time Save per Intervall')
# plt.show()

# Data
values = [irrelavent_range, irrelevant_til_two, two_til_four, four_til_ten, ten_til_hundred, hundred_til_thousand, above_thousand]
names = ['(0,0.5)', '[0.5,2)', '[2,4)', '[4,10)', '\n[10,100)', '[100,1000)', '\n[1000, inf)']
number_of_instances_in_intervall = [140, 78, 10, 19, 29, 21, 1]

# Calculate the ratio of value to number of instances in the interval
values_ratio = [v / n for v, n in zip(values, number_of_instances_in_intervall)]

# Create bar positions (adjusted for side-by-side bars)
x = np.arange(len(names))

# Width of the bars
bar_width = 0.35

# Plot the bars
plt.bar(x - bar_width / 2, values, bar_width, label='Total Save', color='blue')  # First set of bars
plt.bar(x + bar_width / 2, values_ratio, bar_width, label='(Save on Intervall)/#Instances', color='magenta')  # Second set of bars

# Add title and labels
plt.title('Save per Instance in Intervall')
plt.xticks(x, names, rotation=45, ha='right')  # Set x-ticks with names and rotate labels for readability
plt.legend()

# Layout adjustments
plt.tight_layout()

# Show plot
plt.show()