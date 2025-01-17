import numpy as np
import pandas as pd
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

def plot_sgms(df, labels, title: str) -> None:
    #tuples: (Name of column, float)
    values =df['SGMs']
    # Determine bar colors based on conditions
    bar_colors = ['turquoise','magenta','turquoise']+['green' if value >= 0.8 else 'red' if value <= 0.6 else 'blue' for value in values[3:7]]

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=bar_colors)
    plt.title(title)
    plt.ylim(min(0.5, min(values)*0.9), max(values)*1.01)  # Set y-axis limits for visibility
    plt.xticks(rotation=45, fontsize=6)
    # Create custom legend entries with value annotations
    # legend_labels = [f"{label}: {value}" for label, value in zip(labels, values)]
    # plt.legend(bars, legend_labels, title="Values")
    # Display the plot
    plt.show()
    plt.close()

accs_and_sgm = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/50_collected_shifted_geo_means_with_accuracies_17_01.xlsx')

sgms = [shifted_geometric_mean(row.iloc[1:], 0.001) for index, row in accs_and_sgm.iterrows()]
# accuracies in prozent-> divide by 100 to have same scale as relative sgms
sgms[3:7] = [np.round(sgms[i]/100,6) for i in range(3,7)]
global_sgms = pd.DataFrame({'Origin': accs_and_sgm.iloc[:,0], 'SGMs':sgms})
plot_sgms(global_sgms, global_sgms['Origin'],'Global SGMs on Top4global')

linear_columns = [col for col in accs_and_sgm.columns if 'Linear' in col]
linear_accs_and_sgms = accs_and_sgm[linear_columns].copy()
linear_sgms = [shifted_geometric_mean(row.iloc[1:], 0.001) for index, row in linear_accs_and_sgms.iterrows()]
linear_sgms[3:7] = [np.round(linear_sgms[i]/100,6) for i in range(3,7)]
linear_sgms_df = pd.DataFrame({'Origin': accs_and_sgm.iloc[:,0], 'SGMs':linear_sgms})
plot_sgms(linear_sgms_df, linear_sgms_df['Origin'],'Linear SGMs')


forest_columns = [col for col in accs_and_sgm.columns if 'Forest' in col]
forest_accs_and_sgms = accs_and_sgm[forest_columns].copy()
forest_accs_and_sgms = forest_accs_and_sgms
forest_sgms = [shifted_geometric_mean(row.iloc[1:], 0.001) for index, row in forest_accs_and_sgms.iterrows()]
forest_sgms[3:7] = [np.round(forest_sgms[i]/100,6) for i in range(3,7)]
forest_sgms_df = pd.DataFrame({'Origin': accs_and_sgm.iloc[:,0], 'SGMs':forest_sgms})
plot_sgms(forest_sgms_df, forest_sgms_df['Origin'],'Forest SGMs')




