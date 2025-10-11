from typing import List, Union, Literal
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm



def bar_plot_time(tuples :List[tuple], title: str) -> None:
    # tuples: (Name of column, float)
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

    # bars = plt.bar(sorted_labels, sorted_values, colors)
    bars = plt.bar(sorted_labels, sorted_values, color=colors)#color=['blue', 'green', 'turquoise', 'red']
    #plt.title(title)
    plt.ylim(min(0, min(values) ) *1.1, max(values ) *1.1)  # Set y-axis limits for visibility
    plt.xticks(rotation=45, fontsize=6)
    # Create custom legend entries with value annotations
    legend_labels = [f"{label}: {value}" for label, value in zip(sorted_labels, sorted_values)]
    plt.legend(bars, legend_labels, title="Values")
    # Display the plot
    plt.show()
    plt.close()

def feature_histo(df : pd.DataFrame, columns : List[str], number_bins=10):
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
    else:
        for i, col in enumerate(columns):
            # Filter values in (0, 1)
            filtered_data = df[col][(df[col] > -100) & (df[col] < 100)]

            # Plot histogram with color distinction
            color = 'red' if 'Mixed' in col else ('magenta' if 'Int' in col else 'orange')
            axs[i].hist(filtered_data, bins=number_bins, color=color, alpha=1, label=f'Filtered ({len(filtered_data)} points)')
            axs[i].set_title(f'{col} (Values in (0, 1))')
            # Add legend to each subplot
            axs[i].legend()
    # Adjust layout
    plt.tight_layout()
    # Show the plots once all are created
    plt.show()
    # Close the plot to free up memory
    plt.close()