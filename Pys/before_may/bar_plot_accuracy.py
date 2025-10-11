import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the Excel file into a DataFrame
# df = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/Master_Excel/accuracy_per_intervall.xlsx')
# df = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/Master_Excel/ridge_accuracy_per_intervall.xlsx')
# df = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/accuracy_per_intervall_everything.csv')

def by_intervall(accuracy_df):
    # Define custom x-tick labels for each group of four columns
    xtick_labels = [
        "LinReg Mean byHand", "LinReg Mean Yeo-Johnson",
        "LinReg Median byHand", "LinReg Median Yeo-Johnson",
        "ForReg Mean byHand", "ForReg Mean Yeo-Johnson",
        "ForReg Median byHand", "ForReg Median Yeo-Johnson"
    ]

    # Iterate over each row for separate plots
    for idx, row in accuracy_df.iterrows():
        # Extract the columns to be plotted (1:17, assumed numerical data)
        data = row.iloc[1:].values
        titles = ['Total Acc', 'Mid Acc', 'Extreme Acc']

        # Create x positions for bars
        num_columns = len(data)
        group_size = 4
        groups = num_columns // group_size
        x_positions = []

        # Assign positions for the bars with spaces between groups
        for group in range(groups):
            x_positions.extend(range(group * (group_size + 1), group * (group_size + 1) + group_size))

        # Set bar colors based on the value
        colors = ['red' if val <= 50 else 'green' if val >= 80 else 'blue' for val in data]

        # Plot the bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(x_positions, data, color=colors)

        # Add labels and title
        #plt.title(titles[idx]+": All, Top 8 global, Top 8 Lin, Top 8 For")
        plt.xlabel("Columns")
        plt.ylabel("Values")
        plt.ylim(40, 100)
        # Add custom x-tick labels for each group of four
        tick_positions = [sum(range(group * (group_size + 1), group * (group_size + 1) + group_size)) / group_size for group
                          in range(groups)]
        plt.xticks(tick_positions, labels=xtick_labels, rotation=45, ha="right")
        # Show the plot
        plt.tight_layout()
        plt.show()

def compare_by_method(data : pd.DataFrame, title=False):
    """
        Creates a grouped bar plot with groups of three bars touching,
        and small gaps between groups.

        Args:
            data (pd.DataFrame): The DataFrame containing data (rows are grouped bars, columns are groups).
            group_labels (list of str): Labels for each group (columns in the DataFrame).
            row_labels (list of str): Labels for rows (used for legend).
            title (str): Title of the plot.
        """
    # Row labels for legend
    row_labels = data['Intervall']
    # Group labels for x-axis
    data = data.iloc[:,1:]
    group_labels = data.columns

    # Validate inputs
    num_groups = data.shape[1]  # Number of columns = number of groups
    num_rows = data.shape[0]  # Number of rows = bars per group
    assert len(group_labels) == num_groups, "Group labels must match the number of columns in data"
    assert len(row_labels) == num_rows, "Row labels must match the number of rows in data"

    # Define bar positions
    group_width = 0.8  # Total width of each group
    bar_width = group_width / num_rows  # Width of each individual bar
    group_positions = np.arange(num_groups) * (1 + group_width)  # Add space between groups

    # Flatten data for colors
    flat_data = data.values.flatten()
    colors = ['red' if val <= 60 else 'green' if val >= 80 else 'blue' for val in flat_data]

    # Plot each row of data
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, row_label in enumerate(row_labels):
        # Position bars for the row
        bar_positions = group_positions + (i - num_rows / 2) * bar_width + bar_width / 2
        # Generate the subset of colors for the current row
        row_colors = colors[i * data.shape[1]:(i + 1) * data.shape[1]]
        # Plot the row's bars
        ax.bar(bar_positions, data.iloc[i], width=bar_width, label=row_label, color=row_colors)

        # Level labels
        level1 = ["All", "Global8", "Lin8", "For8"] * (num_groups // 4)  # Repeat to match group count
        level2 = ["byHand", "Power"] * (num_groups // 8)  # Repeat for pairs of groups
        level3 = ["Mean", "Median"] * (num_groups // 16)  # Repeat for blocks of 4 groups

        # Calculate tick positions for levels
        level1_positions = group_positions  # Directly corresponds to group positions
        level2_positions = group_positions.reshape(-1, 4).mean(axis=1)  # Middle of each pair of groups
        level3_positions = group_positions.reshape(-1, 8).mean(axis=1)  # Middle of each block of 4 groups

        # Set x-ticks for each level
        ax.set_xticks(level1_positions, minor=False)
        ax.set_xticklabels(level1, minor=False, rotation=45)

        # Add level2 and level3 ticks as secondary and tertiary x-ticks
        for positions, labels, offset in zip(
                [level2_positions, level3_positions],
                [level2, level3],
                [0.2, 0.3],  # Adjust offset for clarity
        ):
            for pos, label in zip(positions, labels):
                ax.text(
                    pos,
                    -offset,
                    label,
                    ha="center",
                    va="top",
                    transform=ax.get_xaxis_transform(),
                    fontsize=10,
                    color="black",
                )
    # Add labels and formatting
    ax.set_ylabel("Values")
    ax.set_ylim(40, 100)
    # ax.set_xticks(group_positions)  # Set x-tick positions at group positions
    # ax.set_xticklabels(group_labels, rotation=45, ha="right")  # Set group labels as x-tick labels
    if title:
        #ax.set_title(title)
    ax.legend(title="Values by Group")
    #plt.xticks([])
    plt.tight_layout()
    plt.show()

# lin_cols = ['Intervall'] + df.filter(like="LinearRegression").columns.tolist()
# if len(lin_cols) > 1:
#     lin = df[lin_cols]
#     # compare_by_method(lin, 'LinAcc: All, Top 8 global, Top 8 Lin, Top 8 For')
#     # by_intervall(lin)
# for_cols = ['Intervall'] + df.filter(like="RandomForestRegressor").columns.tolist()
# if len(for_cols)>1:
#     forest = df[for_cols]
#     # compare_by_method(forest, 'ForestAcc: All, Top 8 global, Top 8 Lin, Top 8 For')
#     # by_intervall(forest)





"""1.regressor 2.imputation 3.scaler 4.feature_space"""