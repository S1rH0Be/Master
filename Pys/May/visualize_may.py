import pandas as pd
import matplotlib.pyplot as plt
from may_regression import shifted_geometric_mean


#TODO: Adjust Accuracy and run_time to new format

global_path = '/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasDos'

# SGM RUNTIME BLOCK
def sgm():
    def get_sgm_of_sgm(data_frame, shift):
        col_names = data_frame.columns.tolist()
        # Frage: SGM of relative SGMs oder von total SGMs?
        # Ich mach erstaml total sgms
        sgm_sgm_df = pd.DataFrame(columns=col_names, index=['Value'])
        for col in col_names:
            sgm_sgm_df.loc[:, col] = shifted_geometric_mean(data_frame[col], shift)
        return sgm_sgm_df

    def relative_to_mmixed(values):
        mixed = values[0]
        values = [value/mixed for value in values]
        return values

    def visualize_sgm(data_frame, title: str = 'SGMs'):
        linear_df = data_frame[data_frame['Model']=='LinearRegression'].iloc[:,3:]
        forest_series = data_frame[data_frame['Model']=='RandomForest'].iloc[:,5]
        linear_df.columns = ['SGM Mixed', 'SGM Int', 'SGM Linear', 'SGM VBS']
        linear_df.insert(loc=3, column='SGM Forest', value=forest_series.values)
        complete_df = linear_df

        complete_sgm_df = get_sgm_of_sgm(complete_df, 0)


        values = complete_sgm_df.iloc[0,:].tolist()
        values_relative = relative_to_mmixed(values)
        labels = ['Mixed', 'Int', 'Linear', 'Forest', 'VBS']
        # Determine bar colors based on conditions
        bar_colors = (['turquoise', 'magenta'])
                      # + ['green' if value >= 0.8 else 'red' if value <= 0.6 else 'blue' for value in values[3:7]])

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.bar(labels, values_relative, color=bar_colors)
        plt.title(title)
        plt.ylim(min(0.5, min(values_relative) * 0.9), max(values_relative) * 1.01)  # Set y-axis limits for visibility
        plt.xticks(rotation=45, fontsize=6)
        # Create custom legend entries with value annotations
        # legend_labels = [f"{label}: {value}" for label, value in zip(labels, values)]
        # plt.legend(bars, legend_labels, title="Values")
        # Display the plot
        plt.show()
        plt.close()

    scip_sgm_df = pd.read_csv(f'{global_path}/RunTime/scip_sgm_runtime.csv')
    scip_no_pseudos_df = pd.read_csv(f'{global_path}/RunTime/scip_no_pseudo_sgm_runtime.csv')
    fico_sgm_df = pd.read_csv(f'{global_path}/RunTime/fico_sgm_runtime.csv')
    visualize_sgm(scip_sgm_df, 'SCIP SGM')
    visualize_sgm(scip_no_pseudos_df, 'SCIP no Pseudo SGM')
    visualize_sgm(fico_sgm_df, 'FICO SGM')

# WHAT RULES DID THE MODELS CHOOSE
def shares():
    def get_share_mixed_and_int(data_frame):
        mixed = (data_frame > 0).sum().sum()
        pref_int = (data_frame < 0).sum().sum()
        return [mixed, pref_int]

    def histogram_shares(data_frame, title: str = 'Share of Mixed and Preferred Int'):
        values = get_share_mixed_and_int(data_frame)
        total_relevant_predictions = sum(values)
        values = [(value/total_relevant_predictions)*100 for value in values]
        bar_colors = (['turquoise', 'magenta'])

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.bar(['Mixed', 'Prefer Int'], values, color=bar_colors)
        plt.title(title)
        plt.ylim(30, 65)  # Set y-axis limits for visibility
        plt.xticks(rotation=45, fontsize=6)
        # Create custom legend entries with value annotations
        # legend_labels = [f"{label}: {value}" for label, value in zip(labels, values)]
        # plt.legend(bars, legend_labels, title="Values")
        # Display the plot
        plt.show()
        plt.close()

    scip_default_predictions = pd.read_csv(f'{global_path}/Prediction/scip_prediction_df.csv')
    fico_predictions = pd.read_csv(f'{global_path}/Prediction/fico_prediction_df.csv')
    scip_no_pseudo_predictions = pd.read_csv(f'{global_path}/Prediction/scip_no_pseudo_prediction_df.csv')
    histogram_shares(scip_default_predictions, title='SCIP Default: Share of Mixed and Preferred Int')
    histogram_shares(scip_no_pseudo_predictions, title='SCIP NO Pseudocosts: Share of Mixed and Preferred Int')
    histogram_shares(fico_predictions, title='FICO: Share of Mixed and Preferred Int')

# ACCURACY BLOCK
# TODO: Add parameter: Filter cols by string components, e.g Get all Linear columns with Median Imputation
def accuracy():
    def get_sgm_series(pandas_series, shift):
        return shifted_geometric_mean(pandas_series, shift)

    def get_sgm_acc(data_frame):
        sgm_accuracy = get_sgm_series(data_frame['Accuracy'], data_frame['Accuracy'].mean())
        sgm_extreme_accuracy = get_sgm_series(data_frame['Extreme Accuracy'].dropna(), data_frame['Extreme Accuracy'].dropna().mean())
        return sgm_accuracy, sgm_extreme_accuracy


    def visualize_acc(data_frame, filter_by:[int], title: str = 'Accuracy'):
        linear_df = data_frame[data_frame['Model']=='LinearRegression'].loc[:,['Accuracy', 'Extreme Accuracy',
                                                                               'Number of extreme instances']]
        forest_df = data_frame[data_frame['Model']=='RandomForest'].loc[:,['Accuracy', 'Extreme Accuracy',
                                                                               'Number of extreme instances']]
        lin_acc, lin_ex_acc = get_sgm_acc(linear_df)
        for_acc, for_ex_acc = get_sgm_acc(forest_df)

        values = [lin_acc, lin_ex_acc, for_acc, for_ex_acc]

        # Create the plot
        bar_colors = (['turquoise', 'turquoise', 'magenta', 'magenta'])
        plt.figure(figsize=(8, 5))
        plt.bar(['LinAcc', 'LinExAcc', 'ForAcc', 'ForExAcc', ], values, color=bar_colors)
        plt.title(title)
        # plt.ylim(min(0.5, min(values) * 0.9), max(values) * 1.1)  # Set y-axis limits for visibility
        plt.ylim(0,100)
        plt.xticks(rotation=45, fontsize=6)
        # Create custom legend entries with value annotations
        # legend_labels = [f"{label}: {value}" for label, value in zip(labels, values)]
        # plt.legend(bars, legend_labels, title="Values")
        # Display the plot
        plt.show()
        plt.close()

    scip_default_accuracy = pd.read_csv(f'{global_path}/Accuracy/scip_acc_df.csv')
    fico_accuracy = pd.read_csv(f'{global_path}/Accuracy/fico_acc_df.csv')
    scip_no_pseudo_accuracy = pd.read_csv(f'{global_path}/Accuracy/scip_no_pseudo_acc_df.csv')
    visualize_acc(scip_default_accuracy, 'SCIP Default Accuracy')
    visualize_acc(scip_no_pseudo_accuracy, 'SCIP NO Pseudocosts Accuracy')
    visualize_acc(fico_accuracy, 'FICO Accuracy')

# Feature Importances, as sgm
def importance():
    def feature_importance(data_frame, title: str = 'Feature Importance'):
        feature_names = data_frame.index.tolist()
        importance_dict = {}
        linear_columns = [lin_col for lin_col in data_frame.columns if 'LinearRegression' in lin_col]
        forest_columns = [for_col for for_col in data_frame.columns if 'RandomForest' in for_col]

        for feature in feature_names:
            minimum = min(data_frame.loc[feature,:])
            mean = data_frame.loc[feature,:].mean()
            importance_dict[feature] = shifted_geometric_mean(data_frame.loc[feature,:], abs(minimum)+abs(mean))

        sgm_importance_df = pd.DataFrame.from_dict(importance_dict, orient='index' )

        return sgm_importance_df

    def importance_bar_plot(data_frame, title):
        importance_df = feature_importance(data_frame, title)
        values = importance_df.iloc[:,0].tolist()
        # Create the plot
        bar_colors = (['turquoise', 'magenta'])
        plt.figure(figsize=(8, 5))
        plt.bar([i for i in range(len(values))], values, color=bar_colors)
        plt.title(title)
        # plt.ylim(min(0.5, min(values) * 0.9), max(values) * 1.1)  # Set y-axis limits for visibility
        # plt.ylim(0, 100)
        plt.xticks(rotation=45, fontsize=6)
        # Create custom legend entries with value annotations
        # legend_labels = [f"{label}: {value}" for label, value in zip(labels, values)]
        # plt.legend(bars, legend_labels, title="Values")
        # Display the plot
        plt.show()
        plt.close()

    def plot_importances_by_regressor(data_frame, title):
        linear_cols = [col_name for col_name in data_frame.columns if 'LinearRegression' in col_name]
        forest_cols = [col_name for col_name in data_frame.columns if 'RandomForest' in col_name]
        linear_importance_df = data_frame[linear_cols]
        forest_importance_df = data_frame[forest_cols]
        importance_bar_plot(linear_importance_df, f'{title} LinearRegression')
        importance_bar_plot(forest_importance_df, f'{title} RandomForest')

    def get_score(data_series):
        sorted = data_series.abs().sort_values(ascending=False)
        score_dict = {feature:0 for feature in sorted.index.tolist()}
        score = 0
        for feature in sorted.index.tolist():
            score_dict[feature] = score
            score += 1

        return score_dict

    def get_importance_score_df(data_frame):
        linear_cols = [col_name for col_name in data_frame.columns if 'LinearRegression' in col_name]
        forest_cols = [col_name for col_name in data_frame.columns if 'RandomForest' in col_name]
        linear_importance_df = data_frame[linear_cols]
        forest_importance_df = data_frame[forest_cols]

        linear_scores_dict = {feature_name:0 for feature_name in linear_importance_df.index.tolist()}
        forest_scores_dict = {feature_name:0 for feature_name in forest_importance_df.index.tolist()}

        # linear scores
        for run in linear_importance_df.columns:
            run_scores = get_score(linear_importance_df[run])
            for feature in linear_scores_dict.keys():
                linear_scores_dict[feature] += run_scores[feature]

        # forest scores
        for run in forest_importance_df.columns:
            run_scores = get_score(forest_importance_df[run])
            for feature in forest_scores_dict.keys():
                forest_scores_dict[feature] += run_scores[feature]

        return pd.DataFrame({'Linear': linear_scores_dict, 'Forest': forest_scores_dict})

    def plot_importance(data_frame, title):
        importance_df = get_importance_score_df(data_frame)
        importance_df.plot(kind='bar', figsize=(10, 6), color=['turquoise', 'magenta'])
        plt.title(title)
        plt.ylabel('Importance Score')
        plt.xlabel('Feature')
        plt.xticks(rotation=270, fontsize=6)
        plt.show()
        plt.close()

    scip_default_importance = pd.read_csv(f'{global_path}/Importance/scip_importance_df.csv', index_col=0)
    scip_no_pseudo_importance = pd.read_csv(f'{global_path}/Importance/scip_no_pseudo_importance_df.csv', index_col=0)
    fico_importance = pd.read_csv(f'{global_path}/Importance/fico_importance_df.csv', index_col=0)
    plot_importances_by_regressor(scip_default_importance, 'SCIP Default Feature Importance')
    plot_importances_by_regressor(scip_no_pseudo_importance, 'SCIP NO Pseudocosts Feature Importance')
    plot_importances_by_regressor(fico_importance, 'FICO Feature Importance')
    plot_importance(scip_default_importance, 'SCIP Default Feature Importance Score')
    plot_importance(scip_no_pseudo_importance, 'SCIP NO Pseudocosts Feature Importance Score')
    plot_importance(fico_importance, 'FICO Feature Importance Score')

# abs time save
def time_save():
    def create_time_save_df(data_frame):
        time_save_df = data_frame[['Final solution time (cumulative) Mixed',
                                    'Final solution time (cumulative) Int',
                                    'Cmp Final solution time (cumulative)']].copy()

        # Copy relevant columns for clarity
        mixed_col = time_save_df['Final solution time (cumulative) Mixed']
        int_col = time_save_df['Final solution time (cumulative) Int']
        label_col = time_save_df['Cmp Final solution time (cumulative)']

        # Condition: mixed > int and label ≠ 0
        condition = (mixed_col > int_col) & (label_col != 0.0)

        # Compute absolute time save where condition is True, else 0
        time_save_df['Possible Time Save'] = 0.0
        time_save_df.loc[condition, 'Possible Time Save'] = (mixed_col - int_col).abs()

        return time_save_df

    def visualize_time_save(data_frame):
        tdf = create_time_save_df(data_frame)
        sgm_mixed = shifted_geometric_mean(tdf['Final solution time (cumulative) Mixed'],
                                           tdf['Final solution time (cumulative) Mixed'].mean())
        sgm_int = shifted_geometric_mean(tdf['Final solution time (cumulative) Int'],
                                         tdf['Final solution time (cumulative) Mixed'].mean())
        sgm_time_save = shifted_geometric_mean(tdf['Possible Time Save'], tdf['Possible Time Save'].mean())

        print(sgm_mixed, sgm_int, sgm_time_save)
        print(tdf['Final solution time (cumulative) Mixed'].sum(), tdf['Final solution time (cumulative) Int'].sum(), tdf['Possible Time Save'].sum())

    scip_default_data = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_default_clean_data.csv')
    scip_no_pseudocosts_data = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_no_pseudocosts_clean_data.csv')
    fico_data = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/918/clean_data_final_06_03.xlsx')

    visualize_time_save(scip_default_data)
    visualize_time_save(scip_no_pseudocosts_data)
    visualize_time_save(fico_data)

# label analysis
def label():
    scip_default_label = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_default_clean_data.csv')
    scip_no_pseudocosts_label = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_no_pseudocosts_clean_data.csv')
    fico_label = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/918/clean_data_final_06_03.xlsx')

    labels = [(scip_default_label['Cmp Final solution time (cumulative)'], 'SCIP Default'),
              (scip_no_pseudocosts_label['Cmp Final solution time (cumulative)'], 'SCIP No Pseudocosts'),
              (fico_label['Cmp Final solution time (cumulative)'], 'FICO Xpress')]

    for label in labels:
        plt.figure(figsize=(8, 6))
        plt.scatter([0] * len(label[0]), label[0], color='white', edgecolor='k', alpha=1)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)  # Reference line at y=0
        plt.title(f"Label {label[1]}")
        plt.xlabel("Pred values")
        plt.ylabel("Actual Values")
        plt.show()


# sgm()
# shares()
# accuracy()
# importance()
# time_save()
# label()



