import pandas as pd
import matplotlib.pyplot as plt
from may_regression import shifted_geometric_mean
import numpy as np
import sys
import os
'''
@TODO rewrite it again.........
So das ich nur den ordner inputten muss und es dann von da alles alleine findet aka 
Ich inputte TreffenMasVeinte, dann sucht es sich alles von dort aus wie accuracy, und und und
1. Add functions to find best imputer/scaling kombi
'''

global_path = '/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasVeinte/SoloQuantilePreScaled'

def get_files(directory_path:str, index_col=False):
    """
    Reads all CSV and Excel files in the given directory.

    Parameters:
        directory_path (str): Path to the directory.

    Returns:
        dict: A dictionary with filenames as keys and DataFrames as values.
    """
    dataframes = {}
    for filename in os.listdir(directory_path):
        file_key = filename.replace('.csv', '').replace('_', ' ').replace('df', '').title()
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            try:
                if filename.lower().endswith('.csv'):
                    if index_col:
                        df = pd.read_csv(file_path, index_col=0)
                    else:
                        df = pd.read_csv(file_path)
                    dataframes[file_key] = df
                elif filename.lower().endswith(('.xls', '.xlsx', '.xlsm')):
                    if index_col:
                        df = pd.read_excel(file_path, index_col=0)
                    else:
                        df = pd.read_excel(file_path)
                    dataframes[file_key] = df
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return dataframes

# SGM RUNTIME BLOCK
def sgm(scaled_label=True):

    def get_sgm_of_sgm(data_frame, shift):
        col_names = data_frame.columns.tolist()
        # Frage: SGM of relative SGMs oder von total SGMs?
        # Ich mach erstmal total sgms
        sgm_sgm_df = pd.DataFrame(columns=col_names, index=['Value'])
        for col in col_names:
            sgm_sgm_df.loc[:, col] = shifted_geometric_mean(data_frame[col], shift)
        return sgm_sgm_df

    def relative_to_default(value_dict:dict, dataset:str):
        if dataset == 'fico':
            default_rule = 'Mixed'
        elif dataset == 'scip':
            default_rule = 'Int'
        else:
            print(f'{dataset} is not a valid dataset')
            sys.exit(1)

        default = value_dict[default_rule]
        values = [value_dict[rule]/default for rule in value_dict.keys()]
        return values

    def get_values_for_plot(dataframe:pd.DataFrame, data_set):
        mixed = dataframe['Mixed']
        pref_int = dataframe['Int']
        prediction = dataframe['Predicted']
        vbs = dataframe['VBS']

        means = [mixed.quantile(0.05).min(), pref_int.quantile(0.05).min(), prediction.quantile(0.05).min(),
                 vbs.quantile(0.05).min()]

        mean_mean = np.min(means)
        mixed_values = shifted_geometric_mean(mixed, mean_mean)
        pref_int_values = shifted_geometric_mean(pref_int, mean_mean)
        predicted_values = shifted_geometric_mean(prediction, mean_mean)
        vbs_values = shifted_geometric_mean(vbs, mean_mean)
        value_dictionary = {'Int': pref_int_values, 'Mixed': mixed_values, 'Predicted': predicted_values,
                            'VBS': vbs_values}
        values_relative = relative_to_default(value_dictionary, data_set)

        return values_relative

    def sgm_plot(dataframe, title:str, data_set:str):

        values = get_values_for_plot(dataframe, data_set)
        labels = ['PrefInt', 'Mixed', 'Predicted', 'VBS']

        bar_colors = ['turquoise', 'magenta']

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.bar(labels, values, color=bar_colors)
        plt.title(title)
        if data_set == 'fico':
            plt.ylim(0.5, 1.35)  # Set y-axis limits for visibility
        else:
            plt.ylim(0.8, 1.06)  # Set y-axis limits for visibility
        plt.xticks(rotation=45, fontsize=6)
        # Display the plot
        plt.show()
        plt.close()

    def call_sgm_visualization(scaledlabel, title=False):
        if scaledlabel:
            dataframes = get_files(global_path+'/ScaledLabel/SGM/', index_col=True)
        else:
            dataframes = get_files(global_path + '/UnscaledLabel/SGM/', index_col=True)

        for dataframe in dataframes.keys():
            df = dataframes[dataframe]
            sgm_plot(df, dataframe+' combined', dataframe[:4])
            linear_runs = [index for index in df.index if 'LinearRegression' in index]
            forest_runs = [index for index in df.index if 'RandomForest' in index]
            linear_df = df.loc[linear_runs, :]
            forest_df = df.loc[forest_runs, :]
            sgm_plot(linear_df, dataframe+' LinearRgeression', dataframe[:4])
            sgm_plot(forest_df, dataframe+' RandomForestRegression', dataframe[:4])

    call_sgm_visualization(scaledlabel=scaled_label)

# WHAT RULES DID THE MODELS CHOOSE
def shares(scip_default_original_data, fico_original_data, scaledlabel=True, complete_data=False):
    def get_share_mixed_and_int(data_frame):
        mixed = (data_frame > 0).sum().sum()
        pref_int = (data_frame < 0).sum().sum()
        return [mixed, pref_int]

    def histogram_shares(data_frame, title_add_on, origis=False):
        if origis:
            values = get_share_mixed_and_int(data_frame['Cmp Final solution time (cumulative)'])
        else:
            values = get_share_mixed_and_int(data_frame)

        total_relevant_predictions = sum(values)
        values = [(value/total_relevant_predictions)*100 for value in values]
        bar_colors = (['turquoise', 'magenta'])
        title = f'Share of Mixed and Preferred Int {title_add_on}'
        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.bar(['Mixed', 'Prefer Int'], values, color=bar_colors)
        plt.title(title)
        plt.ylim(0, 105)  # Set y-axis limits for visibility
        plt.xticks(rotation=45, fontsize=6)
        # Create custom legend entries with value annotations
        # legend_labels = [f"{label}: {value}" for label, value in zip(labels, values)]
        # plt.legend(bars, legend_labels, title="Values")
        # Display the plot
        plt.show()
        plt.close()

    if scaledlabel:
        dataframes = get_files(global_path+'/ScaledLabel/Prediction/', index_col=True)
    else:
        dataframes = get_files(global_path+'/UnscaledLabel/Prediction/', index_col=True)

    for dataframe in dataframes.keys():
        histogram_shares(dataframes[dataframe], dataframe)
    # TODO: Keep in mind when changing base files
    if complete_data:
        histogram_shares(fico_original_data, 'FICO Complete Set', origis=True)
        histogram_shares(scip_default_original_data, 'SCIP Complete Set', origis=True)

# ACCURACY BLOCK
def accuracy(scaled_label=True):
    def get_sgm_series(pandas_series, shift):
        return shifted_geometric_mean(pandas_series, shift)

    def get_sgm_acc(data_frame):
        data_frame['Accuracy'] = pd.to_numeric(data_frame['Accuracy'], errors='coerce')
        data_frame['Extreme Accuracy'] = pd.to_numeric(data_frame['Extreme Accuracy'], errors='coerce')

        sgm_accuracy = get_sgm_series(data_frame['Accuracy'], data_frame['Accuracy'].mean()+0.1)
        sgm_extreme_accuracy = get_sgm_series(data_frame['Extreme Accuracy'].dropna(), data_frame['Extreme Accuracy'].dropna().mean()+0.1)
        return sgm_accuracy, sgm_extreme_accuracy


    def visualize_acc(data_frame, filter_by:str, title: str = 'Accuracy'):
        acc_df = data_frame.copy()
        # if no corresponding acc is found just plot it as 0
        lin_acc, lin_ex_acc, for_acc, for_ex_acc = 0, 0, 0, 0

        if filter_by != '':
            wanted_runs = [run for run in data_frame.columns if filter_by in run]
            acc_df = acc_df.loc[:, wanted_runs]

        linear_rows = [lin_rows for lin_rows in acc_df.index if 'LinearRegression' in lin_rows]
        linear_df = acc_df.loc[linear_rows, :]

        forest_rows = [for_row for for_row in acc_df.index if 'RandomForest' in for_row]
        forest_df = acc_df.loc[forest_rows,:]

        if len(linear_df) == len(forest_df) == 0:
            print(f'{title}: No data found')
            return None
        else:
            if len(linear_df)>0:
                lin_acc, lin_ex_acc = get_sgm_acc(linear_df)
            if len(forest_df)>0:
                for_acc, for_ex_acc = get_sgm_acc(forest_df)

        values = [lin_acc, lin_ex_acc, for_acc, for_ex_acc]

        # Create the plot
        bar_colors = (['turquoise', 'turquoise', 'magenta', 'magenta'])
        plt.figure(figsize=(8, 5))
        plt.bar(['LinAcc', 'LinExAcc', 'ForAcc', 'ForExAcc'], values, color=bar_colors)
        plt.title(title)
        plt.ylim(0, 105)  # Set y-axis limits for visibility
        plt.xticks(rotation=45, fontsize=6)
        # Create custom legend entries with value annotations
        # Display the plot
        plt.show()
        plt.close()

    def call_acc_visualization(scaledlabel, title_add_on=''):
        if scaledlabel:
            dataframes = get_files(global_path+'/ScaledLabel/Accuracy/', index_col=True)
        else:
            dataframes = get_files(global_path + '/UnscaledLabel/Accuracy/', index_col=True)

        for dataframe in dataframes.keys():
            df = dataframes[dataframe]
            visualize_acc(df, filter_by='', title=dataframe+title_add_on)

    call_acc_visualization(scaledlabel=scaled_label)

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
            importance_dict[feature] = shifted_geometric_mean(data_frame.loc[feature,:], abs(minimum)+0.1)

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

    def get_importance_score_df(data_frame, title):
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

        df = pd.DataFrame({
            'Feature': list(linear_scores_dict.keys()),
            'Linear': list(linear_scores_dict.values()),
            'Forest': list(forest_scores_dict.values())
        })
        # Add the 'Combined' column
        df['Combined'] = df['Linear'] + df['Forest']

        # Sort by 'Combined' in ascending order (smallest on top)
        df = df.sort_values(by='Combined', ascending=True)
        df.to_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasQuince/FICOonSCIP/SoloQuantile/ScaledLabel/Importance/{title}.csv', index=False)

        return pd.DataFrame({'Linear': linear_scores_dict, 'Forest': forest_scores_dict})

    def plot_importance(data_frame, title):
        importance_df = get_importance_score_df(data_frame, title)
        importance_df.plot(kind='bar', figsize=(10, 6), color=['turquoise', 'magenta'])
        plt.title(title)
        plt.ylabel('Importance Score')
        plt.xlabel('Feature')
        plt.xticks(rotation=270, fontsize=6)
        plt.show()
        plt.close()

# abs time save
def time_save(scip_default_original_data, fico_original_data):
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

    def visualize_time_save(default, fico, title:str):
        tdf_default = create_time_save_df(default)
        tdf_fico = create_time_save_df(fico)

        sgm_time_save_default = shifted_geometric_mean(tdf_default['Possible Time Save'],
                                                       tdf_default['Possible Time Save'].mean())
        sgm_time_save_fico = shifted_geometric_mean(tdf_fico['Possible Time Save'],
                                                         tdf_fico['Possible Time Save'].mean())

        values = [sgm_time_save_default, sgm_time_save_fico]
        bar_colors = (['turquoise', 'magenta'])

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.bar(['SCIP Default', 'SCIP No Pseudocosts', 'FICO Xpress'], values, color=bar_colors)
        plt.title(title)
        plt.ylim(0, max(values)*1.1)  # Set y-axis limits for visibility
        plt.xticks(rotation=45, fontsize=6)
        # Create custom legend entries with value annotations
        # legend_labels = [f"{label}: {value}" for label, value in zip(labels, values)]
        # plt.legend(bars, legend_labels, title="Values")
        # Display the plot
        plt.show()
        plt.close()

# label analysis
def label(scip_default_original_data, fico_original_data, scaled=False):

    labels = [(scip_default_original_data['Cmp Final solution time (cumulative)'], 'SCIP Default'),
              (fico_original_data['Cmp Final solution time (cumulative)'], 'FICO Xpress')]

    for label in labels:
        plt.figure(figsize=(8, 6))
        plt.scatter([0] * len(label[0]), label[0], color='white', edgecolor='k', alpha=1)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)  # Reference line at y=0
        if scaled:
            plt.title(f"Scaled Label {label[1]}")
        else:
            plt.title(f"Unscaled Label {label[1]}")
        plt.xlabel("Pred values")
        plt.ylabel("Actual Values")
        plt.show()

def label_scaling(label):
    y_pos = label[label >= 0]
    y_neg = label[label < 0]
    y_pos_log = np.log(y_pos + 1)
    y_neg_log = np.log(abs(y_neg) + 1) * -1
    y_log = pd.concat([y_pos_log, y_neg_log]).sort_index()
    return y_log

def comp_fico_scip(scip_default_base, fico_base):
    scip_cols = scip_default_base.columns.tolist()
    fico_cols = fico_base.columns.tolist()
    intersec = list(set(scip_cols).intersection(set(fico_cols)))

    for col in intersec:
        if scip_default_base[col].dtype != 'object':
            scip_mean = scip_default_base[col].mean()
            fico_mean = fico_base[col].mean()
            print(f'{col}:\n SCIP: {scip_mean}, FICO: {fico_mean}')

def create_scip_feature_name_df():
    scip_feats = pd.read_csv(
        '/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/cleaned_scip/scip_default_clean_feats.csv')
    scip_feat_df = pd.DataFrame(columns=['Feature Name'], data=scip_feats.columns.tolist())
    scip_feat_df = scip_feat_df.replace({'#': r'\#', '%': r'\%'}, regex=True)

    scip_feat_df.to_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/cleaned_scip/scip_feature_names.csv',
                        index=False)
    return scip_feat_df

def main(treffmas, scale_label=True, visualize_sgm=False, visualize_shares=False, visualize_accuracy=False,
         visualize_importance=False, visualize_time_save=False, visualize_label=False, comp_ficip=False, title_add_on='Wurm',
         label_scaled=True):

    scip_default_base = pd.read_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/cleaned_scip/scip_default_clean_data.csv')
    fico_base = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/918/clean_data_final_06_03.xlsx')
    fico_feats = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/918/base_feats_no_cmp_918_24_01.xlsx')
    scip_feature_names = create_scip_feature_name_df()


    if scale_label:
        scip_default_base.loc[:, 'Cmp Final solution time (cumulative)'] = label_scaling(scip_default_base.loc[:, 'Cmp Final solution time (cumulative)'])
        fico_base.loc[:, 'Cmp Final solution time (cumulative)'] = label_scaling(fico_base.loc[:, 'Cmp Final solution time (cumulative)'])

    global global_path
    global_path = f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffmas}'
    if visualize_sgm:
        sgm(scaled_label=label_scaled)
    if visualize_shares:
        shares(scip_default_base, fico_base, scaledlabel=label_scaled, complete_data=True)
    if visualize_accuracy:
        accuracy(label_scaled)
    if visualize_importance:
        importance()
    if visualize_time_save:
        time_save(scip_default_base, fico_base)
    if visualize_label:
        label(scip_default_base, fico_base, scale_label)
    if comp_ficip:
        scip_fic = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/scip_default_schnitt.csv')
        fic_scip = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/fico_schnitt.csv')
        comp_fico_scip(scip_fic, fic_scip)

main('TreffenMasVeinte/SoloQuantilePreScaled', scale_label=True, visualize_sgm=False, visualize_shares=False,
     visualize_accuracy=True, visualize_importance=False, visualize_time_save=False, visualize_label=False,
     comp_ficip=False, title_add_on='Only Quantile')









