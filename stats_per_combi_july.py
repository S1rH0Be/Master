import pandas as pd
import os


from visualize_july import shifted_geometric_mean
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal


def setup_directory(new_directory):
    # Setup directory
    os.makedirs(os.path.join(f'{new_directory}'), exist_ok=True)

def get_sgm_of_cols(df, shift):
    col_names = df.columns.tolist()
    sgm_sgm_df = pd.DataFrame(columns=col_names, index=['Value'])
    for col in col_names:
        sgm_sgm_df.loc[:, col] = shifted_geometric_mean(df[col], shift)
    return sgm_sgm_df

def sgm(data_frame, title:str):

    def get_sgm_of_sgm(df, shift):
        col_names = df.columns.tolist()
        sgm_sgm_df = pd.DataFrame(columns=col_names, index=['Value'])
        for col in col_names:
            sgm_sgm_df.loc[:, col] = shifted_geometric_mean(df[col], shift)
        return sgm_sgm_df

    def relative_to_mixed(value_df):
        values = value_df.iloc[0, :].tolist()
        mixed = value_df['Mixed'].iloc[0]
        values = [value/mixed for value in values]
        return values

    def visualize_sgm(df, plot_title: str = 'SGMs'):
        pred_df = df.copy()
        forest_index = [ind for ind in df.index if "RandomForest" in ind]
        linear_index = [ind for ind in df.index if "Linear" in ind]

        forest_sgm_df = get_sgm_of_sgm(pred_df.loc[forest_index, :], pred_df.loc[forest_index, :].mean().mean())
        linear_sgm_df = get_sgm_of_sgm(pred_df.loc[linear_index, :], pred_df.loc[linear_index, :].mean().mean())

        new_order = ['Int', 'Mixed', 'Predicted', 'VBS']
        forest_sgm_df = forest_sgm_df.reindex(columns=new_order)
        linear_sgm_df = linear_sgm_df.reindex(columns=new_order)

        values_relative_forest = relative_to_mixed(forest_sgm_df)
        values_relative_linear = relative_to_mixed(linear_sgm_df)
        values_relative = values_relative_linear[:3]+values_relative_forest[2:]

        labels = ['Int', 'Mixed', 'Linear', 'Forest', 'VBS']
        # Determine bar colors based on conditions
        bar_colors = (['turquoise', 'magenta'])

        # Create the plot
        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, values_relative, color=bar_colors)
        plt.title(plot_title)
        plt.ylim(0.5, max(values_relative)*1.05)  # Set y-axis limits for visibility
        plt.xticks(rotation=45, fontsize=6)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom')
        # Display the plot
        plt.show()
        plt.close()
        return values_relative
    sgms = visualize_sgm(data_frame, title)

    return sgms

def train_vs_test_sgm(training_frame_path:str, test_frame_path:str, version: str, fico_or_scip:Literal['scip', 'fico'],
                      save_to:str, only_linear=False, only_forest=False, index_column=None, sgm_shift=10,
                      save_plot=False):
    setup_directory(save_to)
    def get_sgm_of_sgm(data_frame, shift):
        col_names = data_frame.columns.tolist()
        sgm_sgm_df = pd.DataFrame(columns=col_names, index=['Value'])
        for col in col_names:
            sgm_sgm_df.loc[:, col] = shifted_geometric_mean(data_frame[col], shift)
        return sgm_sgm_df

    def relative_to_mixed(value_df):
        values = value_df.iloc[0, :].tolist()
        mixed = value_df['Mixed'].iloc[0]
        return [value / mixed for value in values]

    def relative_to_int(value_df):
        values = value_df.iloc[0, :].tolist()
        int = value_df['Int'].iloc[0]
        return [value / int for value in values]


    def visualize_sgm(dfs, fico_or_scip, version="", plot_title: str = 'SGMs', saving_directory="",
                      only_lin=True, only_for=False, shift_by=10, save=False):
        if fico_or_scip == 'scip':
            labels = ['Int', 'Linear', 'Forest', 'VBS']
        else:
            labels = ['Mixed', 'Linear', 'Forest', 'VBS']
        x = np.arange(len(labels))
        bar_width = 0.25
        n_dfs = len(dfs)
        plt.figure(figsize=(10, 6))
        colors = ['purple', 'pink']
        for i, df in enumerate(dfs):
            pred_df = df.copy()
            lin_runs = [index for index in pred_df.index if 'LinearRegression' in index]
            for_runs = [index for index in pred_df.index if 'RandomForest' in index]
            lin_pred_df = pred_df.loc[lin_runs, :]
            for_pred_df = pred_df.loc[for_runs, :]
            sgm_lin_df = get_sgm_of_sgm(lin_pred_df, shift_by)
            sgm_for_df = get_sgm_of_sgm(for_pred_df, shift_by)
            if fico_or_scip=='fico':
                values_relative_lin = relative_to_mixed(sgm_lin_df)
                values_relative_for = relative_to_mixed(sgm_for_df)
                values_relative = [values_relative_lin[1], values_relative_lin[0], values_relative_for[0],
                                   values_relative_for[3]]
            else:
                values_relative_lin = relative_to_int(sgm_lin_df)
                values_relative_for = relative_to_int(sgm_for_df)
                values_relative = [values_relative_lin[2], values_relative_lin[0], values_relative_for[0],
                                   values_relative_for[3]]

            if only_lin:
                labels = ['Int', 'Linear', 'VBS']
                x=np.arange(3)
                values_relative_lin = relative_to_mixed(sgm_lin_df)
                values_relative_for = relative_to_mixed(sgm_for_df)
                values_relative = [values_relative_lin[1], values_relative_lin[0],
                                   values_relative_for[3]]
            if only_for:
                labels = ['Int', 'Forest', 'VBS']
                values_relative_lin = relative_to_mixed(sgm_lin_df)
                values_relative_for = relative_to_mixed(sgm_for_df)
                values_relative = [values_relative_lin[1], values_relative_for[0],
                                   values_relative_for[3]]
                x = np.arange(3)
            bars = plt.bar(x + i * bar_width, values_relative, width=bar_width, label=f'{dfs[i].name}', color=colors[i%2])
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.2f}',
                         ha='center', va='bottom')

        plt.xticks(x + bar_width * (n_dfs - 1) / 2, labels)
        plt.title(f"{plot_title} {version}")
        plt.ylim(0, 1.2)
        plt.ylabel('Relative SGM (to Mixed)')

        if 'SCIP' in plot_title:
            plt.legend(frameon=True, facecolor='white', framealpha=1.0)
        else:
            plt.legend()
        plt.tight_layout()
        if save:
            filename = f"{saving_directory}/{fico_or_scip}_{version}_train_vs_test_sgm.png"
            plt.savefig(filename)
        plt.show()
        plt.close()

    if fico_or_scip == "scip":
        scip_testset = pd.read_csv(training_frame_path, index_col=index_column)
        scip_testset.name = f'Testset'
        scip_trainset = pd.read_csv(test_frame_path, index_col=index_column)
        scip_trainset.name = f'Trainingset'
        visualize_sgm([scip_trainset, scip_testset], fico_or_scip=fico_or_scip, plot_title=f'SCIP',
                      saving_directory=save_to, version=version, only_lin=only_linear, only_for=only_forest,
                      shift_by=sgm_shift, save=save_plot)

    if fico_or_scip == "fico":
        fico_trainset = pd.read_csv(training_frame_path, index_col=index_column)
        fico_trainset.name = f'Trainingset'

        fico_testset = pd.read_csv(test_frame_path, index_col=index_column)
        fico_testset.name = f'Testset'
        visualize_sgm([fico_trainset, fico_testset], fico_or_scip=fico_or_scip, plot_title="FICO Xpress",
                      saving_directory=save_to, version=version, only_lin=only_linear, only_for=only_forest,
                      shift_by=sgm_shift, save=save_plot)

def train_vs_test_accuracy(training_frame_path:str, test_frame_path:str, version: str, fico_or_scip:Literal['scip', 'fico'],
                          save_to:str, only_linear=False, only_forest=False, index_column=None, sgm_shift=10,
                          save_plot=False):
    setup_directory(save_to)
    def get_sgm_series(pandas_series, shift):
        if len(pandas_series) == 0:
            return 0
        return shifted_geometric_mean(pandas_series, shift)

    def get_sgm_acc(data_frame, fico_or_scip, shift_values_by):

        data_frame['Accuracy'] = pd.to_numeric(data_frame['Accuracy'], errors='coerce')
        data_frame['Mid Accuracy'] = pd.to_numeric(data_frame['Mid Accuracy'], errors='coerce')
        if fico_or_scip == 'fico':
            data_frame['Extreme Accuracy'] = pd.to_numeric(data_frame['Extreme Accuracy'], errors='coerce')

        sgm_accuracy = get_sgm_series(data_frame['Accuracy'], shift=shift_values_by)
        sgm_mid_accuracy = get_sgm_series(data_frame['Mid Accuracy'], shift=shift_values_by)
        return_list = [sgm_accuracy, sgm_mid_accuracy]
        if fico_or_scip == 'fico':
            if len(data_frame['Extreme Accuracy'])==0:
                return_list.append(0)
            else:
                sgm_extreme_accuracy = get_sgm_series(data_frame['Extreme Accuracy'].dropna(), shift=shift_values_by)
                return_list.append(sgm_extreme_accuracy)

        return return_list

    def visualize_acc(dfs, fico_or_scip, plot_title:str, saving_directory:str, version:str,
                      only_lin=False, only_for=False, save=False, shift_by=10):

        if fico_or_scip == 'fico':
            if only_lin:
                categories = ['Linear', 'Linear MidLabel', 'Linear LargeLabel']
            elif only_for:
                categories = ['Forest', 'Forest MidLabel', 'Forest LargeLabel']
            else:
                categories = ['Linear', 'Linear MidLabel', 'Linear LargeLabel', 'Forest', 'Forest MidLabel',
                              'Forest LargeLabel']
        else:
            if only_lin:
                categories = ['Linear', 'Linear MidLabel']
            elif only_for:
                categories = ['Forest', 'Forest MidLabel']
            else:
                categories = ['Linear', 'Linear MidLabel', 'Forest', 'Forest MidLabel']

        n_dfs = len(dfs)
        x = np.arange(len(categories))  # label locations
        bar_width = 0.25

        plt.figure(figsize=(10, 6))

        for i, df in enumerate(dfs):
            acc_df = df.copy()
            print(df.name)
            linear_rows = [row for row in acc_df.index if 'LinearRegression' in row]
            forest_rows = [row for row in acc_df.index if 'RandomForest' in row]

            linear_df = acc_df.loc[linear_rows]
            forest_df = acc_df.loc[forest_rows]

            lin_acc, lin_mid_acc, lin_ex_acc, for_acc, for_mid_acc, for_ex_acc = 0, 0, 0, 0, 0, 0

            if len(linear_df) > 0:
                lin_acc = get_sgm_acc(linear_df, fico_or_scip, shift_values_by=shift_by)
            if len(forest_df) > 0:
                for_acc = get_sgm_acc(forest_df, fico_or_scip, shift_values_by=shift_by)
            if fico_or_scip == 'scip':
                if only_lin:
                    values = [lin_acc[0], lin_acc[1]]
                    x=np.arange(2)
                elif only_for:
                    values= [for_acc[0], for_acc[1]]
                    x=np.arange(2)
                else:
                    values = [lin_acc[0], lin_acc[1], for_acc[0], for_acc[1]]
            else:
                if only_lin:
                    values = [lin_acc[0], lin_acc[1], lin_acc[2]]
                    x=np.arange(3)
                elif only_for:
                    values= [for_acc[0], for_acc[1], for_acc[2]]
                    x=np.arange(3)
                else:
                    values = [lin_acc[0], lin_acc[1], lin_acc[2], for_acc[0], for_acc[1], for_acc[2]]
            bar_colors = ["seagreen", "darkturquoise"]

            bars = plt.bar(x + i * bar_width, values, width=bar_width, label=f'{dfs[i].name}', color=bar_colors[i%2])
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.1f}',
                         ha='center', va='bottom')
        plt.xticks(x + bar_width * (n_dfs - 1) / 2, categories)
        plt.title(plot_title)
        plt.ylim(0, 105)
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        if save:
            filename = f"{save_to}/{fico_or_scip}_{version}_train_vs_test_acc.png"
            plt.savefig(filename)
        plt.show()
        plt.close()

    if fico_or_scip == "scip":
        scip_trainset = pd.read_csv(training_frame_path, index_col=index_column)
        scip_trainset.name = f'Trainingset'

        scip_testset = pd.read_csv(test_frame_path, index_col=index_column)
        scip_testset.name = f'Testset'

        visualize_acc([scip_trainset, scip_testset], fico_or_scip = fico_or_scip, plot_title = f'SCIP',
        saving_directory = save_to, version = version, only_lin = only_linear, only_for = only_forest,
        shift_by = sgm_shift, save = save_plot)


    if fico_or_scip == "fico":
        fico_trainset = pd.read_csv(training_frame_path, index_col=index_column)
        fico_trainset.name = f'Trainingset'

        fico_testset = pd.read_csv(test_frame_path, index_col=index_column)
        fico_testset.name = f'Testset'

        visualize_acc([fico_trainset, fico_testset], fico_or_scip=fico_or_scip, plot_title=f"FICO Xpress {version}",
                      saving_directory=save_to, version=version, only_lin=only_linear, only_for=only_forest,
                      shift_by=sgm_shift, save=save_plot)

def get_sgm_list(df, shift):
    sgms = []
    for col in df.columns:
        wumr_gore = shifted_geometric_mean(df[col], shift)
        sgms.append(wumr_gore)
    return sgms

def compare_scalers_accuracy(path_to_accuracy_df, index_column,  version, deep, saving_directory, scip=True, fico=True, linear=True, forest=True):
    def plotter(value_list, title, save_dir, scip=False):

        columns = ['Accuracy', 'Mid Accuracy', 'Extreme Accuracy']
        values = [[value[0] for value in value_list], [value[1] for value in value_list],
                  [value[2] for value in value_list]]
        if scip:
            columns = ['Accuracy', 'Mid Accuracy']
            values = [[value[0] for value in value_list], [value[1] for value in value_list]]
        scalers = ['Quantile', 'Power', 'Robust', 'None']
        # Bar position setup
        x = np.arange(3)

        if scip:
            x=np.arange(2)
        width = 0.2  # Width of the bars
        colors = ['purple', 'pink', 'mediumpurple', 'teal']
        # Plotting
        fig, ax = plt.subplots()

        # Create bars for each set of values
        for i in range(4):
            ax.bar(x + i * width, [value[i] for value in values], width, label=f'{scalers[i]}',
                   color=colors[i])

        # Formatting
        ax.set_title(title)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(columns)
        ax.set_ylim([20, 100])
        ax.legend()

        plt.tight_layout()
        # Display the plot
        fig.savefig(save_dir)
        plt.show()

    if scip:
        scip_complete = pd.read_csv(path_to_accuracy_df, index_col=index_column)
        scip_complete.name = f'SCIP {version} Accuracy On Testset all Combinations'

        imputer = ['median', 'mean']
        scaler_names = ['Quantile', 'Power', 'Robust', 'None']
        ficos_lin = []
        ficos_for = []
        for imputation in imputer:
            for scaler_name in scaler_names:
                indices_lin = [ind for ind in scip_complete.index if
                               f'LinearRegression_{imputation}_{scaler_name}' in ind]
                indices_for = [ind for ind in scip_complete.index if
                               f'RandomForest_{imputation}_{scaler_name}' in ind]
                ficos_lin.append(
                    get_sgm_list(scip_complete.loc[indices_lin, ['Accuracy', 'Mid Accuracy', 'Extreme Accuracy']],
                                 10))
                ficos_for.append(
                    get_sgm_list(scip_complete.loc[indices_for, ['Accuracy', 'Mid Accuracy', 'Extreme Accuracy']],
                                 10))
        if linear:
            plotter(ficos_lin[:4], f'SCIP {version} depth={deep} LinearRegressor Median Imputed', scip=True,
                    save_dir=saving_directory)
            plotter(ficos_lin[4:], f'SCIP {version} depth={deep} LinearRegressor Mean Imputed', scip=True,
                    save_dir=saving_directory)
        if forest:
            plotter(ficos_for[:4], f'SCIP{version} depth={deep} RandomForest Median Imputed', scip=True,
                    save_dir=saving_directory)
            plotter(ficos_for[4:], f'SCIP{version} depth={deep} RandomForest Mean Imputed', scip=True,
                    save_dir=saving_directory)

    if fico:
        fico_complete = pd.read_csv(path_to_accuracy_df, index_col=index_column)
        fico_complete.name = f'FICO Xpress 9.{version} Accuracy On Testset all Combinations'

        imputer = ['median', 'mean']
        scaler_names = ['Quantile', 'Power', 'Robust', 'None']
        ficos_lin = []
        ficos_for = []
        for imputation in imputer:
            for scaler_name in scaler_names:
                indices_lin = [ind for ind in fico_complete.index if f'LinearRegression_{imputation}_{scaler_name}' in ind]
                indices_for = [ind for ind in fico_complete.index if f'RandomForest_{imputation}_{scaler_name}' in ind]
                ficos_lin.append(
                    get_sgm_list(fico_complete.loc[indices_lin, ['Accuracy', 'Mid Accuracy', 'Extreme Accuracy']], 10))
                ficos_for.append(
                    get_sgm_list(fico_complete.loc[indices_for, ['Accuracy', 'Mid Accuracy', 'Extreme Accuracy']], 10))

        if linear:
            plotter(ficos_lin[:4], f'FICO 9.{version} depth={deep} LinearRegressor Median Imputed', save_dir=
            saving_directory)
            plotter(ficos_lin[4:], f'FICO 9.{version} depth={deep} LinearRegressor Mean Imputed', save_dir=
            saving_directory)
        if forest:
            plotter(ficos_for[:4], f'FICO 9.{version} depth={deep} RandomForest Median Imputed', save_dir=
                                   saving_directory)
            plotter(ficos_for[4:], f'FICO 9.{version} depth={deep} RandomForest Mean Imputed', save_dir=
            saving_directory)

def compare_scalers_runtime(path_to_accuracy_df, index_column, saving_directory,  version, deep, scip=True, fico=True, linear=True, forest=True):
    def relative_to_mixed(values):
        mixed = values[1]
        return [(value / mixed) for value in values]

    def relative_to_int(values):
        int = values[2]
        return [(value / int) for value in values]

    def plotter(value_list, title, save_dir):
        columns = ['Mixed', 'Int',	'Predicted', 'VBS']
        values = [[value[1] for value in value_list],
                  [value[2] for value in value_list], [value[0] for value in value_list], [value[3] for value in value_list]]
        scalers = ['Quantile', 'Power', 'Robust', 'None']
        # Bar position setup
        x = np.arange(4)
        width = 0.2  # Width of the bars
        colors = ['purple', 'pink', 'mediumpurple', 'teal']
        # Plotting
        fig, ax = plt.subplots()
        # Create bars for each set of values
        for i in range(4):
            ax.bar(x + i * width, [value[i] for value in values], width, label=f'{scalers[i]}',
                   color=colors[i])

        # Formatting
        ax.set_title(title)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(columns)
        ax.set_ylim([0, 1.2])
        ax.legend()

        plt.tight_layout()

        # Display the plot
        fig.savefig(save_dir)
        plt.show()

    if scip:
        scip_complete = pd.read_csv(path_to_accuracy_df, index_col=index_column)
        scip_complete.name = f'SCIP {version} Accuracy On Testset all Combinations'

        imputer = ['median', 'mean']
        scaler_names = ['Quantile', 'Power', 'Robust', 'None']
        ficos_lin = []
        ficos_for = []
        for imputation in imputer:
            for scaler_name in scaler_names:
                indices_lin = [ind for ind in scip_complete.index if
                               f'LinearRegression_{imputation}_{scaler_name}' in ind]
                indices_for = [ind for ind in scip_complete.index if f'RandomForest_{imputation}_{scaler_name}' in ind]
                lin_sgm = get_sgm_list(scip_complete.loc[indices_lin, ['Predicted', 'Mixed', 'Int', 'VBS']], 10)
                for_sgm = get_sgm_list(scip_complete.loc[indices_for, ['Predicted', 'Mixed', 'Int', 'VBS']], 10)
                ficos_lin.append(relative_to_int(lin_sgm))
                ficos_for.append(relative_to_int(for_sgm))

        if linear:
            plotter(ficos_lin[:4], f'SCIP {version} depth={deep} LinearRegressor Median Imputed',
                    save_dir=saving_directory)
            plotter(ficos_lin[4:], f'SCIP {version} depth={deep} LinearRegressor Mean Imputed',
                    save_dir=saving_directory)
        if forest:
            plotter(ficos_for[:4], f'SCIP {version} depth={deep} RandomForest Median Imputed',
                    save_dir=saving_directory)
            plotter(ficos_for[4:], f'SCIP {version} depth={deep} RandomForest Mean Imputed',
                    save_dir=saving_directory)

    if fico:
        fico_complete = pd.read_csv(path_to_accuracy_df, index_col=index_column)
        fico_complete.name = f'FICO Xpress 9.{version} Accuracy On Testset all Combinations'

        imputer = ['median', 'mean']
        scaler_names = ['Quantile', 'Power', 'Robust', 'None']
        ficos_lin = []
        ficos_for = []

        for imputation in imputer:
            for scaler_name in scaler_names:
                indices_lin = [ind for ind in fico_complete.index if f'LinearRegression_{imputation}_{scaler_name}' in ind]
                indices_for = [ind for ind in fico_complete.index if f'RandomForest_{imputation}_{scaler_name}' in ind]
                lin_sgm  = get_sgm_list(fico_complete.loc[indices_lin, ['Predicted', 'Mixed', 'Int',	'VBS']], 10)
                for_sgm = get_sgm_list(fico_complete.loc[indices_for, ['Predicted',	'Mixed', 'Int',	'VBS']], 10)
                ficos_lin.append(relative_to_mixed(lin_sgm))
                ficos_for.append(relative_to_mixed(for_sgm))

        if linear:
            plotter(ficos_lin[:4], f'FICO 9.{version} depth={deep} LinearRegressor Median Imputed',
                    save_dir=saving_directory)
            plotter(ficos_lin[4:], f'FICO 9.{version} depth={deep} LinearRegressor Mean Imputed',
                    save_dir=saving_directory)
        if forest:
            plotter(ficos_for[:4], f'FICO 9.{version} depth={deep} RandomForest Median Imputed',
                    save_dir=saving_directory)
            plotter(ficos_for[4:], f'FICO 9.{version} depth={deep} RandomForest Mean Imputed',
                    save_dir=saving_directory)

def get_scaler_runs(df:pd.DataFrame, scaler_name:str):
    scaler_runs = [scaler_run for scaler_run in df.index if scaler_name in scaler_run]
    scaler_run_df = df.loc[scaler_runs, :]
    return scaler_run_df

def ranking_feature_importance(importance_df, feature_names):

    ranking_df = pd.DataFrame(index=feature_names, columns=['Feature', 'Linear Score', 'Forest Score'])

    linear_runs = [run for run in importance_df.columns if 'LinearRegression' in run]
    linear_df = importance_df.loc[:, linear_runs]

    forest_runs = [run for run in importance_df.columns if 'RandomForest' in run]
    forest_df = importance_df.loc[:, forest_runs]


    lin_scores = linear_df.copy()
    for col in linear_df.columns:
        # Get ranks based on absolute value (highest gets rank 0)
        ranked = linear_df[col].abs().rank(method='first', ascending=False) - 1
        lin_scores[col] = ranked.astype(int)

    for_scores = forest_df.copy()
    for col in forest_df.columns:
        ranked = forest_df[col].abs().rank(method='first', ascending=False) - 1
        for_scores[col] = ranked.astype(int)

    ranking_df['Feature'] = feature_names
    ranking_df.loc[:, 'Linear Score'] = lin_scores.sum(axis=1)
    ranking_df['Forest Score'] = for_scores.sum(axis=1)
    ranking_df['Combined'] = ranking_df['Linear Score']+ranking_df['Forest Score']
    ranking_df.sort_values(by=['Combined'], ascending=True, inplace=True)

    return ranking_df



