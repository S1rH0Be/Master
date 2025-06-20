import pandas as pd
import os

from Pys.May import scip_data
from visualize_may import shifted_geometric_mean
import matplotlib.pyplot as plt
import numpy as np

def accuracy_lin_for(data_frame, title:str, run=''):
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
        bar_colors = ['green' if val >= 75 else 'red' if val < 50 else 'orange' for val in values]
        plt.figure(figsize=(8, 5))
        plt.bar(['LinAcc', 'LinExAcc', 'ForAcc', 'ForExAcc'], values, color=bar_colors)
        plt.title(title)
        plt.ylim(min(0, min(values)*0.9), max(100, max(values)*1.1))  # Set y-axis limits for visibility
        plt.xticks(rotation=45, fontsize=6)
        # Create custom legend entries with value annotations
        # Display the plot
        plt.show()
        plt.close()
        return lin_acc, lin_ex_acc, for_acc, for_ex_acc

    lin_acc, lin_ex_acc, for_acc, for_ex_acc = visualize_acc(data_frame, filter_by='', title=title)
    return lin_acc, lin_ex_acc, for_acc, for_ex_acc

def sgm(data_frame, title:str, complete=False):

    def get_sgm_of_sgm(data_frame, shift):
        col_names = data_frame.columns.tolist()
        # Frage: SGM of relative SGMs oder von total SGMs?
        # Ich mach erstaml total sgms
        sgm_sgm_df = pd.DataFrame(columns=col_names, index=['Value'])
        for col in col_names:
            sgm_sgm_df.loc[:, col] = shifted_geometric_mean(data_frame[col], shift)
        return sgm_sgm_df

    def relative_to_mixed(value_df):
        values = value_df.iloc[0, :].tolist()
        mixed = value_df['Mixed'].iloc[0]
        values = [value/mixed for value in values]
        return values

    def visualize_sgm(data_frame, title: str = 'SGMs'):
        pred_df = data_frame.copy()

        complete_sgm_df = get_sgm_of_sgm(pred_df, pred_df.mean().mean())
        new_order = ['Int', 'Mixed', 'Predicted', 'VBS']
        complete_sgm_df = complete_sgm_df.reindex(columns=new_order)
        values_relative = relative_to_mixed(complete_sgm_df)

        labels = ['Int', 'Mixed', 'Predicted', 'VBS']
        # Determine bar colors based on conditions
        bar_colors = (['turquoise', 'magenta'])
                      # + ['green' if value >= 0.8 else 'red' if value <= 0.6 else 'blue' for value in values[3:7]])

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.bar(labels, values_relative, color=bar_colors)
        plt.title(title)
        plt.ylim(min(values_relative)*0.9, max(values_relative)*1.01)  # Set y-axis limits for visibility
        plt.xticks(rotation=45, fontsize=6)
        # Create custom legend entries with value annotations
        # legend_labels = [f"{label}: {value}" for label, value in zip(labels, values)]
        # plt.legend(bars, legend_labels, title="Values")
        # Display the plot
        plt.show()
        plt.close()
        return values_relative

    sgms = visualize_sgm(data_frame, title)
    return sgms

def split_df_by_data_set(data_frame, col_or_row:str):

    if col_or_row == 'row':
        first_col = data_frame.columns[0]
        fico_df = data_frame[data_frame[first_col].astype(str).str.contains('fico', na=False)]
        scip_default_df = data_frame[data_frame[first_col].astype(str).str.contains('scip', na=False)]
        scip_no_pseudocosts_df = data_frame[data_frame[first_col].astype(str).str.contains('no_pseudocosts', na=False)]

    else:
        fico_cols = [col_name for col_name in data_frame.columns if 'fico' in col_name]
        scip_default_cols = [col_name for col_name in data_frame.columns if 'scip' in col_name]
        no_pseudo_cols = [col_name for col_name in data_frame.columns if 'no_pseudocosts' in col_name]
        fico_df = data_frame[fico_cols]
        scip_default_df = data_frame[scip_default_cols]
        scip_no_pseudocosts_df = data_frame[no_pseudo_cols]

    return fico_df, scip_default_df, scip_no_pseudocosts_df

def get_splitup_dfs(stat_version:str, stats_be_filtered:str):
    scalers = ['None', 'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'PowerTransformer', 'QuantileTransformer']
    imputers = ['median', 'mean']
    models = ['LinearRegression', 'RandomForest']

    os.makedirs(os.path.join(f'{stat_version}/{stats_be_filtered}/SplitUp'), exist_ok=True)


    acc = [] # model_imputation_scaler <- 0-th column: row names
    imp = [] # model_imputation_scaler <- col names
    pred = [] # model_imputation_scaler <- col names
    runtime = [] # model_imputation_scaler <- 0-th column: row names

    subdirs = {'Accuracy': acc, 'Importance': imp, 'Prediction': pred, 'RunTime': runtime}
    for subdir in subdirs.keys():
        for root, _, files in os.walk(f'{stat_version}/{subdir}'):
            if not root.endswith('SplitUp'):
                for file in files:
                    if file.endswith('.csv'):
                        file_path = os.path.join(root, file)
                        try:
                            df = pd.read_csv(file_path)
                            df.name = file
                            subdirs[subdir].append((df, df.name))
                        except Exception as e:
                            print(f"Failed to read {file_path}: {e}")

    filter_by_row = ['Accuracy', 'RunTime']
    filter_by_col = ['Importance', 'Prediction']

    partial_dfs = []

    acc_df = pd.DataFrame(columns=['Run', 'Accuracy', 'Extreme Accuracy'])
    run_df = pd.DataFrame(columns=['Run', 'Int', 'Mixed', 'Predicted', 'VBS'])

    for model in models:
        for imputer in imputers:
            for scaler in scalers:
                if stats_be_filtered in filter_by_row:
                    for df_tuple in subdirs[stats_be_filtered]:
                        df = df_tuple[0]
                        if 'fico' in df_tuple[1]:
                            df_name = 'fico'
                        elif 'no_pseudo' in df_tuple[1]:
                            df_name = 'no_pseudocosts'
                        else:
                            df_name = 'scip'

                        first_col = df.columns[0]
                        df = df[df[first_col].astype(str).str.contains(f'{model}_{imputer}_{scaler}', na=False)]
                        partial_dfs.append(df)
                        df.to_csv(f'{stat_version}/{stats_be_filtered}/SplitUp/{model}_{imputer}_{scaler}_{df_name}.csv', index=False)

                        if stats_be_filtered == 'Accuracy':
                            df.set_index(df.columns[0], inplace=True, drop=True)
                            lin_acc, lin_ex_acc, for_acc, for_ex_acc = accuracy_lin_for(df, f'{model}_{imputer}_{scaler}_{df_name}')
                            accuritaet = max(lin_acc, for_acc)
                            extreme_accuracy = max(lin_ex_acc, for_ex_acc)
                            acc_df.loc[len(acc_df)] = [f'{model}_{imputer}_{scaler}_{df_name}', accuritaet, extreme_accuracy]

                        if stats_be_filtered == 'RunTime':
                            inte, mmixer, pred, vbs = sgm(df.iloc[:,1:])
                            run_df.loc[len(run_df)] = [f'{model}_{imputer}_{scaler}_{df_name}', inte, mmixer, pred, vbs]

                elif stats_be_filtered in filter_by_col:
                    for df in subdirs[stats_be_filtered]:
                        columns = df[0].columns
                        relevant_col_names = [col_name for col_name in columns if f'{model}_{imputer}_{scaler}' in col_name]
                        if stats_be_filtered == 'Importance':
                            relevant_col_names.insert(0, columns[0])
                        df = df[0][relevant_col_names]
                        partial_dfs.append(df)
                        df.to_csv(f'{stat_version}/{stats_be_filtered}/SplitUp/{stats_be_filtered}_{model}_{imputer}_{scaler}.csv',
                                  index=False)

    if len(acc_df) > 0:
        fico, default, no_pseudo = split_df_by_data_set(acc_df, 'row')
        fico = fico.sort_values(by='Accuracy', ascending=False)
        default = default.sort_values(by='Accuracy', ascending=False)
        no_pseudo = no_pseudo.sort_values(by='Accuracy', ascending=False)

        fico.to_csv(f'{stat_version}/{stats_be_filtered}/SplitUp/{stats_be_filtered}_fico_per_split.csv')
        default.to_csv(f'{stat_version}/{stats_be_filtered}/SplitUp/{stats_be_filtered}_scip_default_per_split.csv')
        no_pseudo.to_csv(f'{stat_version}/{stats_be_filtered}/SplitUp/{stats_be_filtered}_no_pseudocosts_per_split.csv')

    if len(run_df) > 0:
        fico, default, no_pseudo = split_df_by_data_set(run_df, 'row')
        fico = fico.sort_values(by='Predicted', ascending=True)
        default = default.sort_values(by='Predicted', ascending=True)
        no_pseudo = no_pseudo.sort_values(by='Predicted', ascending=True)

        fico.to_csv(f'{stat_version}/{stats_be_filtered}/SplitUp/{stats_be_filtered}_fico_per_split.csv')
        default.to_csv(f'{stat_version}/{stats_be_filtered}/SplitUp/{stats_be_filtered}_scip_default_per_split.csv')
        no_pseudo.to_csv(f'{stat_version}/{stats_be_filtered}/SplitUp/{stats_be_filtered}_no_pseudocosts_per_split.csv')

# UNSCALED
# get_splitup_dfs('/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasDiez/UnscaledLabel', 'Accuracy')
# get_splitup_dfs('/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasDiez/UnscaledLabel', 'Importance')
# get_splitup_dfs('/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasDiez/UnscaledLabel', 'Prediction')
# get_splitup_dfs('/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasDiez/UnscaledLabel', 'RunTime')

# SCALED
# get_splitup_dfs('/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasVeinteUno/ScaledLabel', 'Accuracy')
# get_splitup_dfs('/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasDiez/ScaledLabel', 'Importance')
# get_splitup_dfs('/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasDiez/ScaledLabel', 'Prediction')
# get_splitup_dfs('/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasDiez/ScaledLabel', 'RunTime')

def multiple_sgm_plot(data_frames, title: str, complete=False):
    if not isinstance(data_frames, list):
        data_frames = [data_frames]  # ensure it's a list

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

    def visualize_sgm(data_frames, title: str = 'SGMs'):
        labels = ['Int', 'Mixed', 'Predicted', 'VBS']
        x = np.arange(len(labels))
        bar_width = 0.25
        n_dfs = len(data_frames)
        colors = ['pink', 'purple', 'turquoise'][:n_dfs]

        plt.figure(figsize=(10, 6))

        for i, df in enumerate(data_frames):
            pred_df = df.copy()
            sgm_df = get_sgm_of_sgm(pred_df, pred_df.mean().mean())
            sgm_df = sgm_df.reindex(columns=labels)
            values_relative = relative_to_mixed(sgm_df)
            plt.bar(x + i * bar_width, values_relative, width=bar_width, label=f'{data_frames[i].name}', color=colors[i])

        plt.xticks(x + bar_width * (n_dfs - 1) / 2, labels)
        plt.title(title)
        plt.ylim(0, max(values_relative)*1.1)  # Ensure scale visibility
        plt.ylabel('Relative SGM (to Mixed)')
        if 'SCIP' in title:
            plt.legend(loc='lower left', frameon=True, facecolor='white', framealpha=1.0)
        else:
            plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    visualize_sgm(data_frames, title)

def multiple_accuracy_plot(data_frames, title: str, run=''):
    if not isinstance(data_frames, list):
        data_frames = [data_frames]  # wrap in list if single DataFrame is passed

    def get_sgm_series(pandas_series, shift):
        return shifted_geometric_mean(pandas_series, shift)

    def get_sgm_acc(data_frame):
        data_frame['Accuracy'] = pd.to_numeric(data_frame['Accuracy'], errors='coerce')
        data_frame['Extreme Accuracy'] = pd.to_numeric(data_frame['Extreme Accuracy'], errors='coerce')

        sgm_accuracy = get_sgm_series(data_frame['Accuracy'], data_frame['Accuracy'].mean()+0.1)
        sgm_extreme_accuracy = get_sgm_series(data_frame['Extreme Accuracy'].dropna(), data_frame['Extreme Accuracy'].dropna().mean()+0.1)
        return sgm_accuracy, sgm_extreme_accuracy

    def visualize_acc(data_frames, filter_by: str, title: str = 'Accuracy'):
        categories = ['LinAcc', 'LinExAcc', 'ForAcc', 'ForExAcc']
        n_dfs = len(data_frames)
        x = np.arange(len(categories))  # label locations
        bar_width = 0.25  # width of each bar
        colors = ['turquoise', 'darkorange', 'limegreen'][:n_dfs]

        plt.figure(figsize=(10, 6))

        for i, df in enumerate(data_frames):
            acc_df = df.copy()

            if filter_by != '':
                wanted_runs = [run for run in df.columns if filter_by in run]
                acc_df = acc_df.loc[:, wanted_runs]

            linear_rows = [row for row in acc_df.index if 'LinearRegression' in row]
            forest_rows = [row for row in acc_df.index if 'RandomForest' in row]

            linear_df = acc_df.loc[linear_rows]
            forest_df = acc_df.loc[forest_rows]

            lin_acc, lin_ex_acc, for_acc, for_ex_acc = 0, 0, 0, 0

            if len(linear_df) > 0:
                lin_acc, lin_ex_acc = get_sgm_acc(linear_df)
            if len(forest_df) > 0:
                for_acc, for_ex_acc = get_sgm_acc(forest_df)

            values = [lin_acc, lin_ex_acc, for_acc, for_ex_acc]
            plt.bar(x + i * bar_width, values, width=bar_width, label=f'{data_frames[i].name}', color=colors[i])

        plt.xticks(x + bar_width * (n_dfs - 1) / 2, categories)
        plt.title(title)
        plt.ylim(0, 105)
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    visualize_acc(data_frames, filter_by=run, title=title)

def visualisiere_sgm(treffen:str, scaler:str, unscaled=True, scaled=True):
    if unscaled:
        scip_default = pd.read_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/UnscaledLabel/RunTime/scip_sgm_runtime.csv',
                                   index_col=0)
        scip_default.name = f'SCIP Default SGM Only {scaler} Unscaled Label'

        scip_no_pseudo = pd.read_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/UnscaledLabel/RunTime/scip_no_pseudo_sgm_runtime.csv',
                                     index_col=0)
        scip_no_pseudo.name = f'SCIP No Pseudocosts SGM Only {scaler} Unscaled Label'

        fico_default = pd.read_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/UnscaledLabel/RunTime/fico_sgm_runtime.csv',
                                   index_col=0)
        fico_default.name = f'FICO Xpress SGM Only {scaler} Unscaled Label'

        darter = [scip_default, scip_no_pseudo, fico_default]
        for dart in darter:
            sgm(dart, title=dart.name)

    if scaled:
        scip_default = pd.read_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/ScaledLabel/RunTime/scip_sgm_runtime.csv',
                                   index_col=0)
        scip_default.name = f'SCIP Default SGM Only {scaler} Scaled Label'

        scip_no_pseudo = pd.read_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/ScaledLabel/RunTime/scip_no_pseudo_sgm_runtime.csv',
                                     index_col=0)
        scip_no_pseudo.name = f'SCIP No Pseudocosts SGM Only {scaler} Scaled Label'

        fico_default = pd.read_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/ScaledLabel/RunTime/fico_sgm_runtime.csv',
                                   index_col=0)
        fico_default.name = f'FICO Xpress SGM Only {scaler} Scaled Label'

        darter = [scip_default, scip_no_pseudo, fico_default]
        for dart in darter:
            sgm(dart, title=dart.name)

def visualisiere_accuracy(treffen: str, scaler: str, unscaled=True, scaled=True):

    if unscaled:
        scip_default = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/UnscaledLabel/Accuracy/scip_acc_df.csv',
            index_col=0)
        scip_default.name = f'SCIP Default Accuracy Only {scaler} Unscaled Label'

        scip_no_pseudo = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/UnscaledLabel/Accuracy/scip_no_pseudo_acc_df.csv',
            index_col=0)
        scip_no_pseudo.name = f'SCIP No Pseudocosts Accuracy Only {scaler} Unscaled Label'

        fico_default = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/UnscaledLabel/Accuracy/fico_acc_df.csv',
            index_col=0)
        fico_default.name = f'FICO Xpress Accuracy Only {scaler} Unscaled Label'

        dfs = [scip_default, scip_no_pseudo, fico_default]

        multiple_accuracy_plot(dfs, title='Test')

    if scaled:
        scip_default = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/ScaledLabel/Accuracy/scip_acc_df.csv',
            index_col=0)
        scip_default.name = f'SCIP Default Accuracy Only {scaler} Scaled Label'

        scip_no_pseudo = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/ScaledLabel/Accuracy/scip_no_pseudo_acc_df.csv',
            index_col=0)
        scip_no_pseudo.name = f'SCIP No Pseudocosts Accuracy Only {scaler} Scaled Label'

        fico_default = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/ScaledLabel/Accuracy/fico_acc_df.csv',
            index_col=0)
        fico_default.name = f'FICO Xpress Accuracy Only {scaler} Scaled Label'

        dfs = [scip_default, scip_no_pseudo, fico_default]
        multiple_accuracy_plot(dfs, title='Test')

def compare_scalers_accuracy(scaled_or_unscaled:str, treffen='TreffenMasOnce', scip=True, fico=True):
    if scip:
        scip_quantile = pd.read_csv(
                    f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/SoloQuantile/{scaled_or_unscaled}Label/Accuracy/scip_acc_df.csv',
                    index_col=0)
        scip_quantile.name = f'SCIP Default Accuracy Only Quantile {scaled_or_unscaled} Label'

        scip_yeo = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/SoloYeoJohnson/{scaled_or_unscaled}Label/Accuracy/scip_acc_df.csv',
            index_col=0)
        scip_yeo.name = f'SCIP Default Accuracy Only YeoJohnson {scaled_or_unscaled} Label'

        scip_robust = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/SoloRobust/{scaled_or_unscaled}Label/Accuracy/scip_acc_df.csv',
            index_col=0)
        scip_robust.name = f'SCIP Default Accuracy Only Robust {scaled_or_unscaled} Label'


        scippies = [scip_quantile, scip_yeo, scip_robust]

        multiple_accuracy_plot(scippies, title='SCIP Comparison Accuracy Scaled')

    if fico:
        fico_quantile = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/SoloQuantile/{scaled_or_unscaled}Label/Accuracy/fico_acc_df.csv',
            index_col=0)
        fico_quantile.name = f'FICO Xpress Accuracy Only Quantile {scaled_or_unscaled} Label'

        fico_yeo = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/SoloYeoJohnson/{scaled_or_unscaled}Label/Accuracy/fico_acc_df.csv',
            index_col=0)
        fico_yeo.name = f'FICO Xpress Accuracy Only Yeo-Johnson {scaled_or_unscaled} Label'

        fico_robust = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/SoloRobust/{scaled_or_unscaled}Label/Accuracy/fico_acc_df.csv',
            index_col=0)
        fico_robust.name = f'FICO Xpress Accuracy Only Robust {scaled_or_unscaled} Label'

        ficos = [fico_quantile, fico_yeo, fico_robust]
        multiple_accuracy_plot(ficos, title='FICO Comparison Accuracy Scaled')

def compare_scalers_sgm(scaled_or_unscaled:str, treffen='TreffenMasOnce', scip=True, fico=True):
    if scip:
        scip_quantile = pd.read_csv(
                    f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/SoloQuantile/{scaled_or_unscaled}Label/SGM/scip_sgm_runtime.csv',
                    index_col=0)
        scip_quantile.name = f'SCIP Default SGM Only Quantile {scaled_or_unscaled} Label'

        scip_yeo = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/SoloYeoJohnson/{scaled_or_unscaled}Label/SGM/scip_sgm_runtime.csv',
            index_col=0)
        scip_yeo.name = f'SCIP Default SGM Only YeoJohnson {scaled_or_unscaled} Label'

        scip_robust = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/SoloRobust/{scaled_or_unscaled}Label/SGM/scip_sgm_runtime.csv',
            index_col=0)
        scip_robust.name = f'SCIP Default SGM Only Robust {scaled_or_unscaled} Label'


        scippies = [scip_quantile, scip_yeo, scip_robust]

        multiple_sgm_plot(scippies, title='SCIP Comparison SGM Scaled')

    if fico:
        fico_quantile = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/SoloQuantile/{scaled_or_unscaled}Label/SGM/fico_sgm_runtime.csv',
            index_col=0)
        fico_quantile.name = f'FICO Xpress SGM Only Quantile {scaled_or_unscaled} Label'

        fico_yeo = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/SoloYeoJohnson/{scaled_or_unscaled}Label/SGM/fico_sgm_runtime.csv',
            index_col=0)
        fico_yeo.name = f'FICO Xpress SGM Only Yeo-Johnson {scaled_or_unscaled} Label'

        fico_robust = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/SoloRobust/{scaled_or_unscaled}Label/SGM/fico_sgm_runtime.csv',
            index_col=0)
        fico_robust.name = f'FICO Xpress SGM Only Robust {scaled_or_unscaled} Label'

        ficos = [fico_quantile, fico_yeo, fico_robust]
        multiple_sgm_plot(ficos, title='FICO Comparison SGM Scaled')

def compare_train_vs_test(scaled_or_unscaled:str, scaler, treffen='TreffenMasDoce', scip=True, fico=True):
    # Accuracy
    if scip:
        scip_test = pd.read_csv(
                    f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/{scaled_or_unscaled}Label/Accuracy/scip_acc_df.csv',
                    index_col=0)
        scip_test.name = f'SCIP Default Accuracy Testset {scaler} {scaled_or_unscaled} Label'

        scip_train = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/{scaled_or_unscaled}Label/Accuracy/scip_acc_trainset.csv',
            index_col=0)
        scip_train.name = f'SCIP Default Accuracy Trainingset {scaler} {scaled_or_unscaled} Label'

        train_test = [scip_test, scip_train]
        multiple_accuracy_plot(train_test, 'Comparison Accuracy Train vs Testset')

    if fico:
        fico_test = pd.read_csv(
                    f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/{scaled_or_unscaled}Label/Accuracy/fico_acc_df.csv',
                    index_col=0)
        fico_test.name = f'FICO Xpress Accuracy Testset {scaler} {scaled_or_unscaled} Label'

        fico_train = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/{scaled_or_unscaled}Label/Accuracy/fico_acc_trainset.csv',
            index_col=0)
        fico_train.name = f'FICO Xpress Accuracy Trainingset {scaler} {scaled_or_unscaled} Label'

        train_test = [fico_test, fico_train]
        multiple_accuracy_plot(train_test, 'Comparison Accuracy Train vs Testset')


    # RunTime
    if scip:
        scip_test = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/{scaled_or_unscaled}Label/SGM/scip_sgm_runtime.csv',
            index_col=0)
        scip_test.name = f'FICO Xpress SGM Testset {scaler} {scaled_or_unscaled} Label'

        scip_train = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/{scaled_or_unscaled}Label/SGM/scip_sgm_trainset.csv',
            index_col=0)
        scip_train.name = f'FICO Xpress SGM Trainset {scaler} {scaled_or_unscaled} Label'

        scips = [scip_test, scip_train]
        multiple_sgm_plot(scips, title=f'SCIP Comparison RunTime SGM Train vs Testset {scaler} {scaled_or_unscaled} Label')

    if fico:
        fico_test = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/{scaled_or_unscaled}Label/SGM/fico_sgm_runtime.csv',
            index_col=0)
        fico_test.name = f'FICO Xpress SGM Testset {scaler} {scaled_or_unscaled} Label'

        fico_train = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/{scaled_or_unscaled}Label/SGM/fico_sgm_trainset.csv',
            index_col=0)
        fico_train.name = f'FICO Xpress SGM Trainset {scaler} {scaled_or_unscaled} Label'

        ficos = [fico_test, fico_train]
        multiple_sgm_plot(ficos, title=f'FICO Xpress Comparison RunTime SGM Train vs Testset {scaler} {scaled_or_unscaled} Label')

def visualize_fico_on_scip(scaled_or_unscaled:str, scaler, treffen='TreffenMasTrece'):
    # Accuracy
    scip_acc = pd.read_csv(
        f'/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasDoce/Solo{scaler}/{scaled_or_unscaled}Label/Accuracy/scip_acc_df.csv',
        index_col=0)
    scip_acc.name = f'SCIP Default Accuracy {scaler} {scaled_or_unscaled} Label'

    fico_on_scip_acc = pd.read_csv(
        f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/FICOonSCIP/Solo{scaler}/{scaled_or_unscaled}Label/Accuracy/fico_on_scip_acc_df.csv',
        index_col=0)
    fico_on_scip_acc.name = f'FICO XPRESS On SCIP Accuracy {scaler} {scaled_or_unscaled} Label'

    scip_vs_fico_on_scip_acc = [scip_acc, fico_on_scip_acc]
    multiple_accuracy_plot(scip_vs_fico_on_scip_acc, f'FICO Xpress vs SCIP on SCIP Accuracy {scaled_or_unscaled} Label')

    # RunTime
    scip_sgm = pd.read_csv(
        f'/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasDoce/Solo{scaler}/{scaled_or_unscaled}Label/SGM/scip_sgm_runtime.csv',
        index_col=0)
    scip_sgm.name = f'SCIP SGM {scaler} {scaled_or_unscaled} Label'

    fico_on_scip_sgm = pd.read_csv(
        f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/FICOonSCIP/Solo{scaler}/{scaled_or_unscaled}Label/SGM/fico_on_scip_sgm_runtime.csv',
        index_col=0)
    fico_on_scip_sgm.name = f'FICO Xpress on SCIP SGM {scaler} {scaled_or_unscaled} Label'

    fico_vs_scip_on_scip_sgm = [scip_sgm, fico_on_scip_sgm]

    multiple_sgm_plot(fico_vs_scip_on_scip_sgm,
                      title=f'FICO Xpress vs SCIP on SCIP Comparison RunTime SGM {scaler} {scaled_or_unscaled} Label')

def ranking_feature_importance_fico(importance_df, title):
    feature_names = ['Matrix Equality Constraints', 'Matrix Quadratic Elements',
       'Matrix NLP Formula', 'Presolve Columns', 'Presolve Global Entities',
       '#nodes in DAG', '#integer violations at root',
       '#nonlinear violations at root', '% vars in DAG (out of all vars)',
       '% vars in DAG unbounded (out of vars in DAG)',
       '% vars in DAG integer (out of vars in DAG)',
       '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)',
       'Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
       'Avg ticks for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
       'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Mixed',
       'Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
       '#spatial branching entities fixed (at the root) Mixed',
       'Avg coefficient spread for convexification cuts Mixed']
    ranking_df = pd.DataFrame(index=feature_names, columns=['Linear Score', 'Forest Score'])

    linear_df = importance_df.iloc[:,:100]
    forest_df = importance_df.iloc[:,100:]
    lin_scores = pd.DataFrame(index=feature_names)
    for_scores = pd.DataFrame(index=feature_names)
    for col in linear_df.columns:
        # Get ranks based on absolute value (highest gets rank 0)
        ranked = linear_df[col].abs().rank(method='first', ascending=False) - 1
        lin_scores[col] = ranked.astype(int)

    for col in forest_df.columns:
        ranked = forest_df[col].abs().rank(method='first', ascending=False) - 1
        for_scores[col] = ranked.astype(int)

    ranking_df['Linear Score'] = lin_scores.sum(axis=1)
    ranking_df['Forest Score'] = for_scores.sum(axis=1)

    return ranking_df

def importance(treffen, scaler):
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
        importance_bar_plot(linear_importance_df, f'LinearRegression {title}')
        importance_bar_plot(forest_importance_df, f'RandomForest {title}')

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

    def plot_importance_score(data_frame, title):
        importance_df = get_importance_score_df(data_frame)
        importance_df.plot(kind='bar', figsize=(10, 6), color=['turquoise', 'magenta'])
        plt.title(title)
        plt.ylabel('Importance Score')
        plt.xlabel('Feature')
        plt.xticks(rotation=270, fontsize=6)
        plt.show()
        plt.close()

    fico_impo_scaled = pd.read_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/ScaledLabel/Importance/fico_importance_df.csv',
                            index_col=0)
    fico_impo_unscaled = pd.read_csv(
        f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/UnscaledLabel/Importance/fico_importance_df.csv',
        index_col=0)

    fico_impo_all_runs_scaled = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasDiez/ScaledLabel/Importance/fico_importance_df.csv')
    fico_impo_all_runs_unscaled = pd.read_csv(
        '/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasDiez/UnscaledLabel/Importance/fico_importance_df.csv')

    scip_impo_scaled = pd.read_csv(
        f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/ScaledLabel/Importance/scip_importance_df.csv',
        index_col=0)
    scip_impo_unscaled = pd.read_csv(
        f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffen}/Solo{scaler}/UnscaledLabel/Importance/scip_importance_df.csv',
        index_col=0)

    # scip_impo_all_runs_scaled = pd.read_csv(
    #     '/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasDiez/ScaledLabel/Importance/scip_importance_df.csv')
    # scip_impo_all_runs_unscaled = pd.read_csv(
    #     '/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasDiez/UnscaledLabel/Importance/scip_importance_df.csv')

    # plot_importances_by_regressor(scip_impo_all_runs_unscaled, 'Feature Importance Unscaled Label')
    # plot_importances_by_regressor(scip_impo_all_runs_scaled, 'Feature Importance Scaled Label')

    plot_importance_score(scip_impo_unscaled, f'SCIP Importance {scaler} Unscaled Label')
    plot_importances_by_regressor(scip_impo_unscaled, f'SCIP {scaler} Unscaled Label')

    plot_importance_score(scip_impo_scaled, f'SCIP Importance {scaler} Scaled Label')
    plot_importances_by_regressor(scip_impo_scaled, f'SCIP Importance {scaler} Scaled Label')

def time_loss_per_prediction(scip_data, prediction_df):
    """
    Think about: Do i take the sgm of all prediction for a given instance?
                Also für linear gibts nur eine prediction:) weil keine randomness (only rand is choice of train/test set)
    """
    def get_time_loss_per_prediction(df):
        df['Time Loss'] = 0.0

        for index, row in df.iterrows():
            if row['Prediction']<0:
                if row['Cmp Final solution time (cumulative)']>0:
                    df.loc[index, 'Time Loss'] = np.round(row['Final solution time (cumulative) Int']-row['Final solution time (cumulative) Mixed'],6)
            else:
                if row['Cmp Final solution time (cumulative)']<0:
                    df.loc[index,'Time Loss'] = np.round(row['Final solution time (cumulative) Mixed'] - row[
                        'Final solution time (cumulative) Int'], 6)
        return df

    # create time_df
    time_df = pd.DataFrame(index=scip_data.index, columns=['Matrix Name', 'Prediction',
                                                           'Final solution time (cumulative) Mixed',
                                                           'Final solution time (cumulative) Int',
                                                           'Cmp Final solution time (cumulative)',
                                                           'Virtual Best'])
    time_df.loc[:, ['Matrix Name', 'Final solution time (cumulative) Mixed',
                    'Final solution time (cumulative) Int', 'Cmp Final solution time (cumulative)', 'Virtual Best']] = scip_data.loc[:,['Matrix Name', 'Final solution time (cumulative) Mixed',
                    'Final solution time (cumulative) Int', 'Cmp Final solution time (cumulative)',
                                                                                                'Virtual Best']]
    linear_prediction = prediction_df.iloc[:, 5]
    time_df.loc[:, 'Prediction'] = linear_prediction
    linear_time_df = get_time_loss_per_prediction(time_df)
    linear_time_df.to_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/BilderTreffen/ScaledLabel/linear_time_loss.csv', index=False)

    forest_prediction = prediction_df.iloc[:, 100:]

# TODO I think i need regression to use all predictions not only relevant, such that everyth9ng matches
#  for example: len(prediction)=234!=282

scip_data_df = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/cleaned_scip/scip_default_clean_data.csv')

quantile_scaled_prediction_df = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasTrece/FICOonSCIP/SoloQuantile/ScaledLabel/Prediction/fico_on_scip_prediction_df.csv')
# time_loss_per_prediction(scip_data_df, quantile_scaled_prediction_df)

