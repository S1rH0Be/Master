import pandas as pd
import os


from visualize_july import shifted_geometric_mean
import matplotlib.pyplot as plt
import numpy as np

base_data_directory = '/Users/fritz/Downloads/ZIB/Master/October/Runs/FinalScip'


def setup_directory(new_directory):
    # Setup directory
    os.makedirs(os.path.join(f'{new_directory}'), exist_ok=True)

def get_sgm_of_cols(df, shift):
    col_names = df.columns.tolist()
    # Frage: SGM of relative SGMs oder von total SGMs?
    # Ich mach erstaml total sgms
    sgm_sgm_df = pd.DataFrame(columns=col_names, index=['Value'])
    for col in col_names:
        sgm_sgm_df.loc[:, col] = shifted_geometric_mean(df[col], shift)
    return sgm_sgm_df


def sgm(data_frame, title:str):

    def get_sgm_of_sgm(df, shift):
        col_names = df.columns.tolist()
        # Frage: SGM of relative SGMs oder von total SGMs?
        # Ich mach erstaml total sgms
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
        #plt.title(plot_title)
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
            for scaler_name in scalers:
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
                        df = df[df[first_col].astype(str).str.contains(f'{model}_{imputer}_{scaler_name}', na=False)]
                        partial_dfs.append(df)
                        df.to_csv(f'{stat_version}/{stats_be_filtered}/SplitUp/{model}_{imputer}_{scaler_name}_{df_name}.csv', index=False)

                        if stats_be_filtered == 'Accuracy':
                            df.set_index(df.columns[0], inplace=True, drop=True)
                            lin_acc, lin_ex_acc, for_acc, for_ex_acc = accuracy_lin_for(df, f'{model}_{imputer}_{scaler_name}_{df_name}')
                            accuritaet = max(lin_acc, for_acc)
                            extreme_accuracy = max(lin_ex_acc, for_ex_acc)
                            acc_df.loc[len(acc_df)] = [f'{model}_{imputer}_{scaler_name}_{df_name}', accuritaet, extreme_accuracy]

                        if stats_be_filtered == 'RunTime':
                            inte, mmixer, pred, vbs = sgm(df.iloc[:,1:], title=f'SGM_{model}_{imputer}_{scaler_name}_{df_name}')
                            run_df.loc[len(run_df)] = [f'{model}_{imputer}_{scaler_name}_{df_name}', inte, mmixer, pred, vbs]

                elif stats_be_filtered in filter_by_col:
                    for df in subdirs[stats_be_filtered]:
                        columns = df[0].columns
                        relevant_col_names = [col_name for col_name in columns if f'{model}_{imputer}_{scaler_name}' in col_name]
                        if stats_be_filtered == 'Importance':
                            relevant_col_names.insert(0, columns[0])
                        df = df[0][relevant_col_names]
                        partial_dfs.append(df)
                        df.to_csv(f'{stat_version}/{stats_be_filtered}/SplitUp/{stats_be_filtered}_{model}_{imputer}_{scaler_name}.csv',
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

def train_vs_test_sgm(data_frame_path, version: str, fico_or_scip, save_to):
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


    def visualize_sgm(dfs, fico_or_scip, version="", plot_title: str = 'SGMs', saving_directory="", last_minute_lin=True, last_minute_for=False):
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
            sgm_lin_df = get_sgm_of_sgm(lin_pred_df,10)# lin_pred_df.mean().mean())
            sgm_for_df = get_sgm_of_sgm(for_pred_df, 10)#for_pred_df.mean().mean())
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

            if last_minute_lin:
                labels = ['Int', 'Linear', 'VBS']
                x=np.arange(3)
                values_relative_lin = relative_to_mixed(sgm_lin_df)
                values_relative_for = relative_to_mixed(sgm_for_df)
                values_relative = [values_relative_lin[1], values_relative_lin[0],
                                   values_relative_for[3]]
            if last_minute_for:
                labels = ['Int', 'Forest', 'VBS']
                values_relative_lin = relative_to_mixed(sgm_lin_df)
                values_relative_for = relative_to_mixed(sgm_for_df)
                values_relative = [values_relative_lin[1], values_relative_for[0],
                                   values_relative_for[3]]
                x = np.arange(3)
            print(x + i * bar_width, values_relative)
            # print("SGM: ", values_relative)
            bars = plt.bar(x + i * bar_width, values_relative, width=bar_width, label=f'{dfs[i].name}', color=colors[i%2])
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.2f}',
                         ha='center', va='bottom')

        plt.xticks(x + bar_width * (n_dfs - 1) / 2, labels)
        #plt.title(f"{plot_title} {version}")
        plt.ylim(0, 1.1)  # Ensure scale visibility
        plt.ylabel('Relative SGM (to Mixed)')

        if 'SCIP' in plot_title:
            plt.legend(frameon=True, facecolor='white', framealpha=1.0)
        else:
            plt.legend()
        plt.tight_layout()
        filename = f"{saving_directory}/{fico_or_scip}_{version}_train_vs_test_sgm.png"
        plt.savefig(filename)
        plt.show()
        plt.close()

    if fico_or_scip == "scip":
        scip_testset = pd.read_csv(
            f'{data_frame_path}/SGM/scip_sgm_runtime.csv',
            index_col=0)
        scip_testset.name = f'Testset'
        scip_trainset = pd.read_csv(
            f'{data_frame_path}/SGM/scip_sgm_trainset.csv',
            index_col=0)
        scip_trainset.name = f'Trainingset'
        visualize_sgm([scip_testset, scip_trainset], fico_or_scip=fico_or_scip, plot_title=f'SCIP {version}',
                      saving_directory=save_to)

    if fico_or_scip == "fico":
        print(data_frame_path)
        fico_testset = pd.read_csv(
            f'{data_frame_path}/SGM/fico_sgm_runtime.csv',
            index_col=0)
        # chosen_seed = [ind for ind in fico_testset.index if "3094311947" in ind]
        # fico_testset = fico_testset.loc[chosen_seed]
        fico_testset.name = f'Testset'

        fico_trainset = pd.read_csv(
            f'{data_frame_path}/SGM/fico_sgm_trainset.csv',
            index_col=0)
        # fico_trainset = fico_trainset.loc[chosen_seed]
        fico_trainset.name = f'Trainingset'
        visualize_sgm([fico_testset, fico_trainset], fico_or_scip=fico_or_scip, plot_title="FICO Xpress",
                      saving_directory=save_to, version=version)
# checked
def train_vs_test_acuracy(data_frame_path, version: str, fico_or_scip, save_to):
    setup_directory(save_to)
    def get_sgm_series(pandas_series, shift):
        if len(pandas_series) == 0:
            return 0
        return shifted_geometric_mean(pandas_series, shift)

    def get_sgm_acc(data_frame, fico_or_scip):

        data_frame['Accuracy'] = pd.to_numeric(data_frame['Accuracy'], errors='coerce')
        data_frame['Mid Accuracy'] = pd.to_numeric(data_frame['Mid Accuracy'], errors='coerce')
        if fico_or_scip == 'fico':
            data_frame['Extreme Accuracy'] = pd.to_numeric(data_frame['Extreme Accuracy'], errors='coerce')

        sgm_accuracy = get_sgm_series(data_frame['Accuracy'], shift=10)#data_frame['Accuracy'].mean() + 0.1)
        sgm_mid_accuracy = get_sgm_series(data_frame['Mid Accuracy'], shift=10)#data_frame['Mid Accuracy'].mean() + 0.1)
        return_list = [sgm_accuracy, sgm_mid_accuracy]
        if fico_or_scip == 'fico':
            if len(data_frame['Extreme Accuracy'])==0:
                return_list.append(0)
            else:
                sgm_extreme_accuracy = get_sgm_series(data_frame['Extreme Accuracy'].dropna(),
                                                 shift=10)# data_frame['Extreme Accuracy'].dropna().mean() + 0.1)
                return_list.append(sgm_extreme_accuracy)

        return return_list

    def visualize_acc(dfs, fico_or_scip, plot_title:str, saving_directory:str, last_minute_lin=False, last_minute_for=True):

        if fico_or_scip == 'fico':
            categories = ['Linear', 'Linear MidLabel', 'Linear LargeLabel', 'Forest', 'Forest MidLabel', 'Forest LargeLabel']
        else:
            categories = ['Linear', 'Linear MidLabel', 'Forest', 'Forest MidLabel']
        n_dfs = len(dfs)
        x = np.arange(len(categories))  # label locations
        bar_width = 0.25  # width of each bar
        colors = ['turquoise', 'darkorange', 'limegreen', 'purple', 'red'][:n_dfs]

        plt.figure(figsize=(10, 6))

        for i, df in enumerate(dfs):
            acc_df = df.copy()

            linear_rows = [row for row in acc_df.index if 'LinearRegression' in row]
            forest_rows = [row for row in acc_df.index if 'RandomForest' in row]

            linear_df = acc_df.loc[linear_rows]
            forest_df = acc_df.loc[forest_rows]

            lin_acc, lin_mid_acc, lin_ex_acc, for_acc, for_mid_acc, for_ex_acc = 0, 0, 0, 0, 0, 0

            if len(linear_df) > 0:
                lin_acc = get_sgm_acc(linear_df, fico_or_scip)
            if len(forest_df) > 0:
                for_acc = get_sgm_acc(forest_df, fico_or_scip)
            if fico_or_scip == 'scip':
                values = [lin_acc[0], lin_acc[1], for_acc[0], for_acc[1]]
            else:
                values = [lin_acc[0], lin_acc[1], lin_acc[2], for_acc[0], for_acc[1], for_acc[2]]
            if last_minute_lin:
                values = [lin_acc[0], lin_acc[1], lin_acc[2]]
                x=np.arange(3)
            elif last_minute_for:
                values= [for_acc[0], for_acc[1], for_acc[2]]
                x=np.arange(3)
            bar_colors = ["purple", "deeppink"]

            if last_minute_lin:
                categories = ['Linear', 'Linear MidLabel', 'Linear LargeLabel']  # , 'Forest', 'Forest MidLabel',
                # 'Forest LargeLabel']
            if last_minute_for:
                categories = ['Forest', 'Forest MidLabel', 'Forest LargeLabel']
            print(len(values))
            print(x, i * bar_width)
            bars = plt.bar(x + i * bar_width, values, width=bar_width, label=f'{dfs[i].name}', color=bar_colors[i%2])
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.1f}',
                         ha='center', va='bottom')
        print(x + bar_width * (n_dfs - 1) / 2, categories)
        plt.xticks(x + bar_width * (n_dfs - 1) / 2, categories)
        #plt.title(plot_title)
        plt.ylim(0, 105)
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        filename = f"{save_to}/{fico_or_scip}_{version}_train_vs_test_acc.png"
        plt.savefig(filename)
        plt.show()
        plt.close()

    if fico_or_scip == "scip":
        scip_testset = pd.read_csv(
            f'{data_frame_path}/Accuracy/scip_acc_df.csv',
            index_col=0)
        scip_testset.name = f'Testset'
        scip_trainset = pd.read_csv(
            f'{data_frame_path}/Accuracy/scip_acc_trainset.csv',
            index_col=0)
        scip_trainset.name = f'Trainingset'
        visualize_acc([scip_testset, scip_trainset], fico_or_scip=fico_or_scip, plot_title=f'SCIP {version}', saving_directory=save_to)

    if fico_or_scip == "fico":
        fico_testset = pd.read_csv(
            f'{data_frame_path}/Accuracy/fico_acc_df.csv',
            index_col=0)
        fico_testset.name = f'Accuracy on Testset'

        fico_trainset = pd.read_csv(
            f'{data_frame_path}/Accuracy/fico_acc_trainset.csv',
            index_col=0)
        fico_trainset.name = f'Accuracy on Trainingset'
        visualize_acc([fico_testset, fico_trainset], fico_or_scip=fico_or_scip, plot_title=f"FICO Xpress {version}", saving_directory=save_to)
# checked
def visualisiere_sgm(treffen:str, version:str, scaler_name:str, unscaled=True, scaled=True, scip=False, fico=False):
    if unscaled:
        darter = []
        if scip:
            scip_default = pd.read_csv(f'{treffen}/SGM/scip_sgm_runtime.csv',
                                       index_col=0)
            scip_default.name = f'SCIP Runtime on Testset'
            darter.append(scip_default)
            scip_train = pd.read_csv(f'{treffen}/SGM/scip_sgm_trainset.csv',
                                       index_col=0)
            scip_train.name = f'SCIP Runtime on Trainingset'
            darter.append(scip_train)

        if fico:
            print(f'{treffen}/SGM/fico_sgm_runtime.csv')
            fico_default = pd.read_csv(f'{treffen}/SGM/fico_sgm_runtime.csv',
                                       index_col=0)
            fico_train = pd.read_csv(f'{treffen}/SGM/fico_sgm_trainset.csv',
                                       index_col=0)
            fico_default.name = f'FICO Xpress {version} Runtime on Testset'
            fico_train.name = f'FICO Xpress {version} Runtime on Trainingset'
            darter.append(fico_default)
            darter.append(fico_train)
        for dart in darter:
            sgm(dart, title=dart.name)

    if scaled:
        darter = []
        if scip:
            scip_default = pd.read_csv(f'/Users/fritz/Downloads/ZIB/Master/October/Runs/{treffen}/{scaler_name}/ScaledLabel/SGM/scip_sgm_runtime.csv',
                                       index_col=0)
            scip_default.name = f'SCIP Default SGM Only {scaler_name} Scaled Label'
            darter.append(scip_default)

        if fico:
            fico_default = pd.read_csv(f'/Users/fritz/Downloads/ZIB/Master/October/Runs/{treffen}/{scaler_name}/ScaledLabel/SGM/fico_sgm_runtime.csv',
                                       index_col=0)
            fico_default.name = f'FICO Xpress SGM Only {scaler_name} Scaled Label'
            darter.append(fico_default)

        for dart in darter:
            sgm(dart, title=dart.name)
# checked
def visualisiere_accuracy(treffen: str, directory_for_saving_plot, version:str, scip_or_fico: str, scip=False, fico=False):
    def get_sgm_series(pandas_series, shift):
        return shifted_geometric_mean(pandas_series, shift)

    def get_sgm_acc(data_frame):
        data_frame['Accuracy'] = pd.to_numeric(data_frame['Accuracy'], errors='coerce')
        data_frame['Mid Accuracy'] = pd.to_numeric(data_frame['Mid Accuracy'], errors='coerce')
        data_frame['Extreme Accuracy'] = pd.to_numeric(data_frame['Extreme Accuracy'], errors='coerce')

        sgm_accuracy = get_sgm_series(data_frame['Accuracy'], data_frame['Accuracy'].mean() + 0.1)
        sgm_mid_accuracy = get_sgm_series(data_frame['Mid Accuracy'], data_frame['Mid Accuracy'].mean() + 0.1)
        sgm_extreme_accuracy = get_sgm_series(data_frame['Extreme Accuracy'].dropna(),
                                              data_frame['Extreme Accuracy'].dropna().mean() + 0.1)
        return sgm_accuracy, sgm_mid_accuracy, sgm_extreme_accuracy

    def visualize_acc(dfs, scip_or_fico:str, save_to,  plot_title: str = 'Accuracy'):
        if scip_or_fico == "scip":
            categories = ['Linear', 'Linear MidLabel', 'Forest', 'Forest MidLabel']
        elif scip_or_fico == "fico":
            categories = ['Linear', 'Linear MidLabel', 'Linear Large Label', 'Forest', 'Forest MidLabel', 'Forest Large Label']
        else:
            pass
        n_dfs = len(dfs)
        x = np.arange(len(categories))  # label locations
        bar_width = 0.5  # width of each bar

        plt.figure(figsize=(10, 6))

        for i, df in enumerate(dfs):
            acc_df = df.copy()

            linear_rows = [row for row in acc_df.index if 'LinearRegression' in row]
            forest_rows = [row for row in acc_df.index if 'RandomForest' in row]

            linear_df = acc_df.loc[linear_rows]
            forest_df = acc_df.loc[forest_rows]

            lin_acc, lin_mid_acc, lin_ex_acc, for_acc, for_mid_acc, for_ex_acc = 0, 0, 0, 0, 0, 0

            if len(linear_df) > 0:
                lin_acc, lin_mid_acc, lin_ex_acc = get_sgm_acc(linear_df)
            if len(forest_df) > 0:
                for_acc, for_mid_acc, for_ex_acc = get_sgm_acc(forest_df)
            if scip_or_fico == "scip":
                values = [lin_acc, lin_mid_acc, for_acc, for_mid_acc]
                bar_colors = ['purple', 'purple', 'green', 'green']
            else:
                values = [lin_acc, lin_mid_acc, lin_ex_acc, for_acc, for_mid_acc, for_ex_acc]
                bar_colors = ['purple', 'purple', 'purple', 'green', 'green', 'green']


            bars = plt.bar(x + i * bar_width, values, width=bar_width, label=f'{dfs[i].name}', color=bar_colors)

        plt.xticks(x + bar_width * (n_dfs - 1) / 2, categories)
        #plt.title(plot_title)
        plt.ylim(0, 105)
        plt.ylabel('Accuracy')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom')
        plt.tight_layout()
        # save plot
        file_name_add_on = plot_title.replace(" ", "_")
        filename = f"{save_to}/{file_name_add_on}.png"
        plt.savefig(filename)
        plt.show()
        plt.close()

    if scip:
        scip_default = pd.read_csv(
            f'{treffen}/Accuracy/scip_acc_df.csv',
            index_col=0)
        scip_default.name = f'SCIP Accuracy on Testset'
        visualize_acc([scip_default], scip_or_fico, scip_default.name)
        scip_trainset = pd.read_csv(
            f'{treffen}/Accuracy/scip_acc_trainset.csv',
            index_col=0)
        scip_trainset.name = f'SCIP Accuracy on Trainingset'
        visualize_acc([scip_trainset], scip_or_fico, scip_trainset.name)

    if fico:
        fico_default = pd.read_csv(
            f'{treffen}/Accuracy/fico_acc_df.csv',
            index_col=0)
        fico_default.name = f'FICO Xpress 9.{version} Accuracy on Testset'
        visualize_acc([fico_default], save_to=directory_for_saving_plot, scip_or_fico=scip_or_fico, plot_title=fico_default.name)

        fico_trainset = pd.read_csv(
            f'{treffen}/Accuracy/fico_acc_trainset.csv',
            index_col=0)
        fico_trainset.name = f'FICO Xpress 9.{version} Accuracy on Trainingset'
        visualize_acc([fico_trainset], save_to=directory_for_saving_plot, scip_or_fico=scip_or_fico, plot_title=fico_trainset.name)
# checked

def get_sgm_list(df, shift):
    sgms = []
    for col in df.columns:
        wumr_gore = shifted_geometric_mean(df[col], shift)
        sgms.append(wumr_gore)
    return sgms

def compare_scalers_accuracy(version, deep, scip=True, fico=True, linear=True, forest=True):
    def plotter(value_list, title, fico=True, scip=False):

        columns = ['Accuracy', 'Mid Accuracy', 'Extreme Accuracy']
        values = [[value[0] for value in value_list], [value[1] for value in value_list],
                  [value[2] for value in value_list]]
        if scip:
            columns = ['Accuracy', 'Mid Accuracy']
            values = [[value[0] for value in value_list], [value[1] for value in value_list]]
        print(f'{version} {title} ACC')
        print(values)
        # print('---------------------------------------------------------------------------------------------------')
        # print(values[0], values[1], values[2])
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
        #ax.set_title(title)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(columns)
        ax.set_ylim([20, 100])
        ax.legend()

        plt.tight_layout()

        # Display the plot
        title = title.replace(" ", "_")
        if fico:
            fig.savefig(
                f'/Users/fritz/Downloads/ZIB/Master/Writing/Tex/LastMinute/FindingBestCombi/9{version}/Accuracy/BestCombi_depth{title}_accuracy.png')
        elif scip:
            print("SCCCIIIIP")
            fig.savefig(
                f'/Users/fritz/Downloads/ZIB/Master/Writing/Tex/LastMinute/FindingBestCombi/{version}/Accuracy/{title}_accuracy.png')
        plt.show()

    if scip:
        print("SCCCIIIIP")
        scip_complete = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Octesty/BestCombiSearch/SCIP/FindingBestCombi/depth{deep}/Accuracy/scip_acc_df.csv',
            index_col=0)
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
            plotter(ficos_lin[:4], f'SCIP {version} depth={deep} LinearRegressor Median Imputed', fico=False, scip=True)
            plotter(ficos_lin[4:], f'SCIP {version} depth={deep} LinearRegressor Mean Imputed', fico=False, scip=True)
        if forest:
            plotter(ficos_for[:4], f'SCIP{version} depth={deep} RandomForest Median Imputed', fico=False, scip=True)
            plotter(ficos_for[4:], f'SCIP{version} depth={deep} RandomForest Mean Imputed', fico=False, scip=True)

    if fico:
        fico_complete = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Octesty/BestCombiSearch/FICO{version}/FindingBestCombi/depth{deep}/Accuracy/fico_acc_df.csv',
            index_col=0)
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
            plotter(ficos_lin[:4], f'FICO 9.{version} depth={deep} LinearRegressor Median Imputed')
            plotter(ficos_lin[4:], f'FICO 9.{version} depth={deep} LinearRegressor Mean Imputed')
        if forest:
            plotter(ficos_for[:4], f'FICO 9.{version} depth={deep} RandomForest Median Imputed')
            plotter(ficos_for[4:], f'FICO 9.{version} depth={deep} RandomForest Mean Imputed')


def compare_scalers_runtime(version, deep, scip=True, fico=True, linear=True, forest=True):
    def relative_to_mixed(values):
        mixed = values[1]
        return [(value / mixed) for value in values]

    def relative_to_int(values):
        int = values[2]
        return [(value / int) for value in values]

    def plotter(value_list, title, fico=True, scip=False):
        columns = ['Mixed', 'Int',	'Predicted', 'VBS']
        values = [[value[1] for value in value_list],
                  [value[2] for value in value_list], [value[0] for value in value_list], [value[3] for value in value_list]]
        print(f'9{version}/BestCombi_depth{title}_sgm.')
        print(values[2])
        # print('---------------------------------------------------------------------------------------------------')
        # print(values[0], values[1], values[2])
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
        #ax.set_title(title)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(columns)
        ax.set_ylim([0, 1.2])
        ax.legend()

        plt.tight_layout()

        # Display the plot
        title = title.replace(" ", "_")
        if fico:
            fig.savefig(f'/Users/fritz/Downloads/ZIB/Master/Writing/Tex/LastMinute/FindingBestCombi/9{version}/SGM/BestCombi_depth{title}_sgm.png')
        if scip:
            fig.savefig(
                f'/Users/fritz/Downloads/ZIB/Master/Writing/Tex/LastMinute/FindingBestCombi/{version}/SGM/{title}_runtime.png')
        plt.show()

    if scip:
        scip_complete = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Octesty/BestCombiSearch/SCIP/FindingBestCombi/depth{deep}/SGM/scip_sgm_runtime.csv',
            index_col=0)
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
            plotter(ficos_lin[:4], f'SCIP {version} depth={deep} LinearRegressor Median Imputed', fico=False, scip=True)
            plotter(ficos_lin[4:], f'SCIP {version} depth={deep} LinearRegressor Mean Imputed', fico=False, scip=True)
        if forest:
            plotter(ficos_for[:4], f'SCIP {version} depth={deep} RandomForest Median Imputed', fico=False, scip=True)
            plotter(ficos_for[4:], f'SCIP {version} depth={deep} RandomForest Mean Imputed', fico=False, scip=True)

    if fico:
        fico_complete = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Octesty/BestCombiSearch/FICO{version}/FindingBestCombi/depth{deep}/SGM/fico_sgm_runtime.csv',
            index_col=0)
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
            plotter(ficos_lin[:4], f'FICO 9.{version} depth={deep} LinearRegressor Median Imputed', fico=True, scip=False)
            plotter(ficos_lin[4:], f'FICO 9.{version} depth={deep} LinearRegressor Mean Imputed', fico=True, scip=False)
        if forest:
            plotter(ficos_for[:4], f'FICO 9.{version} depth={deep} RandomForest Median Imputed', fico=True, scip=False)
            plotter(ficos_for[4:], f'FICO 9.{version} depth={deep} RandomForest Mean Imputed', fico=True, scip=False)


# checked
def get_scaler_runs(df:pd.DataFrame, scaler_name:str):
    scaler_runs = [scaler_run for scaler_run in df.index if scaler_name in scaler_run]
    scaler_run_df = df.loc[scaler_runs, :]
    return scaler_run_df
# checked
def compare_scalers(path_to_csv:str):
    acc_df = pd.read_csv(path_to_csv, index_col=0)
    acc_df = acc_df[['Accuracy', 'Mid Accuracy', 'Extreme Accuracy']]

    scaler_namen = ['None', 'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'PowerTransformer', 'QuantileTransformer']
    linear_rows = [lin_row for lin_row in acc_df.index if 'LinearRegression' in lin_row]
    forest_rows = [for_row for for_row in acc_df.index if 'RandomForest' in for_row]
    linear_df = acc_df.loc[linear_rows, :]
    forest_df = acc_df.loc[forest_rows, :]
    for scaler_name in scaler_namen:
        linear_scaler_run_df = get_scaler_runs(linear_df, scaler_name)
        forest_scaler_run_df = get_scaler_runs(forest_df, scaler_name)
        lin_scaler_sgm_acc = []
        forest_scaler_sgm_acc = []
        for acc in acc_df.columns:
            lin_scaler_sgm_acc.append(shifted_geometric_mean(linear_scaler_run_df[acc], shift=10))# linear_scaler_run_df[acc].mean()))
            forest_scaler_sgm_acc.append(shifted_geometric_mean(forest_scaler_run_df[acc], shift=10))# forest_scaler_run_df[acc].mean()))
# checking....
def compare_train_vs_test(scaled_or_unscaled:str, scaler_name:str, treffen, scip=True, fico=True):
    # Accuracy
    if scip:
        scip_test = pd.read_csv(
                    f'/Users/fritz/Downloads/ZIB/Master/October/Runs/{treffen}/SCIP/{scaler_name}/{scaled_or_unscaled}Label/Accuracy/scip_acc_df.csv',
                    index_col=0)
        scip_test.name = f'SCIP Default Accuracy Testset {scaler_name} {scaled_or_unscaled} Label'

        scip_train = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/October/Runs/{treffen}/SCIP/{scaler_name}/{scaled_or_unscaled}Label/Accuracy/scip_acc_trainset.csv',
            index_col=0)
        scip_train.name = f'SCIP Default Accuracy Trainingset {scaler_name} {scaled_or_unscaled} Label'

        train_test = [scip_test, scip_train]
        multiple_accuracy_plot(train_test, 'Comparison Accuracy Train vs Testset')

    if fico:
        fico_test = pd.read_csv(
                    f'/Users/fritz/Downloads/ZIB/Master/October/Runs/{treffen}/FICO/{scaler_name}/{scaled_or_unscaled}Label/Accuracy/fico_acc_df.csv',
                    index_col=0)
        fico_test.name = f'FICO Xpress Accuracy Testset {scaler_name} {scaled_or_unscaled} Label'

        fico_train = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/October/Runs/{treffen}/FICO/{scaler_name}/{scaled_or_unscaled}Label/Accuracy/fico_acc_trainset.csv',
            index_col=0)
        fico_train.name = f'FICO Xpress Accuracy Trainingset {scaler_name} {scaled_or_unscaled} Label'

        train_test = [fico_test, fico_train]
        multiple_accuracy_plot(train_test, 'Comparison Accuracy Train vs Testset')


    # RunTime
    if scip:
        scip_test = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/October/Runs/{treffen}/SCIP/{scaler_name}/{scaled_or_unscaled}Label/SGM/scip_sgm_runtime.csv',
            index_col=0)
        scip_test.name = f'FICO Xpress SGM Testset {scaler_name} {scaled_or_unscaled} Label'

        scip_train = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/October/Runs/{treffen}/SCIP/{scaler_name}/{scaled_or_unscaled}Label/SGM/scip_sgm_trainset.csv',
            index_col=0)
        scip_train.name = f'FICO Xpress SGM Trainset {scaler_name} {scaled_or_unscaled} Label'

        scips = [scip_test, scip_train]
        multiple_sgm_plot(scips, title=f'SCIP Comparison RunTime SGM Train vs Testset {scaler_name} {scaled_or_unscaled} Label')

    if fico:
        fico_test = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/October/Runs/{treffen}/FICO/{scaler_name}/{scaled_or_unscaled}Label/SGM/fico_sgm_runtime.csv',
            index_col=0)
        fico_test.name = f'FICO Xpress SGM Testset {scaler_name} {scaled_or_unscaled} Label'

        fico_train = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/October/Runs/{treffen}/FICO/{scaler_name}/{scaled_or_unscaled}Label/SGM/fico_sgm_trainset.csv',
            index_col=0)
        fico_train.name = f'FICO Xpress SGM Trainset {scaler_name} {scaled_or_unscaled} Label'

        ficos = [fico_test, fico_train]
        multiple_sgm_plot(ficos, title=f'FICO Xpress Comparison RunTime SGM Train vs Testset {scaler_name} {scaled_or_unscaled} Label')

def visualize_fico_on_scip(scaled_or_unscaled:str, scaler_name:str, treffen):
    # Accuracy
    scip_acc = pd.read_csv(
        f'/Users/fritz/Downloads/ZIB/Master/October/Runs/TreffenMasDoce/{scaler_name}/{scaled_or_unscaled}Label/Accuracy/scip_acc_df.csv',
        index_col=0)
    scip_acc.name = f'SCIP Default Accuracy {scaler_name} {scaled_or_unscaled} Label'

    fico_on_scip_acc = pd.read_csv(
        f'/Users/fritz/Downloads/ZIB/Master/October/Runs/{treffen}/FICOonSCIP/{scaler_name}/{scaled_or_unscaled}Label/Accuracy/fico_on_scip_acc_df.csv',
        index_col=0)
    fico_on_scip_acc.name = f'FICO XPRESS On SCIP Accuracy {scaler_name} {scaled_or_unscaled} Label'

    scip_vs_fico_on_scip_acc = [scip_acc, fico_on_scip_acc]
    multiple_accuracy_plot(scip_vs_fico_on_scip_acc, f'FICO Xpress vs SCIP on SCIP Accuracy {scaled_or_unscaled} Label')

    # RunTime
    scip_sgm = pd.read_csv(
        f'/Users/fritz/Downloads/ZIB/Master/October/Runs/TreffenMasDoce/{scaler_name}/{scaled_or_unscaled}Label/SGM/scip_sgm_runtime.csv',
        index_col=0)
    scip_sgm.name = f'SCIP SGM {scaler_name} {scaled_or_unscaled} Label'

    fico_on_scip_sgm = pd.read_csv(
        f'/Users/fritz/Downloads/ZIB/Master/October/Runs/{treffen}/FICOonSCIP/{scaler_name}/{scaled_or_unscaled}Label/SGM/fico_on_scip_sgm_runtime.csv',
        index_col=0)
    fico_on_scip_sgm.name = f'FICO Xpress on SCIP SGM {scaler_name} {scaled_or_unscaled} Label'

    fico_vs_scip_on_scip_sgm = [scip_sgm, fico_on_scip_sgm]

    multiple_sgm_plot(fico_vs_scip_on_scip_sgm,
                      title=f'FICO Xpress vs SCIP on SCIP Comparison RunTime SGM {scaler_name} {scaled_or_unscaled} Label')

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

def importance(treffen, scaler_name:str, scaled_or_unscaled:str, scip=False, fico=False):
    def feature_importance(data_frame):
        feature_names = data_frame.index.tolist()
        importance_dict = {}
        # linear_columns = [lin_col for lin_col in data_frame.columns if 'LinearRegression' in lin_col]
        # forest_columns = [for_col for for_col in data_frame.columns if 'RandomForest' in for_col]

        for feature in feature_names:
            minimum = min(data_frame.loc[feature,:])
            mean = data_frame.loc[feature,:].mean()
            importance_dict[feature] = shifted_geometric_mean(data_frame.loc[feature,:], shift=10)# abs(minimum)+abs(mean))

        sgm_importance_df = pd.DataFrame.from_dict(importance_dict, orient='index' )

        return sgm_importance_df

    def importance_bar_plot(data_frame, title):
        importance_df = feature_importance(data_frame)
        values = importance_df.iloc[:,0].tolist()
        # Create the plot
        bar_colors = (['turquoise', 'magenta'])
        plt.figure(figsize=(8, 5))
        plt.bar([i for i in range(len(values))], values, color=bar_colors)
        #plt.title(title)
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
        sorted_series = data_series.abs().sort_values(ascending=False)
        score_dict = {feature:0 for feature in sorted_series.index.tolist()}
        score = 0
        for feature in sorted_series.index.tolist():
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
        #plt.title(title)
        plt.ylabel('Importance Score')
        plt.xlabel('Feature')
        plt.xticks(rotation=270, fontsize=6)
        plt.show()
        plt.close()

    if fico:
        fico_impo = pd.read_csv(f'/Users/fritz/Downloads/ZIB/Master/October/Runs/{treffen}/Fico/{scaler_name}/{scaled_or_unscaled}Label/Importance/fico_importance_df.csv',
                                index_col=0)
        plot_importance_score(fico_impo, f'FICO Importance {scaler_name} Scaled Label')
        plot_importances_by_regressor(fico_impo, f'FICO Importance {scaler_name} Scaled Label')

    if scip:
        scip_impo = pd.read_csv(
            f'/Users/fritz/Downloads/ZIB/Master/October/Runs/{treffen}/Scip/{scaler_name}/{scaled_or_unscaled}Label/Importance/scip_importance_df.csv',
            index_col=0)

        plot_importance_score(scip_impo, f'SCIP Importance {scaler_name} Scaled Label')
        plot_importances_by_regressor(scip_impo, f'SCIP Importance {scaler_name} Scaled Label')

    # scip_impo_all_runs_scaled = pd.read_csv(
    #     '/Users/fritz/Downloads/ZIB/Master/October/Runs/TreffenMasDiez/ScaledLabel/Importance/scip_importance_df.csv')
    # scip_impo_all_runs_unscaled = pd.read_csv(
    #     '/Users/fritz/Downloads/ZIB/Master/October/Runs/TreffenMasDiez/UnscaledLabel/Importance/scip_importance_df.csv')

    # plot_importances_by_regressor(scip_impo_all_runs_unscaled, 'Feature Importance Unscaled Label')
    # plot_importances_by_regressor(scip_impo_all_runs_scaled, 'Feature Importance Scaled Label')

def sgm_by_combination(df, title, def_rule:str):
    scalers = ['None', 'QuantileTransformer']
    models = ['LinearRegression', 'RandomForest']
    imputers = ['median']
    linear_df_list = []
    forest_df_list = []
    # LinearRegression_median_None_288314836
    for scaler in scalers:
        for imputer in imputers:
            for model in models:
                combi = f"{model}_{imputer}_{scaler}"
                relevant_indices = [index for index in df.index if combi in index]
                combi_df = df.loc[relevant_indices, :]
                combi_df.name = f"{model} {scaler}"
                if model == 'LinearRegression':
                    linear_df_list.append(combi_df)
                elif model == 'RandomForest':
                    forest_df_list.append(combi_df)

    pass

def scip_train_test():
    scip_acc = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/October/Runs/Treffen_13_8/SCIP/WithOutlier/ScaledLabel/Accuracy/scip_acc_df.csv',
                           index_col=0)
    scip_acc_train = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/October/Runs/Treffen_13_8/SCIP/WithOutlier/ScaledLabel/Accuracy/scip_acc_trainset.csv',
                                 index_col=0)
    scip_sgm = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/October/Runs/Treffen_13_8/SCIP/WithOutlier/ScaledLabel/SGM/scip_sgm_runtime.csv', index_col=0)
    scip_sgm_train = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/October/Runs/Treffen_13_8/SCIP/WithOutlier/ScaledLabel/SGM/scip_sgm_trainset.csv', index_col=0)

    fico_acc = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/October/Runs/Treffen_6_8/FICO/NoOutlier/BestCombi/ScaledLabel/Accuracy/fico_acc_df.csv', index_col=0)
    fico_acc_train = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/October/Runs/Treffen_6_8/FICO/NoOutlier/BestCombi/ScaledLabel/Accuracy/fico_acc_trainset.csv', index_col=0)

    fico_sgm = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/October/Runs/Treffen_6_8/FICO/NoOutlier/BestCombi/ScaledLabel/SGM/fico_sgm_runtime.csv', index_col=0)
    fico_sgm_train = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/October/Runs/Treffen_6_8/FICO/NoOutlier/BestCombi/ScaledLabel/SGM/fico_sgm_trainset.csv', index_col=0)

    scip_acc.name = 'SCIP ACC Testset'
    scip_acc_train.name = 'SCIP ACC Trainset'
    scip_sgm.name = 'SCIP SGM Testset'
    scip_sgm_train.name = 'SCIP SGM Trainset'
    fico_acc.name = 'FICO ACC Testset'
    fico_acc_train.name = 'FICO ACC Trainset'
    fico_sgm.name = 'FICO SGM Testset'
    fico_sgm_train.name = 'FICO SGM Trainset'

def plot_sumtime(file_name, move_by, title):
    def get_sgm_column(pandas_series, shift):
        value = shifted_geometric_mean(pandas_series, shift)
        return value
    def plot_values(values, labels,  plot_title):
        # Determine bar colors based on conditions
        bar_colors = (['turquoise', 'magenta'])

        # Create the plot
        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, values, color=bar_colors)
        #plt.title(plot_title)
        plt.ylim(min(values)*0.9, max(values) * 1.05)  # Set y-axis limits for visibility
        plt.xticks(rotation=45, fontsize=6)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom')
        # Display the plot
        plt.show()
        plt.close()

    sum_time_df = pd.read_csv(file_name, index_col=0)
    values_sgm_lin = []
    values_total_lin = []
    values_sgm_for = []
    values_total_for = []
    lin_indices = [ind for ind in sum_time_df.index if 'Linear' in ind]
    for_indices = [ind for ind in sum_time_df.index if 'Random' in ind]
    lin_df = sum_time_df.loc[lin_indices]
    for_df = sum_time_df.loc[for_indices]
    for col in sum_time_df.columns[:4]:
        values_sgm_lin.append(np.round(get_sgm_column(lin_df[col], shift=10), 2))
        values_total_lin.append(np.round(lin_df[col].sum(), 2))
        values_sgm_for.append(np.round(get_sgm_column(for_df[col], shift=10), 2))
        values_total_for.append(np.round(for_df[col].sum(), 2))
    # int, mixed, linear, forest, vbs
    values_sgm = [values_sgm_lin[2], values_sgm_lin[1], values_sgm_lin[0], values_sgm_for[0], values_sgm_lin[3]]
    values_total = [values_total_lin[2], values_total_lin[1], values_total_lin[0], values_total_for[0], values_total_lin[3]]
    values_sgm_relative = [np.round(value/values_sgm[1], 2) for value in values_sgm]
    values_total_relative = [np.round(value / values_total[1], 2) for value in values_total]

    plot_values(values_sgm_relative, labels=['Int', 'Mixed', 'Linear', 'Forest', 'VBS'], plot_title=title+'SGM Runtime')
    plot_values(values_total_relative, labels=['Int', 'Mixed', 'Linear', 'Forest', 'VBS'], plot_title=title+'Total Runtime')

def sgm_base_set(dataframe, only_relevant, title, shift_name):
    def get_sgm_of_sgm(df,  shift):
        sgm_df = df.copy()
        col_names = df.columns.tolist()
        # Frage: SGM of relative SGMs oder von total SGMs?
        # Ich mach erstmal total sgms
        sgm_sgm_df = pd.DataFrame(columns=col_names, index=['SGM'])
        for col in col_names:
            sgm_sgm_df.loc[:, col] = shifted_geometric_mean(sgm_df[col], shift)
        return sgm_sgm_df

    def relative_to_mixed(value_df):
        values = value_df.iloc[0, :].tolist()
        mixed = value_df['Mixed'].iloc[0]
        values = [value/mixed for value in values]
        return values

    rel_df = dataframe.copy()
    if only_relevant:
        rel_indices = [ind for ind in rel_df.index if rel_df['Cmp Final solution time (cumulative)'].loc[ind] != 0]
        rel_df = rel_df.loc[rel_indices, :]
    if isinstance(shift_name, int):
        shifter = shift_name
    elif shift_name == 'mean':
        shifter = rel_df['Virtual Best'].mean()
    elif shift_name == 'median':
        shifter = rel_df['Virtual Best'].median()
    time_df = rel_df[['Final solution time (cumulative) Int', 'Final solution time (cumulative)', 'Virtual Best']].copy()
    time_df.columns = ['Int', 'Mixed', 'VBS']

    values = get_sgm_of_sgm(time_df, shift=10)
    values_relative = relative_to_mixed(values)

    labels = ['Int', 'Mixed', 'VBS']
    # Determine bar colors based on conditions
    bar_colors = (['purple', 'pink'])

    # Create the plot
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values_relative, color=bar_colors)
    #plt.title(title)
    plt.ylim(0.5, max(values_relative) * 1.05)  # Set y-axis limits for visibility
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



fico_5 = pd.read_csv("/Users/fritz/Downloads/ZIB/Master/October/Bases/FICO/Cleaned/9_5_ready_to_ml.csv")
fico_6 = pd.read_csv("/Users/fritz/Downloads/ZIB/Master/October/Bases/FICO/Cleaned/9_6_ready_to_ml.csv")

fico_5_path = "/Users/fritz/Downloads/ZIB/Master/October/Runs/Polishing/FICO5/depth5"
fico_5_path_10 = "/Users/fritz/Downloads/ZIB/Master/October/Runs/Polishing/FICO5/depth10"
fico_5_path_none = "/Users/fritz/Downloads/ZIB/Master/October/Runs/Polishing/FICO5/depthNone"

fico_5_top_forest = "/Users/fritz/Downloads/ZIB/Master/October/Runs/Polishing/FICO/Top2Forest965/depth5"
fico_5_top_forest_10 = "/Users/fritz/Downloads/ZIB/Master/October/Runs/Polishing/FICO/Top2Forest965/depth10"
fico_5_top_forest_none = "/Users/fritz/Downloads/ZIB/Master/October/Runs/Polishing/FICO/Top2Forest965/depthNone"
fico_5_top_lin = "/Users/fritz/Downloads/ZIB/Master/October/Runs/Polishing/FICO/Top4Linear965/depth5"
fico_5_top_lin_10 = "/Users/fritz/Downloads/ZIB/Master/October/Runs/Polishing/FICO/Top4Linear965/depth10"
fico_5_top_lin_none = "/Users/fritz/Downloads/ZIB/Master/October/Runs/Polishing/FICO/Top4Linear965/depthNone"

fico_6_top_forest = "/Users/fritz/Downloads/ZIB/Master/October/Runs/Polishing/FICO/Top2Forest966/depth5"
fico_6_top_forest_10 = "/Users/fritz/Downloads/ZIB/Master/October/Runs/Polishing/FICO/Top2Forest966/depth10"
fico_6_top_forest_none = "/Users/fritz/Downloads/ZIB/Master/October/Runs/Polishing/FICO/Top2Forest966/depthNone"
fico_6_top_lin = "/Users/fritz/Downloads/ZIB/Master/October/Runs/Polishing/FICO/Top4Linear966/depth5"
fico_6_top_lin_10 = "/Users/fritz/Downloads/ZIB/Master/October/Runs/Polishing/FICO/Top4Linear966/depth10"
fico_6_top_lin_none = "/Users/fritz/Downloads/ZIB/Master/October/Runs/Polishing/FICO/Top4Linear966/depthNone"



scip_top_forest = ""
scip_top_lin = ""

fico_6_path = "/Users/fritz/Downloads/ZIB/Master/October/Runs/Polishing/FICO6/depth5"
fico_6_path_10 = "/Users/fritz/Downloads/ZIB/Master/October/Runs/Polishing/FICO6/depth10"
fico_6_path_none = "/Users/fritz/Downloads/ZIB/Master/October/Runs/Polishing/FICO6/depthNone"

scip_path = '/Users/fritz/Downloads/ZIB/Master/October/Runs/Polishing/SCIP/depth5'
scip_path_10 = '/Users/fritz/Downloads/ZIB/Master/October/Runs/Polishing/SCIP/depth10'
scip_path_none = '/Users/fritz/Downloads/ZIB/Master/October/Runs/Polishing/SCIP/depthNone'


# /Users/fritz/Downloads/ZIB/Master/Octesty/BestCombiSearch/SCIP



# compare_scalers_accuracy(version="5", deep="None", scip=False, fico=True)#, forest=False)
# compare_scalers_runtime(version="5", deep="None", scip=False, fico=True)#, forest=False)
#
# compare_scalers_accuracy(version="6", deep="None", scip=False, fico=True)#, forest=False)
# compare_scalers_runtime(version="6", deep="None", scip=False, fico=True)#, forest=False)
# #
# compare_scalers_accuracy(version="10", deep="None", scip=True, fico=False)#, forest=False)
# compare_scalers_runtime(version="10", deep="None", scip=True, fico=False)#, forest=False)
#
# #
# train_vs_test_acuracy(data_frame_path="/Users/fritz/Downloads/ZIB/Master/Octesty/Train5Test6/Top3ForFICO5/ScaledLabel", fico_or_scip='fico', version="Train5Test6Forest", save_to=f"/Users/fritz/Downloads/ZIB/Master/Writing/Tex/LastMinute/Train5Test6")
# train_vs_test_sgm(data_frame_path="/Users/fritz/Downloads/ZIB/Master/Octesty/Train5Test6/Top3ForFICO5/ScaledLabel", fico_or_scip='fico', version="Train5Test6Forest", save_to=f"/Users/fritz/Downloads/ZIB/Master/Writing/Tex/LastMinute/Train5Test6")
#
# train_vs_test_acuracy(data_frame_path="/Users/fritz/Downloads/ZIB/Master/Octesty/Train6Test5/Top4ForFICO6/ScaledLabel", fico_or_scip='fico', version="Train6Test5Forest", save_to=f"/Users/fritz/Downloads/ZIB/Master/Writing/Tex/LastMinute/Train6Test5")
# train_vs_test_sgm(data_frame_path="/Users/fritz/Downloads/ZIB/Master/Octesty/Train6Test5/Top4ForFICO6/ScaledLabel", fico_or_scip='fico', version="Train6Test5Forest", save_to=f"/Users/fritz/Downloads/ZIB/Master/Writing/Tex/LastMinute/Train6Test5")
