import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split


import sys
import os

def setup_directory(new_directory):
    # Setup directory
    os.makedirs(os.path.join(f'{new_directory}'), exist_ok=True)

def create_directory(parent_name):
    base_path = f'{parent_name}'
    subdirs = ['Prediction', 'Accuracy', 'Importance', 'SGM']
    for subdir in subdirs:
        os.makedirs(os.path.join(base_path, subdir), exist_ok=True)
    return 0

def get_files(directory_path:str, index_col=False):
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

def shifted_geometric_mean(values, shift):
    values = np.array(values)
    if values.dtype == 'object':
        # Attempt to convert to float
        values = values.astype(float)
    # Check if shift is large enough
    if shift <= -values.min():
        raise ValueError(f"Shift too small. Minimum value is {values.min()}, so shift must be > {-values.min()}")
    # shift values by constant
    shifted_values = values + shift

    shifted_values_log = np.log(shifted_values)  # Step 1: Log of each element in shifted_values

    log_mean = np.mean(shifted_values_log)  # Step 2: Compute the mean of the log values
    geo_mean = np.exp(log_mean) - shift
    # geo_mean = np.round(geo_mean, 6)
    return geo_mean

# WHAT RULES DID THE MODELS CHOOSE
def show_shares(path_to_prediction_directory, scip_default_original_data=None, fico_original_data=None, dataset_name="",
           subdirectory=""):

    def get_share_mixed_and_int(data_frame):
        mixed = (data_frame > 0).sum().sum()
        pref_int = (data_frame < 0).sum().sum()
        return [mixed, pref_int]

    def real_shares(dataset):
        hundred_seeds = [2207168494, 288314836, 1280346069, 1968903417, 1417846724, 2942245439, 2177268096, 571870743,
                         1396620602, 3691808733, 4033267948, 3898118442, 24464804, 882010483, 2324915710, 316013333,
                         3516440788, 535561664, 1398432260, 572356937, 398674085, 4189070509, 429011752, 2112194978,
                         3234121722, 2237947797, 738323230, 3626048517, 733189883, 4126737387, 2399898734, 1856620775,
                         829894663, 3495225726, 1844165574, 1282240360, 2872252636, 1134263538, 1174739769, 2128738069,
                         1900004914, 3146722243, 3308693507, 4218641677, 563163990, 568995048, 263097927, 1693665289,
                         1341861657, 1387819803, 157390416, 2921975935, 1640670982, 4226248960, 698121968, 1750369715,
                         3843330071, 2093310729, 1822225600, 958203997, 2478344316, 3925818254, 2912980295, 1684864875,
                         362704412, 859117595, 2625349598, 3108382227, 1891799436, 1512739996, 1533327828, 1210988828,
                         3504138071, 1665201999, 1023133507, 4024648401, 1024137296, 3118826909, 4052173232, 3143265894,
                         1584118652, 1023587314, 666405231, 2782652704, 744281271, 3094311947, 3882962880, 325283101,
                         923999093, 4013370079, 2033245880, 289901203, 3049281880, 1507732364, 698625891, 1203175353,
                         1784663289, 2270465462, 537517556, 2411126429]
        test_set = {}
        train_set = {}
        for seed in hundred_seeds:
            X_train, X_test, y_train, y_test = train_test_split(dataset,
                                                                dataset['Cmp Final solution time (cumulative)'],
                                                                test_size=0.2,
                                                                random_state=seed)
            test_set[seed] = y_test
            train_set[seed] = y_train
        return pd.DataFrame.from_dict(test_set, orient='columns'), pd.DataFrame.from_dict(train_set, orient='columns')

    def histogram_shares(data_frame, title_add_on, dataset_name=dataset_name, test_or_train="", origis=False):
        if origis:
            values = get_share_mixed_and_int(data_frame['Cmp Final solution time (cumulative)'])
        else:
            values = get_share_mixed_and_int(data_frame)

        total_relevant_predictions = sum(values)
        values = [(value/total_relevant_predictions)*100 for value in values]
        bar_colors = (['purple', 'darkgreen'])
        title = f'Predicted share of Mixed and Int on {dataset_name} {test_or_train} by {title_add_on}'
        # Create the plot
        plt.figure(figsize=(8, 5))
        bars = plt.bar(['Mixed', 'Prefer Int'], values, color=bar_colors)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom')
        plt.title(title)
        plt.ylim(0, 105)  # Set y-axis limits for visibility
        plt.xticks(rotation=45, fontsize=6)
        # Display the plot
        plt.show()
        plt.close()

    dataframes = get_files(path_to_prediction_directory, index_col=True)

    for dataframe in dataframes.keys():
        df = dataframes[dataframe]
        lin_cols = [col_name for col_name in df.columns if "Linear" in col_name]
        forest_df = [col_name for col_name in df.columns if "Forest" in col_name]
        if 'Trainset' in dataframe:
            tester_or_trainer = 'Trainset'
        else:
            tester_or_trainer = 'Testset'
        histogram_shares(df[lin_cols], title_add_on="Linear", test_or_train=tester_or_trainer, origis=False)
        histogram_shares(df[forest_df], title_add_on="RandomForest", test_or_train=tester_or_trainer, origis=False)
    if scip_default_original_data is not None:
        real_test, real_train = real_shares(scip_default_original_data)
    elif fico_original_data is not None:
        real_test, real_train = real_shares(fico_original_data)
    else:
        sys.exit(1)
    histogram_shares(real_test, "Actual", test_or_train="Testset")
    histogram_shares(real_train, "Actual", test_or_train="Trainset")

def accuracy_visualize(path_to_accuracy_directory, scaled_label=True, title_add_on='', plot=True):
    def get_sgm_series(pandas_series, shift):
        return shifted_geometric_mean(pandas_series, shift)

    def get_sgm_acc(data_frame):
        data_frame['Accuracy'] = pd.to_numeric(data_frame['Accuracy'], errors='coerce')
        data_frame['Extreme Accuracy'] = pd.to_numeric(data_frame['Extreme Accuracy'], errors='coerce')

        sgm_accuracy = get_sgm_series(data_frame['Accuracy'], data_frame['Accuracy'].mean()+0.1)
        sgm_extreme_accuracy = get_sgm_series(data_frame['Extreme Accuracy'].dropna(), data_frame['Extreme Accuracy'].dropna().mean()+0.1)
        return sgm_accuracy, sgm_extreme_accuracy


    def visualize_acc(data_frame, filter_by:str, title: str = 'Accuracy', plot=True):
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
            return None
        else:
            if len(linear_df)>0:
                lin_acc, lin_ex_acc = get_sgm_acc(linear_df)
            if len(forest_df)>0:
                for_acc, for_ex_acc = get_sgm_acc(forest_df)

        values = [lin_acc, lin_ex_acc, for_acc, for_ex_acc]

        if plot:
            # Create the plot
            bar_colors = (['turquoise', 'turquoise', 'magenta', 'magenta'])
            plt.figure(figsize=(8, 5))
            plt.bar(['LinAcc', 'LinExAcc', 'ForAcc', 'ForExAcc'], values, color=bar_colors)
            plt.title(title)
            plt.ylim(0, 105)  # Set y-axis limits for visibility
            plt.xticks(rotation=45, fontsize=6)
            # Display the plot
            plt.show()
            plt.close()

    def call_acc_visualization(path_to_acc_dir, title_add_on='', plot=True):
        dataframes = get_files(path_to_acc_dir, index_col=True)

        for dataframe in dataframes.keys():
            df = dataframes[dataframe]
            visualize_acc(df, filter_by='', title=dataframe+title_add_on, plot=plot)
    if plot:
        call_acc_visualization(path_to_acc_dir=path_to_accuracy_directory, title_add_on=title_add_on)
    else:
        call_acc_visualization(path_to_acc_dir=path_to_accuracy_directory, plot=False)

def importance(path_to_importance_dir, saving_directory):

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
        df = pd.DataFrame({
            'Feature': list(linear_scores_dict.keys()),
            'Linear': list(linear_scores_dict.values()),
            'Forest': list(forest_scores_dict.values())
        })
        # Add the 'Combined' column
        df['Combined'] = df['Linear'] + df['Forest']

        # Sort by 'Combined' in ascending order (smallest on top)
        df = df.sort_values(by='Combined', ascending=True)

        return df, pd.DataFrame({'Linear': linear_scores_dict, 'Forest': forest_scores_dict})

    def plot_importance(data_frame, save_dir):
        all_impo_df, importance_df = get_importance_score_df(data_frame)
        all_impo_df.to_csv(save_dir, index=False)

        importance_df.plot(kind='bar', figsize=(10, 6), color=['turquoise', 'magenta'])
        #plt.title(title)
        plt.ylabel('Importance Score')
        plt.xlabel('Feature')
        plt.xticks(rotation=270, fontsize=6)
        plt.show()
        plt.close()

    impo_dfs = get_files(path_to_importance_dir, index_col=True)

    for impo_df in impo_dfs:
        plot_importance(data_frame=impo_dfs[impo_df], save_dir=saving_directory)

def feature_reduction_graph(feature_ranking:str, data_set:str, saving_directory,
                            fico_or_scip, lin_accuracy=None, lin_sgm=None, for_accuracy=None,
                            for_sgm=None, title_add_on="", axvline_position=14):
    setup_directory(saving_directory)
    if (lin_accuracy is None) or (lin_sgm is None) or (for_accuracy is None) or (for_sgm is None):
        print(f'None values inputed.')
        sys.exit(1)
    number_of_features = max(len(lin_accuracy), len(lin_sgm), len(for_accuracy), len(for_sgm))
    x_labels = [str(i) for i in range(number_of_features, 0, -1)]
    if feature_ranking == "linear":
        lin_vbs = (np.array(lin_sgm["VBS"]) * 100).tolist()
        lin_sgm = (np.array(lin_sgm["SGM relative to Default"]) * 100).tolist()
        # lin_sgm = [value[0] for value in lin_sgm]
    if feature_ranking == "forest":
        for_vbs = (np.array(for_sgm["VBS"]) * 100).tolist()
        for_sgm = (np.array(for_sgm["SGM relative to Default"]) * 100).tolist()


    if data_set == 'scip':
        if isinstance(lin_accuracy, list):
            pass
        else:
            lin_accuracy.drop('Extreme Accuracy', axis=1, inplace=True)
        if isinstance(for_accuracy, list):
            pass
        else:
            for_accuracy.drop('Extreme Accuracy', axis=1, inplace=True)


    if feature_ranking.lower() == 'combined':
        colors = ['gold', 'orange', 'darkgreen', 'green', 'turquoise', 'seagreen']
        plt.figure(figsize=(8, 6))
        plt.plot(lin_accuracy[['Mid Accuracy']], color=colors[0])
        plt.plot(lin_accuracy[['Extreme Accuracy']], color=colors[1])
        plt.plot(for_accuracy[['Mid Accuracy']], color=colors[2])
        plt.plot(for_accuracy[['Extreme Accuracy']], color=colors[3])
        plt.plot(lin_sgm, color=colors[4])
        plt.plot(for_sgm, color=colors[5])
        plt.plot(for_sgm)
        plt.ylim(35, 115)
        plt.legend(['MidLabel Accuracy Linear', 'LargeLabel Accuracy Linear', 'MidLabel Accuracy Random Forest',
                    'LargeLabel Accuracy Random Forest', 'SGM Linear', 'SGM Random Forest'])
        plt.xticks(ticks=range(len(lin_accuracy)), labels=x_labels)
        plt.show()
    elif feature_ranking.lower() == 'linear':
        if data_set == 'scip':
            plt.figure(figsize=(8, 6))
            plt.plot(lin_accuracy.iloc[:, 0], color='green')
            plt.plot(lin_accuracy.iloc[:, 1], color='purple')
            plt.plot(lin_sgm, color='red')
            plt.plot(lin_vbs, color='lightcoral')
            #plt.title(label=f'{title_add_on}')
            plt.xlabel('Number of Features')
            plt.legend(['Accuracy', 'MidLabel Accuracy', 'Predicted Run Time', 'Virtual Best Run Time'])
            #plt.axvline(x=axvline_position, color='red', linestyle='--', label='Threshold')
        else:
            plt.figure(figsize=(8, 6))
            plt.plot(lin_accuracy.iloc[:, 0], color='green')
            plt.plot(lin_accuracy.iloc[:, 1], color='purple')
            plt.plot(lin_accuracy.iloc[:, 2], color='blue')
            plt.plot(lin_sgm, color='red')
            plt.plot(lin_vbs, color='lightcoral')
            #plt.title(label=f'{title_add_on}')
            plt.xlabel('Number of Features')
            plt.legend(['Accuracy', 'MidLabel Accuracy', 'LargeLabel Accuracy', 'Predicted Run Time', 'Virtual Best Run Time'])
            #plt.axvline(x=axvline_position, color='red', linestyle='--', label='Threshold')
        plt.xticks(ticks=range(len(lin_accuracy)), labels=x_labels)
        plt.ylim(45, 115)
        safe = title_add_on.replace(' ', '_')
        filename = f"{fico_or_scip}_{safe}_feat_reduction_linear.png"
        plt.savefig(f"{saving_directory}/{filename}")
        plt.show()

    elif feature_ranking.lower() == 'forest':
        if data_set == 'scip':
            plt.figure(figsize=(8, 6))
            plt.plot(for_accuracy.iloc[:, 0], color='green')
            plt.plot(for_accuracy.iloc[:, 1], color='purple')
            plt.plot(for_sgm, color='red')
            plt.plot(for_vbs, color='lightcoral')
            #plt.title(label=f'{title_add_on}')
            plt.xlabel('Number of Features')
            plt.legend(['Accuracy', 'MidLabel Accuracy', 'Predicted Run Time', 'Virtual Best Run Time'])
            #plt.axvline(x=axvline_position, color='red', linestyle='--', label='Threshold')
        else:
            plt.figure(figsize=(8, 6))
            plt.plot(for_accuracy.iloc[:, 0], color='green')
            plt.plot(for_accuracy.iloc[:, 1], color='purple')
            plt.plot(for_accuracy.iloc[:, 2], color='blue')
            plt.plot(for_sgm, color='red')
            plt.plot(for_vbs, color='lightcoral')
            #plt.title(label=f'{title_add_on}')
            plt.xlabel('Number of Features')
            plt.legend(['Accuracy', 'MidLabel Accuracy', 'LargeLabel Accuracy', 'Predicted Run Time', 'Virtual Best Run Time'])
            #plt.axvline(x=axvline_position, color='red', linestyle='--', label='Threshold')
        plt.xticks(ticks=range(len(for_accuracy)), labels=x_labels)
        safe = title_add_on.replace(' ', '_')
        filename = f"{saving_directory}/{fico_or_scip}_{safe}_feat_reduction_forest.png"
        plt.ylim(45, 115)
        plt.savefig(filename)
        plt.show()
        plt.close()
    else:
        sys.exit(1)

def plot_feature_reduction(directory:str, fico_or_scip:str, save_to, feature_ranking='combined',
                           title_add_on:str='', threshold=14):
    dfs = get_files(directory)

    accuracy_keys = [key for key in dfs.keys() if 'acc' in key.lower()]
    sgm_keys = [key for key in dfs.keys() if 'sgm' in key.lower()]

    if feature_ranking == 'combined':
        acc_lin_key = [key for key in accuracy_keys if 'combined linear' in key.lower()][0]
        acc_for_key = [key for key in accuracy_keys if 'combined forest' in key.lower()][0]

        sgm_lin_key = [key for key in sgm_keys if 'combined linear' in key.lower()][0]
        sgm_for_key = [key for key in sgm_keys if 'combined forest' in key.lower()][0]

        feature_reduction_graph(feature_ranking, lin_accuracy=dfs[acc_lin_key], lin_sgm=dfs[sgm_lin_key],
                                for_accuracy=dfs[acc_for_key], for_sgm=dfs[sgm_for_key], data_set=fico_or_scip,
                                title_add_on=title_add_on, axvline_position=threshold, saving_directory=save_to,
                                fico_or_scip=fico_or_scip)

    elif feature_ranking == 'linear':
        acc_lin_key = [key for key in accuracy_keys if 'linear linear' in key.lower()][0]
        sgm_lin_key = [key for key in sgm_keys if 'linear linear' in key.lower()][0]
        feature_reduction_graph(feature_ranking, lin_accuracy=dfs[acc_lin_key], lin_sgm=dfs[sgm_lin_key],
                                for_accuracy=[], for_sgm=[], data_set=fico_or_scip, title_add_on=title_add_on,
                                axvline_position=threshold, saving_directory=save_to,
                                fico_or_scip=fico_or_scip)

    elif feature_ranking == 'forest':
        acc_for_key = [key for key in accuracy_keys if 'forest forest' in key.lower()][0]
        sgm_for_key = [key for key in sgm_keys if 'forest forest' in key.lower()][0]
        print(fico_or_scip)
        feature_reduction_graph(feature_ranking, lin_accuracy=[], lin_sgm=[],
                                for_accuracy=dfs[acc_for_key], for_sgm=dfs[sgm_for_key], data_set=fico_or_scip,
                                title_add_on=title_add_on, axvline_position=threshold, saving_directory=save_to,
                                fico_or_scip=fico_or_scip)