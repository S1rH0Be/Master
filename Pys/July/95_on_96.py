import time
import pandas as pd
import numpy as np
import os
import sys

from stats_per_combi_july import ranking_feature_importance

from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer

import logging

def setup_directory(new_directory):
    # Setup directory
    os.makedirs(os.path.join(f'{new_directory}'), exist_ok=True)

def create_directory(parent_name):

    base_path = f'{parent_name}'
    subdirs = ['Prediction', 'Accuracy', 'Importance', 'SGM']
    for subdir in subdirs:
        os.makedirs(os.path.join(base_path, subdir), exist_ok=True)
    return 0

def shifted_geometric_mean(values, shift):
    if len(values)==0:
        print("Nothing Extreme Here")
        return 0
    values = np.array(values)

    if values.dtype == 'object':
        # Attempt to convert to float
        values = values.astype(float)

    # Shift the values by the constant
    # Check if shift is large enough
    if shift <= -values.min():
        print(f"Shift {shift} too small. Minimum value is {values.min()}, so shift must be > {-values.min()}")
        raise ValueError(f"Shift too small. Minimum value is {values.min()}, so shift must be > {-values.min()}")

    shifted_values = values + shift

    shifted_values_log = np.log(shifted_values)  # Step 1: Log of each element in shifted_values

    log_mean = np.mean(shifted_values_log)  # Step 2: Compute the mean of the log values
    geo_mean = np.exp(log_mean) - shift
    # geo_mean = np.round(geo_mean, 6)
    return geo_mean

def print_accuracy(acc_df):
    def get_sgm_series(pandas_series, shift):
        return shifted_geometric_mean(pandas_series, shift)
    def get_sgm_acc(data_frame):
        data_frame['Accuracy'] = pd.to_numeric(data_frame['Accuracy'], errors='coerce')
        data_frame['Extreme Accuracy'] = pd.to_numeric(data_frame['Extreme Accuracy'], errors='coerce')

        sgm_accuracy = get_sgm_series(data_frame['Accuracy'], data_frame['Accuracy'].mean()+0.1)
        sgm_mid_accuracy = get_sgm_series(data_frame['Mid Accuracy'], data_frame['Mid Accuracy'].mean()+0.1)
        if len(data_frame['Extreme Accuracy'])>0:
            sgm_extreme_accuracy = get_sgm_series(data_frame['Extreme Accuracy'].dropna(), data_frame['Extreme Accuracy'].dropna().mean()+0.1)
        else:
            sgm_extreme_accuracy = 0
        return [sgm_accuracy, sgm_mid_accuracy, sgm_extreme_accuracy]

    def visualize_acc(data_frame, title: str = 'Accuracy'):
        """Gets sgm of accuracy of a run for the linear and the forest model."""
        acc_df = data_frame.copy()
        # if no corresponding acc is found just plot it as 0
        lin_acc, lin_ex_acc, for_acc, for_ex_acc = 0, 0, 0, 0

        linear_rows = [lin_rows for lin_rows in acc_df.index if 'LinearRegression' in lin_rows]
        linear_df = acc_df.loc[linear_rows, :]

        forest_rows = [for_row for for_row in acc_df.index if 'RandomForest' in for_row]
        forest_df = acc_df.loc[forest_rows,:]

        if len(linear_df) == len(forest_df) == 0:
            print(f'{title}: No data found')
            return None
        else:
            if len(linear_df)>0:
                lin_acc = get_sgm_acc(linear_df)
                return np.round(lin_acc, 2)
            if len(forest_df)>0:
                for_acc = get_sgm_acc(forest_df)
                return np.round(for_acc, 2)

    return visualize_acc(acc_df, title='Accuracy')

def print_sgm(sgm_df, data_set, number_features):

    def get_sgm_of_sgm(data_frame, shift):
        col_names = data_frame.columns.tolist()
        # Frage: SGM of relative SGMs oder von total SGMs?
        # Ich mach erstmal total sgms
        sgm_sgm_df = pd.DataFrame(columns=col_names, index=['Value'])
        for col in col_names:
            print(f'GetSGMofSGM shift: {shift}')
            sgm_sgm_df.loc[:, col] = shifted_geometric_mean(data_frame[col], shift)
        return sgm_sgm_df

    def relative_to_default(value_dict:dict, dataset:str):
        if dataset.lower() == 'fico':
            default_rule = 'Mixed'
        elif dataset[:4].lower() == 'scip':
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
        # print("Means", means)
        means = [mixed.mean(), pref_int.mean(), prediction.mean(), vbs.mean()]
        # print("Means", means)
        sums = [mixed.sum(), pref_int.sum(), prediction.sum(), vbs.sum()]
        # print("Sums", sums)


        mixed_values = shifted_geometric_mean(mixed, mean_mean)
        pref_int_values = shifted_geometric_mean(pref_int, mean_mean)
        predicted_values = shifted_geometric_mean(prediction, mean_mean)
        vbs_values = shifted_geometric_mean(vbs, mean_mean)
        value_dictionary = {'Int': pref_int_values, 'Mixed': mixed_values, 'Predicted': predicted_values,
                            'VBS': vbs_values}

        values_relative = relative_to_default(value_dictionary, data_set)
        return values_relative

    def sgm_plot(dataframe, title:str, data_set:str, plot=True):
        values = get_values_for_plot(dataframe, data_set)
        if plot:
            labels = ['PrefInt', 'Mixed', 'Predicted', 'VBS']

            bar_colors = ['turquoise', 'magenta']

            # Create the plot
            plt.figure(figsize=(8, 5))
            plt.bar(labels, values, color=bar_colors)
            plt.title(title)
            if data_set.lower() == 'fico':
                plt.ylim(0.5, 1.35)  # Set y-axis limits for visibility
            else:
                plt.ylim(0.8, 1.06)  # Set y-axis limits for visibility
            plt.xticks(rotation=45, fontsize=6)
            # Display the plot
            plt.show()
            plt.close()
        return np.round(values[2:], 2)

    def call_sgm_visualization(df, number_features, data_set, plot=True):
        return sgm_plot(df, f'SGM {number_features}', data_set, plot)

    return call_sgm_visualization(sgm_df, number_features, data_set, plot=False)

def get_features_label(data_frame, feature_df, chosen_features):
    features = feature_df[chosen_features]
    label = data_frame['Cmp Final solution time (cumulative)']
    return features, label

def label_scaling(label):
    print("LABEL SCALING")
    y_pos = label[label >= 0]
    y_neg = label[label < 0]
    # log1p calculates log(1+x) numerically stable
    y_pos_log = np.log1p(y_pos)/np.log(10)
    y_neg_log = (np.log1p(abs(y_neg))/np.log(10)) * -1
    y_log = pd.concat([y_pos_log, y_neg_log]).sort_index()
    return y_log

def get_accuracy(prediction, actual, mid_threshold, extreme_threshold):

    # Filter for nonzero labels
    nonzero_indices = actual != 0
    y_test_nonzero = actual[nonzero_indices]

    def overall_accuracy():
        y_pred_nonzero = prediction[nonzero_indices]

        # Calculate percentage of correctly predicted signs
        correct_signs = np.sum(np.sign(y_test_nonzero) == np.sign(y_pred_nonzero))
        percent_correct_signs = correct_signs / len(y_test_nonzero) * 100 if len(y_test_nonzero) > 0 else np.nan
        return percent_correct_signs

    def threshold_accuracy(relevant_threshold):
        # Filter for mid labels
        relevant_indices = abs(actual) >= relevant_threshold

        y_test_relevant = actual[relevant_indices]
        number_relevant_instances = (len(y_test_relevant), len(y_test_nonzero))
        y_pred_relevant = prediction[relevant_indices]

        # Calculate percentage of correctly predicted signs
        number_correct_predictions = np.sum(np.sign(y_test_relevant) == np.sign(y_pred_relevant))
        accuracy_threshold = number_correct_predictions / len(y_test_relevant) * 100 if len(
            y_test_relevant) > 0 else np.nan
        return accuracy_threshold, number_relevant_instances

    overall_acc = overall_accuracy()
    mid_acc, number_mid_instances = threshold_accuracy(mid_threshold)
    extreme_acc, number_extreme_instances = threshold_accuracy(extreme_threshold)

    return overall_acc, mid_acc, number_mid_instances, extreme_acc, number_extreme_instances
# TODO Delete get_sgm_comparison if it runs without it
# def get_sgm_comparison(y_pred, y_test):
#     pred_df = pd.DataFrame({'Prediction': y_pred, 'Actual': y_test},
#                            index=y_pred.index)
#     pred_df['Right or Wrong'] = (np.sign(pred_df['Prediction']) == np.sign(pred_df['Actual'])).astype(int)
#     # add column containing the absolute difference in prediction and actual
#     pred_df['Abs Time Diff'] = abs(pred_df['Prediction'] - pred_df['Actual'])
#     return pred_df

def get_predicted_run_time_sgm(y_pred, data, shift):
    predicted_time = pd.Series(index=y_pred.index, name='Predicted Run Time')
    indices = y_pred.index
    for i in indices:
        if y_pred.loc[i] > 0:
            predicted_time.loc[i] = data.loc[i, 'Final solution time (cumulative)']
        else:
            predicted_time.loc[i] = data.loc[i, 'Final solution time (cumulative) Int']
    try:
        sgm_predicted = shifted_geometric_mean(predicted_time, shift)
        sgm_mixed = shifted_geometric_mean(data['Final solution time (cumulative)'].loc[indices], shift)
        sgm_int = shifted_geometric_mean(data['Final solution time (cumulative) Int'].loc[indices], shift)
        sgm_vbs = shifted_geometric_mean(data['Virtual Best'].loc[indices], shift)

    except ValueError as e:
        logging.error(f"SGM failed due to shift: {e}")
        return None, None, None, None  # or raise again if you want the pipeline to crash
    return sgm_predicted, sgm_mixed, sgm_int, sgm_vbs

def get_predicted_sum_time(y_pred, data):
    indices = y_pred.index
    predicted_time_sum = 0
    mixed_time_sum =  data['Final solution time (cumulative)'].loc[indices].sum()
    int_time_sum = data['Final solution time (cumulative) Int'].loc[indices].sum()
    vbs_sum = data['Virtual Best'].loc[indices].sum()
    for i in indices:
        if y_pred.loc[i] > 0:
            predicted_time_sum += np.round(data.loc[i, 'Final solution time (cumulative)'], 2)
        else:
            predicted_time_sum += np.round(data.loc[i, 'Final solution time (cumulative) Int'], 2)
    return np.round(predicted_time_sum, 2), mixed_time_sum, int_time_sum, vbs_sum

def get_importance_col(importances, feature_names, model_name, imputation, scaler):
    importance_df = pd.DataFrame({f'{model_name}_{imputation}_{scaler}': importances}, index=feature_names)
    return importance_df

def get_prediction_df(dictionary):
    max_len = max(len(lst) for lst in dictionary.values())
    # Pad each list with zeros
    for key in dictionary:
        dictionary[key] += [0] * (max_len - len(dictionary[key]))
    return pd.DataFrame.from_dict(dictionary, orient='columns')

def kick_outlier(feat_df, label_series, threshold : int):
    indices_to_keep = label_series[label_series.abs() <= threshold].index
    feats_to_keep = feat_df.loc[indices_to_keep, :]
    labels_to_keep = label_series.loc[indices_to_keep]
    return feats_to_keep, labels_to_keep

def feature_reduction(train_data_path, test_data_path, train_feature_path, test_feature_path, fico_or_scip, model: str,
                      treffen, scaling, wie_tief, anzahl_estimators, feats_per_tree, remove_outlier=True):
    create_directory(f'{treffen}/FeatureReduction')
    accuracy_lin = []
    sgm_lin = []
    accuracy_for = []
    sgm_for = []

    def acc_sgm_to_csv(accuracy_drops_lin, sgm_drops_lin, data_set, accuracy_drops_for, sgm_drops_for, feature_ranking, directory):
        path = f'{directory}/FeatureReduction/{data_set}'
        create_directory(path)

        if len(accuracy_drops_lin)>0:
            acc_reduct_lin_df = pd.DataFrame(data=accuracy_drops_lin, columns=['Accuracy', 'Mid Accuracy', 'Extreme Accuracy'])
            acc_reduct_lin_df.to_csv(
                f'{path}/acc_{data_set}_{feature_ranking}_linear.csv',
                index=False)

        if len(sgm_drops_lin)>0:
            sgm_reduct_lin_df = pd.DataFrame(data=sgm_drops_lin, columns=['SGM relative to Default', 'VBS'])
            sgm_reduct_lin_df.to_csv(
                f'{path}/sgm_{data_set}_{feature_ranking}_linear.csv',
                index=False)

        if len(accuracy_drops_for)>0:
            acc_reduct_for_df = pd.DataFrame(data=accuracy_drops_for, columns=['Accuracy', 'Mid Accuracy', 'Extreme Accuracy'])
            acc_reduct_for_df.to_csv(
                f'{path}/acc_{data_set}_{feature_ranking}_forest.csv',
                index=False)

        if len(sgm_drops_for)>0:
            sgm_reduct_for_df = pd.DataFrame(data=sgm_drops_for, columns=['SGM relative to Default', 'VBS'])
            sgm_reduct_for_df.to_csv(
                f'{path}/sgm_{data_set}_{feature_ranking}_forest.csv',
                index=False)

    def forest_reduction(data_path_train, data_path_test, feature_path_train, feature_path_test,
                            scip_or_fico, impo_df, number_features, treffmas, scaling_for_features,
                         tiefe, estimatoren, anzahl_feat_per_tree, outlier_removal=True):
        feature_list = impo_df['Feature'].tolist()
        for reduce_by in range(len(feature_list)):
            treff = f'{treffmas}/FeatureReduction/forest/{reduce_by}'
            feature_set = feature_list[:(number_features - reduce_by)]
            print("#Features in ForesReduction:", len(feature_set))
            treffmas = treffmas.replace('/FICO', '')
            print('SCALING:', scaling)
            acc, sgm = main(path_to_trainset_data=data_path_train, path_to_testset_data=data_path_test,
                            path_to_trainset_feat=feature_path_train, path_to_testset_feat=feature_path_test,
                            scip_or_fico=scip_or_fico, treffplusx=treff, feature_scaling=scaling_for_features,
                            label_scalen=True, feature_subset=feature_set,
                            models={'RandomForest': RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)},
                            kick_outlier=outlier_removal, max_depth=tiefe, estimatoren=estimatoren,
                            max_feature_number=anzahl_feat_per_tree)

            accuracy_for.append(acc)
            sgm_for.append(sgm)

    def linear_reduction(data_path_train, data_path_test, feature_path_train, feature_path_test,
                            scip_or_fico, impo_df, number_features, treffmas, scaling_for_features,
                         tiefe, estimatoren, anzahl_feat_per_tree, outlier_removal=True):

        feature_list = impo_df['Feature'].tolist()
        for reduce_by in range(len(feature_list)):
            add_on = scip_or_fico

            treff = f'{treffmas}/FeatureReduction/{add_on}/linear/{reduce_by}'
            if reduce_by>0:
                print(feature_list[(number_features - reduce_by)])
            feature_set = feature_list[:(number_features - reduce_by)]

            acc, sgm = main(path_to_trainset_data=data_path_train, path_to_testset_data=data_path_test,
                            path_to_trainset_feat=feature_path_train, path_to_testset_feat=feature_path_test,
                            scip_or_fico=scip_or_fico, treffplusx=treff, feature_scaling=scaling_for_features,
                            label_scalen=True, feature_subset=feature_set, models={'LinearRegression': LinearRegression()},
                            kick_outlier=outlier_removal, max_depth=tiefe, estimatoren=estimatoren,
                            max_feature_number=anzahl_feat_per_tree)

            accuracy_lin.append(acc)
            sgm_lin.append(sgm)

    if fico_or_scip.lower() == 'fico':
        # TODO: change path to be interactive
        import_df = pd.read_csv(
            f'{treffen}/NoOutlier/Logged/ScaledLabel/Importance/fico_importance_ranking.csv')
        num_features = len(import_df)

    elif fico_or_scip.lower() == 'scip':
        import_df = pd.read_csv(
            f'{treffen}/NoOutlier/Logged/ScaledLabel/Importance/scip_default_importance_ranking.csv')
        num_features = len(import_df)

    else:
        print(f'Data set {fico_or_scip} not recognized. use scip or fico')
        sys.exit(1)

    if model.lower() == 'linear':
        import_df = import_df.sort_values(by=['Linear Score'], ascending=True)
        linear_reduction(data_path_train=train_data_path, data_path_test=test_data_path,
                         feature_path_train=train_feature_path, feature_path_test=test_feature_path, scip_or_fico=fico_or_scip,
                         impo_df=import_df, number_features=num_features, treffmas=treffen, scaling_for_features=scaling,
                         tiefe=wie_tief, estimatoren=anzahl_estimators, anzahl_feat_per_tree=feats_per_tree,
                         outlier_removal=remove_outlier)
        acc_sgm_to_csv(accuracy_drops_lin=accuracy_lin, sgm_drops_lin=sgm_lin, data_set=fico_or_scip, accuracy_drops_for=[], sgm_drops_for=[],
                       feature_ranking='linear', directory=treffen)

    elif model.lower() == 'forest':
        import_df = import_df.sort_values(by=['Forest Score'], ascending=True)
        forest_reduction(data_path_train=train_data_path, data_path_test=test_data_path,
                         feature_path_train=train_feature_path, feature_path_test=test_feature_path,
                         scip_or_fico=fico_or_scip,
                         impo_df=import_df, number_features=num_features, treffmas=treffen, scaling_for_features=scaling,
                         tiefe=wie_tief, estimatoren=anzahl_estimators, anzahl_feat_per_tree=feats_per_tree,
                         outlier_removal=remove_outlier)
        acc_sgm_to_csv(accuracy_drops_lin=[], sgm_drops_lin=[], data_set=fico_or_scip, accuracy_drops_for=accuracy_for, sgm_drops_for=sgm_for,
                       feature_ranking='forest', directory=treffen)

    elif model.lower() == 'combined':
        print('LINEAR')
        linear_reduction(train_data_path, test_data_path, train_feature_path, test_feature_path, scip_or_fico=fico_or_scip,
                         impo_df=import_df, number_features=num_features, treffmas=treffen, scaling_for_features=scaling,
                         tiefe=wie_tief, estimatoren=anzahl_estimators, anzahl_feat_per_tree=feats_per_tree,
                         outlier_removal=remove_outlier)
        print('FOREST')
        forest_reduction(train_data_path, test_data_path, train_feature_path, test_feature_path, scip_or_fico=fico_or_scip,
                         impo_df=import_df, number_features=num_features, treffmas=treffen, scaling_for_features=scaling,
                         tiefe=wie_tief, estimatoren=anzahl_estimators, anzahl_feat_per_tree=feats_per_tree,
                         outlier_removal=remove_outlier)
        acc_sgm_to_csv(accuracy_lin, sgm_lin, fico_or_scip, accuracy_for, sgm_for, feature_ranking='combined',
                       directory=treffen)

    else:
        print(f'Model {model} not recognized. use linear, forest or combined')
        sys.exit(1)

def trainer(imputation, model, model_name, X_train, y_train, seed, data_set):
    start = time.time()
    # Build pipeline
    # Update model-specific parameters
    if model_name == "RandomForest":
        model.random_state = seed
    # Add imputation
    steps = [('imputer', SimpleImputer(strategy=imputation))]
    # Add model
    steps.append(('model', model))
    # Create pipeline
    pipeline = Pipeline(steps)
    # Train the pipeline
    pipeline.fit(X_train, y_train)

    end = time.time()
    return pipeline, end-start

def predict(pipeline, X_test, y_test):
    start = time.time()
    # Evaluate on the test set
    relevant_indices = y_test[y_test != 0].index
    y_test_relevant = y_test.loc[relevant_indices]
    y_pred_relevant = pipeline.predict(X_test.loc[relevant_indices, :])
    y_pred_relevant = pd.Series(y_pred_relevant, index=relevant_indices, name='Prediction')
    end = time.time()
    return y_pred_relevant, y_test_relevant, end - start

def regression(data_train, data_test, data_set_name, label_train, label_test, features_train, features_test, feature_names, models, scalers, imputer, random_seeds,
               label_scale=False, mid_threshold=0.1, extreme_threshold=4.0):

    """
    Gets a csv file as input
    trains a ml model
    outputs csv files: Accuracy, Time save/loss, Feature Importance
    """
    start_time = time.time()
    training_time = 0
    prediction_time = 0

    if label_scale:
        label_train = label_scaling(label_train)
        label_test = label_scaling(label_test)
        mid_threshold = np.log1p(mid_threshold)/np.log(10.0)
        extreme_threshold = np.log1p(extreme_threshold)/np.log(10.0)

    accuracy_dictionary = {}
    run_time_dictionary = {}
    sum_time_dictionary = {}
    prediction_dictionary = {}
    importance_dictionary = {}
    logging.info(f"{'-' * 80}\n{data_set_name}\n{'-' * 80}")

    #  on train set
    accuracy_dictionary_train = {}
    run_time_dictionary_train = {}
    prediction_dictionary_train = {}

    for model_name, model in models.items():
        randos = random_seeds
        # print(f'Training {model_name}')
        if model_name not in ['LinearRegression', 'RandomForest']:
            logging.info(f'AHHHHHHHHHHHHHHHHHHHHHHHH. {model_name} is not a valid regressor!')
            continue
        if model_name == 'LinearRegression':
            print("LINEAR")
            randos = [random_seeds[0]] #for linear regressor its no difference what random seed
        elif model_name == 'RandomForest':
            print("RANDOMFOREST")
        for imputation in imputer:
            for scaler in scalers:
                for seed in randos:
                    X_train, X_test, y_train, y_test = features_train, features_test, label_train, label_test
                    # train the model
                    trained_model, tt = trainer(imputation, model, model_name, X_train, y_train, seed, data_set_name)
                    training_time += tt

                    # let the model make predictions
                    y_pred_relevant, y_test_relevant, pt = predict(trained_model, X_test, y_test)

                    prediction_time += pt
                    # get accuracy measure for the model
                    accuracy_dictionary[model_name+'_'+imputation+'_'+str(scaler)+'_'+str(seed)] = get_accuracy(y_pred_relevant, y_test_relevant, mid_threshold=mid_threshold, extreme_threshold=extreme_threshold)
                    # add sgm of run time for this setting to run_time_df
                    run_time_dictionary[model_name+'_'+imputation+'_'+str(scaler)+'_'+str(seed)] = get_predicted_run_time_sgm(y_pred_relevant, data_test, shift=50)
                    sum_time_dictionary[model_name+'_'+imputation+'_'+str(scaler)+'_'+str(seed)] = get_predicted_sum_time(y_pred_relevant, data_test)
                    # return actual prediction
                    prediction_dictionary[model_name+'_'+imputation+'_'+str(scaler)+'_'+str(seed)] = y_pred_relevant.to_list()
                    # feature importance
                    if model_name == 'LinearRegression':
                        importances = trained_model.named_steps['model'].coef_
                    else:
                        importances = trained_model.named_steps['model'].feature_importances_
                    importance_dictionary[model_name+'_'+imputation+'_'+str(scaler)+'_'+str(seed)] = importances.tolist()

                    # make predictions on train set for checking of over/underfitting
                    y_pred_train, y_test_train, pt_train = predict(trained_model, X_train, y_train)
                    accuracy_dictionary_train[
                        model_name + '_' + imputation + '_' + str(scaler) + '_' + str(seed)] = get_accuracy(
                        y_pred_train, y_test_train, mid_threshold, extreme_threshold)
                    # add sgm of run time for this setting to run_time_df
                    run_time_dictionary_train[model_name + '_' + imputation + '_' + str(scaler) + '_' + str(
                        seed)] = get_predicted_run_time_sgm(y_pred_train, data_train, shift=data_train["Final solution time (cumulative)"].mean())
                    prediction_dictionary_train[
                        model_name + '_' + imputation + '_' + str(scaler) + '_' + str(seed)] = y_pred_train.to_list()

    if any(len(d) == 0 for d in [importance_dictionary,prediction_dictionary,accuracy_dictionary,run_time_dictionary]):
        # handle the empty case
        dictionaries = [importance_dictionary, accuracy_dictionary, run_time_dictionary, prediction_dictionary]
        dict_names = ['importance_dictionary', 'accuracy_dictionary', 'run_time_dictionary', 'prediction_dictionary']
        empty_dicts = []
        for i in range(len(dictionaries)):
            if len(dictionaries[i]) == 0:
                empty_dicts.append(dict_names[i])
        print(f'Error while creating: {empty_dicts}')
        end_time = time.time()
        print(f'Final time: {end_time - start_time}')
        return None

    else:
        feature_importance_df = pd.DataFrame.from_dict(importance_dictionary, orient='columns').astype(float)
        feature_importance_df.index = feature_names
        impo_ranking = ranking_feature_importance(feature_importance_df, feature_importance_df.index.tolist())
        accuracy_df = pd.DataFrame.from_dict(accuracy_dictionary, orient='index')
        accuracy_df.columns = ['Accuracy', 'Mid Accuracy', 'Mid Instances', 'Extreme Accuracy', 'Extreme Instances']
        accuracy_df.loc[:, ['Accuracy', 'Mid Accuracy','Extreme Accuracy']] = accuracy_df.loc[:, ['Accuracy', 'Mid Accuracy','Extreme Accuracy']].astype(float)


        prediction_df = get_prediction_df(prediction_dictionary).astype(float)
        run_time_df = pd.DataFrame.from_dict(run_time_dictionary, orient='index').astype(float)
        run_time_df.columns = ['Predicted', 'Mixed', 'Int', 'VBS']
        sum_time_df = pd.DataFrame.from_dict(sum_time_dictionary, orient='index').astype(float)
        sum_time_df.columns = ['Predicted', 'Mixed', 'Int', 'VBS']
        sum_time_df["Mixed-Int"] = sum_time_df["Mixed"] - sum_time_df["Int"]
        sum_time_df["PREDICTION-VBS"] = sum_time_df["Predicted"]-sum_time_df["VBS"]

    run_time_df_trainset = pd.DataFrame.from_dict(run_time_dictionary_train, orient='index').astype(float)
    run_time_df_trainset.columns = ['Predicted', 'Mixed', 'Int', 'VBS']
    prediction_dictionary_train_df = get_prediction_df(prediction_dictionary_train)

    accuracy_df_trainset = pd.DataFrame.from_dict(accuracy_dictionary_train, orient='index')
    accuracy_df_trainset.columns = ['Accuracy', 'Mid Accuracy', 'Mid Instances', 'Extreme Accuracy', 'Extreme Instances']
    accuracy_df_trainset.loc[:, ['Accuracy', 'Mid Accuracy','Extreme Accuracy']] = accuracy_df_trainset.loc[:, ['Accuracy', 'Mid Accuracy','Extreme Accuracy']].astype(
        float)

    end_time = time.time()
    logging.info(f'Training time: {training_time}')
    logging.info(f'Prediction time: {prediction_time}')
    logging.info(f'Final time: {end_time - start_time}')

    return accuracy_df, run_time_df, sum_time_df, prediction_df, feature_importance_df,  impo_ranking, accuracy_df_trainset, run_time_df_trainset, prediction_dictionary_train_df

def run_regression_pipeline(data_name, trainset_data, testset_data, trainset_feat, testset_feat, prefix, treffplusx, models, imputer, scalers,
                            hundred_seeds:list, feature_subset:list, label_scale=False, outlier_threshold=None,remove_outlier=False):
    train_data = pd.read_csv(trainset_data)
    test_data = pd.read_csv(testset_data)
    train_features = pd.read_csv(trainset_feat)
    test_features = pd.read_csv(testset_feat)

    # treat -1 as missing value
    train_data = train_data.replace([-1, -1.0], np.nan)
    test_data = test_data.replace([-1, -1.0], np.nan)
    train_features = train_features.replace([-1, -1.0], np.nan)
    test_features = test_features.replace([-1, -1.0], np.nan)

    train_label = train_data['Cmp Final solution time (cumulative)']
    test_label = test_data['Cmp Final solution time (cumulative)']

    if remove_outlier:
        if data_name == 'fico':
            if outlier_threshold is None:
                train_features, train_label = kick_outlier(train_features, train_label, threshold=350)
                test_features, test_label = kick_outlier(test_features, test_label, threshold=350)
            else:
                train_features, train_label = kick_outlier(train_features, train_label, threshold=outlier_threshold)
                test_features, test_label = kick_outlier(test_features, test_label, threshold=outlier_threshold)
        elif data_name == 'scip_default':
            if outlier_threshold is None:
                train_features, train_label = kick_outlier(train_features, train_label, threshold=40)
                test_features, test_label = kick_outlier(test_features, test_label, threshold=40)
            else:
                train_features, train_label = kick_outlier(train_features, train_label, threshold=outlier_threshold)
                test_features, test_label = kick_outlier(test_features, test_label, threshold=outlier_threshold)
        else:
            sys.exit(f'Invalid data_name: {data_name}')

    # if feature_subset is None, it means that we use all features

    if feature_subset is not None:
        train_features = train_features[feature_subset]
        test_features = test_features[feature_subset]
    print("In Regression: Train Features:", len(train_features.columns), "Test Features:", len(test_features.columns))
    # Set scalers
    def scale_feats(df, scaler):
        if scaler == 'Quantile':
            qt = QuantileTransformer(output_distribution='normal', n_quantiles=int(len(df) * 0.8))
        elif scaler == 'Yeo':
            qt = PowerTransformer(method='yeo-johnson')
        else:
            sys.exit(f'Invalid scaler: {scaler}')
        # Fit and transform the DataFrame
        if isinstance(df, pd.Series):
            # Reshape data for transformation
            df = pd.DataFrame(data=df, index= df.index, columns = [df.name])

        scaled_data = qt.fit_transform(df)
        # Create a new DataFrame with the same index and column names.
        scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
        return scaled_df
    if data_name == 'fico':
        scaler_for_data = 'Quantile'
    else:
        scaler_for_data = 'Yeo'
    train_features = scale_feats(train_features, scaler_for_data)
    test_features = scale_feats(test_features, scaler_for_data)
    # Run regression
    acc_df, runtime_df, sumtime_df, prediction_df, importance_df, impo_ranking, acc_train, sgm_train, pred_train = (
        regression(data_train=train_data, data_test=test_data, data_set_name=data_name, label_train=train_label,
                   label_test=test_label, features_train=train_features, features_test=test_features,
                   feature_names=test_features.columns, models=models, imputer=imputer, random_seeds=hundred_seeds,
                   label_scale=label_scale, scalers=[scaler_for_data]))

    # Save results
    base_path = f'{treffplusx}'

    create_directory(f'{treffplusx}')
    # TestSet
    acc_df.to_csv(f'{base_path}/Accuracy/{prefix}_acc_df.csv', index=True)
    runtime_df.to_csv(f'{base_path}/SGM/{prefix}_sgm_runtime.csv', index=True)
    sumtime_df.to_csv(f'{base_path}/SGM/sumtime.csv', index=True)
    prediction_df.to_csv(f'{base_path}/Prediction/{prefix}_prediction_df.csv')
    importance_df.to_csv(f'{base_path}/Importance/{prefix}_importance_df.csv', index=True)
    impo_ranking.to_csv(f'{base_path}/Importance/{data_name}_importance_ranking.csv', index=False)
    # TrainSet
    acc_train.to_csv(f'{base_path}/Accuracy/{prefix}_acc_trainset.csv', index=True)
    sgm_train.to_csv(f'{base_path}/SGM/{prefix}_sgm_trainset.csv', index=True)
    pred_train.to_csv(f'{base_path}/Prediction/{prefix}_prediction_trainset.csv')

    return print_accuracy(acc_df), print_sgm(runtime_df, data_name, len(train_features.columns))

def main(path_to_trainset_data:str, path_to_testset_data:str, path_to_trainset_feat:str, path_to_testset_feat:str,
         scip_or_fico:str, treffplusx='Wurm', label_scalen=True, feature_scaling=None, feature_subset=None, models=None,
         outlier_value = None, kick_outlier=False, max_depth=200, estimatoren=100, max_feature_number=17):
    setup_directory(treffplusx)


    if label_scalen:
        treffplusx = treffplusx+'/ScaledLabel'
    else:
        treffplusx = treffplusx + '/UnscaledLabel'


    if models is None:
        models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=estimatoren, max_depth=max_depth,
                                                  max_features=max_feature_number, random_state=0, n_jobs=-1)
        }

    imputer = ['median']

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

    create_directory(treffplusx)

    akk, ess_geh_ehm = run_regression_pipeline(
        data_name=scip_or_fico,
        trainset_data=f'{path_to_trainset_data}',
        testset_data=f'{path_to_testset_data}',
        trainset_feat=f'{path_to_trainset_feat}',
        testset_feat=f'{path_to_testset_feat}',
        prefix=scip_or_fico,
        treffplusx=treffplusx,
        models=models,
        imputer=imputer,
        scalers=feature_scaling,
        hundred_seeds=hundred_seeds,
        label_scale=label_scalen,
        feature_subset=feature_subset,
        outlier_threshold=outlier_value,
        remove_outlier=kick_outlier
    )
    return akk, ess_geh_ehm


# TODO OUTLIER IGNORED => DONT IGNORE ANYMORE
# I do until now: Instead of kicking outlier scale whole label down, but maybe both even better:)

def get_number_of_runs(path_to_runs):
    folders = [f for f in os.listdir(path_to_runs) if os.path.isdir(os.path.join(path_to_runs, f))]
    return len(folders)

base_data_directory = '/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Bases'

fico_data_5 = '/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Bases/FICO/Cleaned/9_5_ready_to_ml.csv'
fico_feats_5 = '/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Bases/FICO/Cleaned/9_5_ready_to_ml_features.csv'

fico_data_6 = '/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Bases/FICO/Cleaned/9_6_ready_to_ml.csv'
fico_feats_6 = '/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Bases/FICO/Cleaned/9_6_ready_to_ml_features.csv'

fiveonsix_5_17 = f'/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/5on6/depth5feats17/NoOutlier/Logged'
sixonfive_5_17 = f'/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/6on5/depth5feats17/NoOutlier/Logged'
fiveonsix_feat_reduction_5_17 = f'/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/5on6/depth5feats17'
sixonfive_feat_reduction_5_17 = f'/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/6on5/depth5feats17'

fiveonsix_5_2 = f'/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/5on6/depth5feats2/NoOutlier/Logged'
sixonfive_5_2 = f'/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/6on5/depth5feats2/NoOutlier/Logged'
fiveonsix_feat_reduction_5_2 = f'/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/5on6/depth5feats2'
sixonfive_feat_reduction_5_2 = f'/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/6on5/depth5feats2'

fiveonsix_10_2 = f'/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/5on6/depth10feats2/NoOutlier/Logged'
sixonfive_10_2 = f'/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/6on5/depth10feats2/NoOutlier/Logged'
fiveonsix_feat_reduction_10_2 = f'/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/5on6/depth10feats2'
sixonfive_feat_reduction_10_2 = f'/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/6on5/depth10feats2'

fiveonsix_10_17 = f'/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/5on6/depth10feats17/NoOutlier/Logged'
sixonfive_10_17 = f'/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/6on5/depth10feats17/NoOutlier/Logged'
fiveonsix_feat_reduction_10_17 = f'/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/5on6/depth10feats17'
sixonfive_feat_reduction_10_17 = f'/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/6on5/depth10feats17'


feats = 2
depth = 5
estis = 100
def feat_reduction_hyperparameter_tuning():
    list = [(5, 2,   fiveonsix_5_2,   sixonfive_5_2,   fiveonsix_feat_reduction_5_2,   sixonfive_feat_reduction_5_2),
            (5, 17,  fiveonsix_5_17,  sixonfive_5_17,  fiveonsix_feat_reduction_5_17,  sixonfive_feat_reduction_5_17),
            (10, 2,  fiveonsix_10_2,  sixonfive_10_2,  fiveonsix_feat_reduction_10_2,  sixonfive_feat_reduction_10_2),
            (10, 17, fiveonsix_10_17, sixonfive_10_17, fiveonsix_feat_reduction_10_17, sixonfive_feat_reduction_10_17)]

    for combination in list:
        depth = combination[0]
        feats = combination[1]
        fiveonsix = combination[2]
        sixonfive = combination[3]
        fiveonsix_feat_reduction = combination[4]
        sixonfive_feat_reduction = combination[5]
        estis = 100
        main(path_to_trainset_data=fico_data_5, path_to_testset_data=fico_data_6,
                             path_to_trainset_feat=fico_feats_5, path_to_testset_feat=fico_feats_6,
                             scip_or_fico='fico', treffplusx=sixonfive, label_scalen=True, feature_scaling=['Quantile'], kick_outlier=True,
                             max_depth=depth, estimatoren=estis, max_feature_number=feats)

        main(path_to_trainset_data=fico_data_6, path_to_testset_data=fico_data_5,
                             path_to_trainset_feat=fico_feats_6, path_to_testset_feat=fico_feats_5,
                             scip_or_fico='fico', treffplusx=fiveonsix, label_scalen=True, feature_scaling=['Quantile'], kick_outlier=True,
                             max_depth=depth, estimatoren=estis, max_feature_number=feats)



        feature_reduction(train_data_path=fico_data_6, test_data_path=fico_data_5, train_feature_path=fico_feats_6,
                          test_feature_path=fico_feats_5, fico_or_scip='fico', model='linear',
                          treffen=fiveonsix_feat_reduction, scaling='Quantile', wie_tief=depth,
                          anzahl_estimators=estis, feats_per_tree=feats, remove_outlier=True)

        feature_reduction(train_data_path=fico_data_6, test_data_path=fico_data_5, train_feature_path=fico_feats_6,
                          test_feature_path=fico_feats_5, fico_or_scip='fico', model='forest',
                          treffen=fiveonsix_feat_reduction, scaling='Quantile', wie_tief=depth,
                          anzahl_estimators=estis, feats_per_tree=feats, remove_outlier=True)

        feature_reduction(train_data_path=fico_data_5, test_data_path=fico_data_6, train_feature_path=fico_feats_5,
                          test_feature_path=fico_feats_6, fico_or_scip='fico', model='linear',
                          treffen=sixonfive_feat_reduction, scaling='Quantile', wie_tief=depth,
                          anzahl_estimators=estis, feats_per_tree=feats, remove_outlier=True)

        feature_reduction(train_data_path=fico_data_5, test_data_path=fico_data_6, train_feature_path=fico_feats_5,
                          test_feature_path=fico_feats_6, fico_or_scip='fico', model='forest',
                          treffen=sixonfive_feat_reduction, scaling='Quantile', wie_tief=depth,
                          anzahl_estimators=estis, feats_per_tree=feats, remove_outlier=True)


import stats_per_combi_july
def plot_hyperparameter_tuning(regress=True, acc=True, sgm=False):
    tiefen = [5, 10] # 5
    number_estimators = [100]
    maximum_features = [2, 17]#,4,6,8,10,12,14,16,17]
    print("------6on5------")
    for depth in tiefen:
        for estimator in number_estimators:
            for feat_num in maximum_features:
                print(depth, estimator, feat_num)
                maximum_depth = depth

                sixonfive = f'/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/6on5_depth{maximum_depth}_estimators{estimator}_numfeats{feat_num}/NoOutlier/Logged/'
                if regress:
                    main(path_to_trainset_data=fico_data_5, path_to_testset_data=fico_data_6,
                         path_to_trainset_feat=fico_feats_5, path_to_testset_feat=fico_feats_6,
                         scip_or_fico='fico', treffplusx=sixonfive, label_scalen=True, feature_scaling=['Quantile'], kick_outlier=True,
                         max_depth=maximum_depth, estimatoren=estimator, max_feature_number=feat_num)

                print("PLOTTING")
                if acc:
                    stats_per_combi_july.train_vs_test_acuracy(f"/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/6on5_depth{maximum_depth}_estimators{estimator}_numfeats{feat_num}/NoOutlier/Logged/ScaledLabel",
                                          version=f"Trainingset: 9.5, Testset 9.6, Depth: {depth}, #Feats: {feat_num}", fico_or_scip='fico',
                                          save_to=f"/Users/fritz/Downloads/ZIB/Master/Writing/Tex/FinaleBilder/FightOverfitting/96auf95_depth{maximum_depth}_estimators{estimator}_numfeats{feat_num}/Overfitting")
                if sgm:
                    stats_per_combi_july.train_vs_test_sgm(f"/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/6on5_depth{maximum_depth}_estimators{estimator}_numfeats{feat_num}/NoOutlier/Logged/ScaledLabel",
                                          version=f"Trainingset: 9.5, Testset 9.6, Depth: {depth}, #Feats: {feat_num}", fico_or_scip='fico',
                                          save_to=f"/Users/fritz/Downloads/ZIB/Master/Writing/Tex/FinaleBilder/FightOverfitting/96auf95_depth{maximum_depth}_estimators{estimator}_numfeats{feat_num}/Overfitting")
    print("------5on6------")
    for depth in tiefen:
        for estimator in number_estimators:
            for feat_num in maximum_features:
                print(depth, estimator, feat_num)
                maximum_depth = depth

                fiveonsix = f'/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/5on6_depth{maximum_depth}_estimators{estimator}_numfeats{feat_num}/NoOutlier/Logged/'
                if regress:
                    main(path_to_trainset_data=fico_data_6, path_to_testset_data=fico_data_5,
                         path_to_trainset_feat=fico_feats_6, path_to_testset_feat=fico_feats_5,
                         scip_or_fico='fico', treffplusx=fiveonsix, label_scalen=True, feature_scaling=['Quantile'], kick_outlier=True,
                         max_depth=maximum_depth, estimatoren=estimator, max_feature_number=feat_num)
                print("PLOTTING")
                if acc:
                    stats_per_combi_july.train_vs_test_acuracy(
                        f"/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/5on6_depth{maximum_depth}_estimators{estimator}_numfeats{feat_num}/NoOutlier/Logged/ScaledLabel",
                        version=f"Trainingset: 9.6, Testset 9.5, Depth: {depth}, #Feats: {feat_num}", fico_or_scip='fico',
                        save_to=f"/Users/fritz/Downloads/ZIB/Master/Writing/Tex/FinaleBilder/FightOverfitting/95auf96_depth{maximum_depth}_estimators{estimator}_numfeats{feat_num}/Overfitting")
                if sgm:
                    stats_per_combi_july.train_vs_test_sgm(
                        f"/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/5on6_depth{maximum_depth}_estimators{estimator}_numfeats{feat_num}/NoOutlier/Logged/ScaledLabel",
                        version=f"Trainingset: 9.6, Testset 9.5, Depth: {depth}, #Feats: {feat_num}", fico_or_scip='fico',
                        save_to=f"/Users/fritz/Downloads/ZIB/Master/Writing/Tex/FinaleBilder/FightOverfitting/95auf96_depth{maximum_depth}_estimators{estimator}_numfeats{feat_num}/Overfitting")

# plot_hyperparameter_tuning(regress=True, acc=True, sgm=False)