import time
import pandas as pd
import numpy as np
import os
import sys

from stats_per_combi_july import ranking_feature_importance, train_vs_test_acuracy
from visualize_july import plot_feature_reduction
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
        sgm_extreme_accuracy = get_sgm_series(data_frame['Extreme Accuracy'].dropna(), data_frame['Extreme Accuracy'].dropna().mean()+0.1)
        return [sgm_accuracy, sgm_mid_accuracy, sgm_extreme_accuracy]

    def visualize_acc(data_frame, title: str = 'Accuracy'):
        """Gets sgm of accuracy of a run for the linear and the forest model."""
        accuracy_df = data_frame.copy()
        # if no corresponding acc is found just plot it as 0

        linear_rows = [lin_rows for lin_rows in accuracy_df.index if 'LinearRegression' in lin_rows]
        linear_df = accuracy_df.loc[linear_rows, :]

        forest_rows = [for_row for for_row in accuracy_df.index if 'RandomForest' in for_row]
        forest_df = accuracy_df.loc[forest_rows,:]

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

    def get_values_for_plot(dataframe:pd.DataFrame, data_set_name):

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

        values_relative = relative_to_default(value_dictionary, data_set_name)
        return values_relative

    def sgm_plot(dataframe, title:str, data_set_name:str, plot=True):
        values = get_values_for_plot(dataframe, data_set_name)
        if plot:
            labels = ['PrefInt', 'Mixed', 'Predicted', 'VBS']

            bar_colors = ['turquoise', 'magenta']

            # Create the plot
            plt.figure(figsize=(8, 5))
            plt.bar(labels, values, color=bar_colors)
            plt.title(title)
            if data_set_name.lower() == 'fico':
                plt.ylim(0.5, 1.35)  # Set y-axis limits for visibility
            else:
                plt.ylim(0.8, 1.06)  # Set y-axis limits for visibility
            plt.xticks(rotation=45, fontsize=6)
            # Display the plot
            plt.show()
            plt.close()
        return np.round(values[2:], 2)

    def call_sgm_visualization(df, num_features, data_set_name, plot=True):
        return sgm_plot(df, f'SGM {num_features}', data_set_name, plot)

    return call_sgm_visualization(sgm_df, number_features, data_set, plot=False)

def get_features_label(data_frame, feature_df, chosen_features):
    features = feature_df[chosen_features]
    label = data_frame['Cmp Final solution time (cumulative)']
    return features, label

def label_scaling(label):
    # print("LABEL SCALING")
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

def feature_reduction(data_path, feature_path, data_set, model: str, treffmas, skalierer, remove_outlier=False,
                      outlier_threshold=350, wie_tief=None, wieviele_feats=1.0):
    create_directory(f'{treffmas}/FeatureReduction')
    accuracy_lin = []
    sgm_lin = []
    accuracy_for = []
    sgm_for = []

    def acc_sgm_to_csv(accuracy_drops_lin, sgm_drops_lin, data_set_name, accuracy_drops_for, sgm_drops_for, feature_ranking, directory):
        path = f'{directory}/FeatureReduction/{data_set_name}'
        create_directory(path)

        if len(accuracy_drops_lin)>0:
            acc_reduct_lin_df = pd.DataFrame(data=accuracy_drops_lin, columns=['Accuracy', 'Mid Accuracy', 'Extreme Accuracy'])
            acc_reduct_lin_df.to_csv(
                f'{path}/acc_{data_set_name}_{feature_ranking}_linear.csv',
                index=False)

        if len(sgm_drops_lin)>0:
            sgm_reduct_lin_df = pd.DataFrame(data=sgm_drops_lin, columns=['SGM relative to Default', 'VBS'])
            sgm_reduct_lin_df.to_csv(
                f'{path}/sgm_{data_set_name}_{feature_ranking}_linear.csv',
                index=False)

        if len(accuracy_drops_for)>0:
            acc_reduct_for_df = pd.DataFrame(data=accuracy_drops_for, columns=['Accuracy', 'Mid Accuracy', 'Extreme Accuracy'])
            acc_reduct_for_df.to_csv(
                f'{path}/acc_{data_set_name}_{feature_ranking}_forest.csv',
                index=False)

        if len(sgm_drops_for)>0:
            sgm_reduct_for_df = pd.DataFrame(data=sgm_drops_for, columns=['SGM relative to Default', 'VBS'])
            sgm_reduct_for_df.to_csv(
                f'{path}/sgm_{data_set_name}_{feature_ranking}_forest.csv',
                index=False)

    def forest_reduction(data_path_for, feature_path_for,scip_default_for, fico_for, impo_df_for, number_features_for,
                         treffmas_for, scaling_for_features, outlier_removal=False, threshold_outlier=350,
                         tiefe=None, num_feats=1.0):
        feature_list = impo_df_for['Feature'].tolist()
        for reduce_by in range(len(feature_list)):
            treff = f'{treffmas_for}/FeatureReduction/forest/{reduce_by}'
            feature_set = feature_list[:(number_features_for - reduce_by)]
            # print("#Features in ForesReduction:", len(feature_set))
            # treffmas = treffmas_for.replace('/FICO', '')
            acc, sgm = main(data_path_for, feature_path_for, scip_default_for, fico_for, treffplusx=treff, feature_scaling=scaling_for_features,
                 label_scalen=True, feature_subset=feature_set,
                 models={'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=tiefe,
                                                               max_features=num_feats,  random_state=0, n_jobs=-1)},
                            kick_outliers=outlier_removal)

            accuracy_for.append(acc)
            sgm_for.append(sgm)

    def linear_reduction(data_path_lin, feature_path_lin, scip_default_lin, fico_lin, impo_df_lin, number_features_lin,
                         treffmas_lin, scaling_for_features, outlier_removal = False, threshold_outlier=350,):
        feature_list = impo_df_lin['Feature'].tolist()
        for reduce_by in range(len(feature_list)):
            if scip_default:
                add_on = 'scip'
            elif fico:
                add_on = 'fico'
            else:
                print("Neither scip nor fico")
                sys.exit(1)
            treff = f'{treffmas_lin}/FeatureReduction/{add_on}/linear/{reduce_by}'
            if reduce_by>0:
                print(feature_list[(number_features_lin - reduce_by)])
            feature_set = feature_list[:(number_features_lin - reduce_by)]

            acc, sgm = main(data_path_lin, feature_path_lin, scip_default_lin, fico_lin, treffplusx=treff,
                            feature_scaling=scaling_for_features, label_scalen=True, feature_subset=feature_set,
                            models={'LinearRegression': LinearRegression()}, kick_outliers=outlier_removal)

            accuracy_lin.append(acc)
            sgm_lin.append(sgm)

    if data_set.lower() == 'fico':
        # TODO: change path to be interactive
        impo_df = pd.read_csv(
            f'{treffmas}/NoOutlier/Logged/ScaledLabel/Importance/fico_importance_ranking.csv')
        scip_default = False
        fico = True
        treffmas = f'{treffmas}'
        number_features = len(impo_df)

    elif data_set.lower() == 'scip':
        impo_df = pd.read_csv(
            f'{treffmas}/NoOutlier/Logged/ScaledLabel/Importance/scip_default_importance_ranking.csv')
        scip_default = True
        fico = False
        treffmas = f'{treffmas}'
        number_features = len(impo_df)

    else:
        print(f'Data set {data_set} not recognized. use scip or fico')
        sys.exit(1)

    if model.lower() == 'linear':
        impo_df = impo_df.sort_values(by=['Linear Score'], ascending=True)
        linear_reduction(data_path_lin=data_path, feature_path_lin=feature_path, scip_default_lin=scip_default,
                         fico_lin=fico, impo_df_lin=impo_df, number_features_lin=number_features, treffmas_lin=treffmas,
                         scaling_for_features=skalierer, outlier_removal=remove_outlier, threshold_outlier=outlier_threshold)
        acc_sgm_to_csv(accuracy_drops_lin=accuracy_lin, sgm_drops_lin=sgm_lin, data_set_name=data_set,
                       accuracy_drops_for=[], sgm_drops_for=[], feature_ranking='linear', directory=treffmas)

    elif model.lower() == 'forest':
        impo_df = impo_df.sort_values(by=['Forest Score'], ascending=True)
        forest_reduction(data_path_for=data_path, feature_path_for=feature_path, scip_default_for=scip_default,
                         fico_for=fico, impo_df_for=impo_df, number_features_for=number_features, treffmas_for=treffmas,
                         scaling_for_features=skalierer, outlier_removal=remove_outlier,
                         threshold_outlier=outlier_threshold, tiefe=wie_tief, num_feats=wieviele_feats)
        acc_sgm_to_csv(accuracy_drops_lin=[], sgm_drops_lin=[], data_set_name=data_set, accuracy_drops_for=accuracy_for,
                       sgm_drops_for=sgm_for, feature_ranking='forest', directory=treffmas)

    elif model.lower() == 'combined':
        print('COMBINED')
        print('LINEAR')
        linear_reduction(data_path_lin=data_path, feature_path_lin=feature_path, scip_default_lin=scip_default,
                         fico_lin=fico, impo_df_lin=impo_df, number_features_lin=number_features,
                         outlier_removal=remove_outlier, threshold_outlier=outlier_threshold,
                         treffmas_lin=treffmas, scaling_for_features=skalierer)
        print('FOREST')
        forest_reduction(data_path_for=data_path, feature_path_for=feature_path, scip_default_for=scip_default,
                         fico_for=fico, impo_df_for=impo_df, number_features_for=number_features, treffmas_for=treffmas,
                         outlier_removal=remove_outlier, threshold_outlier=outlier_threshold,
                         scaling_for_features=skalierer, tiefe=wie_tief, num_feats=wieviele_feats)
        acc_sgm_to_csv(accuracy_drops_lin=accuracy_lin, sgm_drops_lin=sgm_lin, data_set_name=data_set,
                       accuracy_drops_for=accuracy_for, sgm_drops_for=sgm_for,
                       feature_ranking='combined', directory=treffmas)

    else:
        print(f'Model {model} not recognized. use linear, forest or combined')
        sys.exit(1)

def trainer(imputation, scaler, model, model_name, X_train, y_train, seed):
    start = time.time()
    # Build pipeline
    # Update model-specific parameters
    if model_name == "RandomForest":
        model.random_state = seed
    # Add imputation
    steps = [('imputer', SimpleImputer(strategy=imputation))]
    # Add scaling if applicable
    if scaler:
        steps.append(('scaler', scaler))
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

def regression(data, data_set_name, label, features, feature_names, models, scalers, imputer, random_seeds,
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
        label = label_scaling(label)
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
        if model_name not in ['LinearRegression', 'RandomForest']:
            logging.info(f'AHHHHHHHHHHHHHHHHHHHHHHHH. {model_name} is not a valid regressor!')
            continue
        if model_name == 'LinearRegression':
            print("LINEAR")
        elif model_name == 'RandomForest':
            print("RANDOMFOREST")
        for imputation in imputer:
            for scaler in scalers:
                for seed in random_seeds:
                    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2,
                                                                        random_state=seed)
                    # train the model
                    trained_model, tt = trainer(imputation, scaler, model, model_name, X_train, y_train, seed)
                    training_time += tt

                    # let the model make predictions
                    y_pred_relevant, y_test_relevant, pt = predict(trained_model, X_test, y_test)

                    prediction_time += pt
                    # get accuracy measure for the model
                    accuracy_dictionary[model_name+'_'+imputation+'_'+str(scaler)+'_'+str(seed)] = get_accuracy(y_pred_relevant, y_test_relevant, mid_threshold=mid_threshold, extreme_threshold=extreme_threshold)
                    # add sgm of run time for this setting to run_time_df
                    run_time_dictionary[model_name+'_'+imputation+'_'+str(scaler)+'_'+str(seed)] = get_predicted_run_time_sgm(y_pred_relevant, data, shift=50)
                    sum_time_dictionary[model_name+'_'+imputation+'_'+str(scaler)+'_'+str(seed)] = get_predicted_sum_time(y_pred_relevant, data)
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
                        seed)] = get_predicted_run_time_sgm(y_pred_train, data, shift=data["Final solution time (cumulative)"].mean())
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

    return accuracy_df, run_time_df, sum_time_df, prediction_df, feature_importance_df, impo_ranking, accuracy_df_trainset, run_time_df_trainset, prediction_dictionary_train_df

def run_regression_pipeline(data_name, data_path, feats_path, is_excel, prefix, treffplusx, models, imputer, scalers,
                            hundred_seeds:list, feature_subset:list, label_scale=False,
                            remove_outlier=False, outlier_threshold=350):
    # Load data
    if is_excel:
        data = pd.read_excel(data_path)
        features = pd.read_excel(feats_path).iloc[:, 1:]
    else:
        data = pd.read_csv(data_path)
        features = pd.read_csv(feats_path)

    # treat -1 as missing value
    # TODO check which features have this property
    data = data.replace([-1, -1.0], np.nan)
    features = features.replace([-1, -1.0], np.nan)
    label = data['Cmp Final solution time (cumulative)']

    if remove_outlier:
        features, label = kick_outlier(features, label, threshold=outlier_threshold)
    # if feature_subset is None, it means that we use all features
    if feature_subset is not None:
        features = features[feature_subset]
    # print("In Regression Features:", len(features.columns))
    # Set scalers
    if scalers is not None:
        scaler_dict = { 'NoScaling':None,
                'Standard':StandardScaler(),
                'MinMax':MinMaxScaler(),
                'Robust':RobustScaler(),
                'Yeo':PowerTransformer(method='yeo-johnson'),
                'Quantile': QuantileTransformer(output_distribution='normal', n_quantiles=int(len(features) * 0.8))
                }
        scalers = [scaler_dict[scaler] for scaler in scalers]

    elif scalers is None:
        scalers = [None]
    else:
        print("Check input for scalers. Needs to be None or list of valid scalers.")
        sys.exit(1)
    # Run regression
    acc_df, runtime_df, sumtime_df, prediction_df, importance_df, impo_ranking, acc_train, sgm_train, pred_train = (
        regression(data, data_name, label, features, features.columns, models, scalers, imputer, hundred_seeds, label_scale))

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

    return print_accuracy(acc_df), print_sgm(runtime_df, data_name, len(features.columns))

def main(path_to_data:str, path_to_features:str, scip_default=False, fico=False, treffplusx='Wurm', label_scalen=True,
         feature_scaling=None, feature_subset=None, models=None, kick_outliers=False,
         outlier_value = None, max_tiefe=200, estimatoren=100, number_features_tree=17):
    start_time = time.time()

    setup_directory(treffplusx)

    akk, ess_geh_ehm = None, None

    if label_scalen:
        treffplusx = treffplusx+'/ScaledLabel'
    else:
        treffplusx = treffplusx + '/UnscaledLabel'


    if models is None:
        models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=estimatoren, max_depth=max_tiefe,
                                                  max_features=number_features_tree, random_state=0, n_jobs=-1)
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


    if scip_default:
        print('SCIP')
        akk, ess_geh_ehm = run_regression_pipeline(
            data_name = 'scip_default',
            data_path=f'{path_to_data}',
            feats_path=f'{path_to_features}',
            is_excel=False,
            prefix='scip',
            treffplusx=treffplusx,
            models=models,
            imputer=imputer,
            scalers=feature_scaling,
            hundred_seeds=hundred_seeds,
            label_scale=label_scalen,
            feature_subset=feature_subset,
            remove_outlier=kick_outliers,
            outlier_threshold=outlier_value
        )

    if fico:
        print('FICO')
        akk, ess_geh_ehm = run_regression_pipeline(
            data_name='fico',
            data_path=f'{path_to_data}',
            feats_path=f'{path_to_features}',
            is_excel=False,
            prefix='fico',
            treffplusx=treffplusx,
            models=models,
            imputer=imputer,
            scalers=feature_scaling,
            hundred_seeds=hundred_seeds,
            label_scale=label_scalen,
            feature_subset=feature_subset,
            remove_outlier=kick_outliers,
            outlier_threshold=outlier_value
        )

        if not (fico|scip_default):
            print('Neither fico or scip_default is specified. use one of them!')
            sys.exit(1)


    end_time = time.time()
    print(f'Time elapsed: {end_time - start_time}')
    return akk, ess_geh_ehm


# TODO OUTLIER IGNORED => DONT IGNORE ANYMORE
# I do until now: Instead of kicking outlier scale whole label down, but maybe both even better:)

def get_number_of_runs(path_to_runs):
    folders = [f for f in os.listdir(path_to_runs) if os.path.isdir(os.path.join(path_to_runs, f))]
    return len(folders)


def hyper_hyper_tuner_tuner_fico(main_regression=True, feat_reduction=False, out_thresh=350):
    fico_data_5 = '/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Bases/FICO/Cleaned/9_5_ready_to_ml.csv'
    fico_feats_5 = '/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Bases/FICO/Cleaned/9_5_ready_to_ml_features.csv'

    fico_data_6 = '/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Bases/FICO/Cleaned/9_6_ready_to_ml.csv'
    fico_feats_6 = '/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Bases/FICO/Cleaned/9_6_ready_to_ml_features.csv'

    data_sets = [(fico_data_5, fico_feats_5, "5"), (fico_data_6, fico_feats_6, "6")]
    combinations = [(5, 2), (5, 4), (5, 5), (5, 17), (10, 2), (10, 4), (10, 5), (10, 17)]
    for data_set in data_sets:
        for combination in combinations:
            print(data_set)
            print(combination)
            if main_regression:
                fico_final = f'/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/FICO{data_set[2]}/depth{combination[0]}feats{combination[1]}/NoOutlier/Logged/'
                print(fico_final)
                main(path_to_data=data_set[0], path_to_features=data_set[1], scip_default=False, fico=True,
                     treffplusx=fico_final, models=None, label_scalen=True, feature_scaling=['Quantile'], kick_outliers=True,
                     outlier_value=out_thresh, max_tiefe=combination[0], number_features_tree=combination[1])
                train_vs_test_acuracy(f"/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/FICO{data_set[2]}/depth{combination[0]}feats{combination[1]}/NoOutlier/Logged/ScaledLabel",
                                      version=f"9.{data_set[2]}, max_depth={combination[0]}, max_feats={combination[1]}", fico_or_scip='fico',
                                      save_to=f"/Users/fritz/Downloads/ZIB/Master/Writing/Tex/FinaleBilder/FightOverfitting/9{data_set[2]}/Overfitting")

            if feat_reduction:
                fico_feat_reduction = f'/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/FICO{data_set[2]}'

                feature_reduction(data_set[0], data_set[1], 'fico', model='linear',
                                  treffmas=fico_feat_reduction, skalierer=['Quantile'], remove_outlier=True,
                                  outlier_threshold=out_thresh)
                feature_reduction(data_set[0], data_set[1], 'fico' , model='forest',
                                  treffmas=fico_feat_reduction, skalierer=['Quantile'], remove_outlier=True,
                                  outlier_threshold=out_thresh)
                # plot_feature_reduction(f'/Users/fritz/Downloads/ZIB/Master/SeptemberFinal/Runs/Final/FightOverfitting/FICO{data_set[2]}/depth{combination[0]}feats{combination[1]}/NoOutlier/Logged/')
                print("DOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOONE")
hyper_hyper_tuner_tuner_fico(main_regression=False, feat_reduction=True)