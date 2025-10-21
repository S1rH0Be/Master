import time
import pandas as pd
import numpy as np
import os
import sys

from stats_per_combi_july import ranking_feature_importance, train_vs_test_acuracy, train_vs_test_sgm
from visualize_july import plot_feature_reduction

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

def shifted_geometric_mean(values):
    values = np.array(values)

    if values.dtype == 'object':
        # Attempt to convert to float
        values = values.astype(float)
    #TODO: Add shift as parameter
    shift=10
    # Check if shift is large enough
    if shift <= -values.min():
        print(f"Shift {shift} too small. Minimum value is {values.min()}, so shift must be > {-values.min()}")
        raise ValueError(f"Shift too small. Minimum value is {values.min()}, so shift must be > {-values.min()}")

    # Shift the values by the constant
    shifted_values = values + shift

    shifted_values_log = np.log(shifted_values)  # Step 1: Log of each element in shifted_values

    log_mean = np.mean(shifted_values_log)  # Step 2: Compute the mean of the log values
    geo_mean = np.exp(log_mean) - shift
    return geo_mean

def get_features_label(data_frame, feature_df, chosen_features):
    features = feature_df[chosen_features]
    label = data_frame['Cmp Final solution time (cumulative)']
    return features, label

def label_scaling(label):
    y_pos = label[label >= 0]
    y_neg = label[label < 0]
    # log1p calculates log(1+x) numerically stable
    y_pos_log = np.log1p(y_pos)/np.log(10)
    y_neg_log = (np.log1p(abs(y_neg))/np.log(10)) * -1
    y_log = pd.concat([y_pos_log, y_neg_log]).sort_index()
    return y_log

#TODO: Add get_Accuracy for all instances not just relevant
def get_accuracy(prediction, actual, mid_threshold, extreme_threshold):
    # Filter for nonzero labels
    nonzero_indices = actual != 0
    y_test_nonzero = actual[nonzero_indices]
    def overall_accuracy():
        y_pred_nonzero = prediction[nonzero_indices]
        # Calculate percentage of correctly predicted signs
        correct_signs = np.sum(np.sign(y_test_nonzero) == np.sign(y_pred_nonzero))

        percent_correct_signs = correct_signs / len(y_test_nonzero) * 100 if len(y_test_nonzero) > 0 else np.nan #revert
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

def get_predicted_run_time_sgm(y_pred, data, threshold_for_pred):
    predicted_time = pd.Series(index=y_pred.index, name='Predicted Run Time')
    indices = y_pred.index
    for i in indices:
        if y_pred.loc[i] < threshold_for_pred:
            predicted_time.loc[i] = data.loc[i, 'Final solution time (cumulative) Int']
        else:
            predicted_time.loc[i] = data.loc[i, 'Final solution time (cumulative)']
    try:
        #TODO: Shift interactive
        shift=10
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
    return predicted_time_sum, mixed_time_sum, int_time_sum, vbs_sum

def get_importance_col(importances, feature_names, model_name, imputation, scaler):
    importance_df = pd.DataFrame({f'{model_name}_{imputation}_{scaler}': importances}, index=feature_names)
    return importance_df

def get_prediction_df(dictionary, data):
    indices = data.index
    # Pad each list with zeros
    for key in dictionary:
        prediction_series = pd.Series([0.0]*len(data), index=indices)
        prediction_series.loc[dictionary[key].index] = dictionary[key]
        dictionary[key] = prediction_series
    return pd.DataFrame.from_dict(dictionary, orient='columns')

def kick_outlier(feat_df, label_series, threshold : int):
    indices_to_keep = label_series[label_series.abs() <= threshold].index
    feats_to_keep = feat_df.loc[indices_to_keep, :]
    labels_to_keep = label_series.loc[indices_to_keep]
    return feats_to_keep, labels_to_keep

def feature_reduction(data_path, feature_path, data_set, model: str, treffmas, skalierer, remove_outlier=False,
                      outlier_threshold=350, max_depp=200, estimatores=100, num_features_tree=None, max_blatt=None,
                      instances_split=2):
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
                         num_feats=None, depth=None, estis=100, leafs=None,
                         instances=2):
        feature_list = impo_df_for['Feature'].tolist()
        for reduce_by in range(len(feature_list)):
            treff = f'{treffmas_for}/FeatureReduction/forest/{reduce_by}'
            feature_set = feature_list[:(number_features_for - reduce_by)]
            acc, sgm = main(path_to_data=data_path_for, path_to_features=feature_path_for, scip_default=scip_default_for,
                            fico=fico_for, treffplusx=treff, feature_scaling=scaling_for_features,
                 label_scalen=True, feature_subset=feature_set,
                 models={'RandomForest': RandomForestRegressor(n_estimators=estis, max_depth=depth,
                                                               max_features=num_feats, max_leaf_nodes=leafs,
                                                               min_samples_split=instances,
                                                               random_state=0, n_jobs=-1)},
                            kick_outliers=outlier_removal, outlier_value=threshold_outlier)

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
            feature_set = feature_list[:(number_features_lin - reduce_by)]

            acc, sgm = main(path_to_data=data_path_lin,
                            path_to_features=feature_path_lin,
                            scip_default=scip_default_lin,
                            fico=fico_lin,
                            treffplusx=treff,
                            label_scalen=True,
                            feature_scaling=scaling_for_features,
                            feature_subset=feature_set,
                            models={'LinearRegression': LinearRegression()},
                            kick_outliers=outlier_removal,
                            outlier_value=threshold_outlier)

            accuracy_lin.append(acc)
            sgm_lin.append(sgm)

    if data_set.lower() == 'fico':
        # TODO: change path to be interactive
        impo_df = pd.read_csv(
            f'{treffmas}/Importance/fico_importance_ranking.csv')
        scip_default = False
        fico = True
        treffmas = f'{treffmas}'
        number_features = len(impo_df)

    elif data_set.lower() == 'scip':
        impo_df = pd.read_csv(
            f'{treffmas}/Importance/scip_default_importance_ranking.csv')
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
                         threshold_outlier=outlier_threshold, num_feats=num_features_tree, depth=max_depp, estis=estimatores, leafs=max_blatt,
                         instances=instances_split)
        acc_sgm_to_csv(accuracy_drops_lin=[], sgm_drops_lin=[], data_set_name=data_set, accuracy_drops_for=accuracy_for,
                       sgm_drops_for=sgm_for, feature_ranking='forest', directory=treffmas)

    else:
        print(f'Model {model} not recognized. use linear, forest or combined')
        sys.exit(1)

def trainer(imputation, scaler, model, model_name, X_train, y_train, seed):
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

    return pipeline

# TODO: Add predict on all instances not only relevant
def predict(pipeline, X_test, y_test):
    # Evaluate on the test set
    relevant_indices = y_test[y_test != 0].index
    y_test_relevant = y_test.loc[relevant_indices]
    y_pred_relevant = pipeline.predict(X_test.loc[relevant_indices, :])
    y_pred_relevant = pd.Series(y_pred_relevant, index=relevant_indices, name='Prediction')
    return y_pred_relevant, y_test_relevant

def regression(data, data_set_name, label, features, feature_names, models, scalers, imputer, random_seeds,
               label_scale=False, mid_threshold=0.1, extreme_threshold=4.0, when_int_for_mixed=0.0):

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
            continue

        for imputation in imputer:
            for scaler in scalers:
                for seed in random_seeds:
                    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2,
                                                                        random_state=seed)
                    # train the model
                    trained_model = trainer(imputation, scaler, model, model_name, X_train, y_train, seed)
                    # let the model make predictions
                    y_pred_relevant, y_test_relevant = predict(trained_model, X_test, y_test)
                    # get accuracy measure for the model
                    accuracy_dictionary[model_name+'_'+imputation+'_'+str(scaler)+'_'+str(seed)] = get_accuracy(y_pred_relevant, y_test_relevant, mid_threshold=mid_threshold, extreme_threshold=extreme_threshold)
                    # add sgm of run time for this setting to run_time_df
                    run_time_dictionary[model_name+'_'+imputation+'_'+str(scaler)+'_'+str(seed)] = get_predicted_run_time_sgm(y_pred_relevant, data, threshold_for_pred=when_int_for_mixed)
                    sum_time_dictionary[model_name+'_'+imputation+'_'+str(scaler)+'_'+str(seed)] = get_predicted_sum_time(y_pred_relevant, data)
                    # return actual prediction
                    prediction_dictionary[model_name+'_'+imputation+'_'+str(scaler)+'_'+str(seed)] = y_pred_relevant
                    # feature importance
                    if model_name == 'LinearRegression':
                        importances = trained_model.named_steps['model'].coef_
                    else:
                        importances = trained_model.named_steps['model'].feature_importances_

                    importance_dictionary[model_name+'_'+imputation+'_'+str(scaler)+'_'+str(seed)] = importances.tolist()

                    # make predictions on train set for checking of over/underfitting

                    y_pred_train, y_test_train = predict(trained_model, X_train, y_train)
                    accuracy_dictionary_train[
                        model_name + '_' + imputation + '_' + str(scaler) + '_' + str(seed)] = get_accuracy(
                        y_pred_train, y_test_train, mid_threshold, extreme_threshold)
                    # add sgm of run time for this setting to run_time_df
                    run_time_dictionary_train[model_name + '_' + imputation + '_' + str(scaler) + '_' + str(
                    seed)] = get_predicted_run_time_sgm(y_pred_train, data, threshold_for_pred=when_int_for_mixed)
                    prediction_dictionary_train[
                    model_name + '_' + imputation + '_' + str(scaler) + '_' + str(seed)] = y_pred_train
    if any(len(d) == 0 for d in [importance_dictionary,prediction_dictionary,accuracy_dictionary,run_time_dictionary]):
        # handle the empty case
        dictionaries = [importance_dictionary, accuracy_dictionary, run_time_dictionary, prediction_dictionary]
        dict_names = ['importance_dictionary', 'accuracy_dictionary', 'run_time_dictionary', 'prediction_dictionary']
        empty_dicts = []
        for i in range(len(dictionaries)):
            if len(dictionaries[i]) == 0:
                empty_dicts.append(dict_names[i])
        print(f'Error while creating: {empty_dicts}')
        return None

    else:
        feature_importance_df = pd.DataFrame.from_dict(importance_dictionary, orient='columns').astype(float)
        feature_importance_df.index = feature_names
        impo_ranking = ranking_feature_importance(feature_importance_df, feature_importance_df.index.tolist())
        accuracy_df = pd.DataFrame.from_dict(accuracy_dictionary, orient='index')
        accuracy_df.columns = ['Accuracy', 'Mid Accuracy', 'Mid Instances', 'Extreme Accuracy', 'Extreme Instances']
        accuracy_df.loc[:, ['Accuracy', 'Mid Accuracy','Extreme Accuracy']] = accuracy_df.loc[:, ['Accuracy', 'Mid Accuracy','Extreme Accuracy']].astype(float)

        prediction_df = get_prediction_df(prediction_dictionary, data).astype(float)
        run_time_df = pd.DataFrame.from_dict(run_time_dictionary, orient='index').astype(float)
        run_time_df.columns = ['Predicted', 'Mixed', 'Int', 'VBS']
        sum_time_df = pd.DataFrame.from_dict(sum_time_dictionary, orient='index').astype(float)
        sum_time_df.columns = ['Predicted', 'Mixed', 'Int', 'VBS']
        sum_time_df["Mixed-Int"] = sum_time_df["Mixed"] - sum_time_df["Int"]
        sum_time_df["PREDICTION-VBS"] = sum_time_df["Predicted"]-sum_time_df["VBS"]

    run_time_df_trainset = pd.DataFrame.from_dict(run_time_dictionary_train, orient='index').astype(float)
    run_time_df_trainset.columns = ['Predicted', 'Mixed', 'Int', 'VBS']
    prediction_dictionary_train_df = get_prediction_df(prediction_dictionary_train, data)

    accuracy_df_trainset = pd.DataFrame.from_dict(accuracy_dictionary_train, orient='index')
    accuracy_df_trainset.columns = ['Accuracy', 'Mid Accuracy', 'Mid Instances', 'Extreme Accuracy', 'Extreme Instances']
    accuracy_df_trainset.loc[:, ['Accuracy', 'Mid Accuracy','Extreme Accuracy']] = accuracy_df_trainset.loc[:, ['Accuracy', 'Mid Accuracy','Extreme Accuracy']].astype(
        float)


    return accuracy_df, run_time_df, sum_time_df, prediction_df, feature_importance_df, impo_ranking, accuracy_df_trainset, run_time_df_trainset, prediction_dictionary_train_df

def run_regression_pipeline(data_name, data_path, feats_path, is_excel, prefix, base_path, models, imputer, scalers,
                            hundred_seeds:list, feature_subset:list, label_scale=False,
                            remove_outlier=False, outlier_threshold=350, mid_threshi=0.1, ex_thresh=4.0, when_follow_pred=0.0):
    # Load data
    if is_excel:
        data = pd.read_excel(data_path)
        features = pd.read_excel(feats_path).iloc[:, 1:]
    else:
        data = pd.read_csv(data_path)
        features = pd.read_csv(feats_path)
    # treat -1 as missing value
    data = data.replace([-1, -1.0], np.nan)
    features = features.replace([-1, -1.0], np.nan)
    label = data['Cmp Final solution time (cumulative)']

    if remove_outlier:
        features, label = kick_outlier(features, label, threshold=outlier_threshold)
    # if feature_subset is None, it means that we use all features
    if feature_subset is not None:
        features = features[feature_subset]
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
        regression(data, data_name, label, features, features.columns, models, scalers, imputer, hundred_seeds,
                   label_scale, mid_threshold=mid_threshi, extreme_threshold=ex_thresh, when_int_for_mixed=when_follow_pred))

    # Save results
    create_directory(base_path)
    # TestSet
    acc_df.to_csv(f'{base_path}/Accuracy/{prefix}_acc_df.csv', index=True)
    runtime_df.to_csv(f'{base_path}/SGM/{prefix}_sgm_runtime.csv', index=True)
    sumtime_df.to_csv(f'{base_path}/SGM/sumtime.csv', index=True)
    prediction_df.to_csv(f'{base_path}/Prediction/{prefix}_prediction_df.csv')
    importance_df.to_csv(f'{base_path}/Importance/{prefix}_importance_df.csv', index=True)
    impo_ranking.to_csv(f'{base_path}/Importance/{data_name}_importance_ranking.csv', index=False)
    # TrainingSet
    acc_train.to_csv(f'{base_path}/Accuracy/{prefix}_acc_trainset.csv', index=True)
    sgm_train.to_csv(f'{base_path}/SGM/{prefix}_sgm_trainset.csv', index=True)
    pred_train.to_csv(f'{base_path}/Prediction/{prefix}_prediction_trainset.csv')


def main(path_to_data:str, path_to_features:str, scip_default=False, fico=False, treffplusx='Wurm', label_scalen=True,
         feature_scaling=None, feature_subset=None, models=None, kick_outliers=False,
         outlier_value = None, max_tiefe=200, estimatoren=100, number_features_tree=None, max_blatter=None,
         instances_per_split=2, mid_threshold=0.1, extreme_threshold=4.0, prediction_threshold=0.0):

    setup_directory(treffplusx)

    akk, ess_geh_ehm = None, None

    if label_scalen:
        treffplusx = treffplusx+'/'
    else:
        treffplusx = treffplusx + '/UnscaledLabel'


    if models is None:
        models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=estimatoren, max_depth=max_tiefe,
                                                  max_features=number_features_tree, max_leaf_nodes=max_blatter,
                                                  min_samples_split=instances_per_split,
                                                  random_state=0, n_jobs=-1)
        }

    imputer = ['median', 'mean']

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
        run_regression_pipeline(
            data_name = 'scip_default',
            data_path=f'{path_to_data}',
            feats_path=f'{path_to_features}',
            is_excel=False,
            prefix='scip',
            base_path=treffplusx,
            models=models,
            imputer=imputer,
            scalers=feature_scaling,
            hundred_seeds=hundred_seeds,
            label_scale=label_scalen,
            feature_subset=feature_subset,
            remove_outlier=kick_outliers,
            outlier_threshold=outlier_value,
            ex_thresh=extreme_threshold,
            when_follow_pred=prediction_threshold
        )

    if fico:
        print('FICO')
        run_regression_pipeline(
            data_name='fico',
            data_path=f'{path_to_data}',
            feats_path=f'{path_to_features}',
            is_excel=False,
            prefix='fico',
            base_path=treffplusx,
            models=models,
            imputer=imputer,
            scalers=feature_scaling,
            hundred_seeds=hundred_seeds,
            label_scale=label_scalen,
            feature_subset=feature_subset,
            remove_outlier=kick_outliers,
            outlier_threshold=outlier_value,
            mid_threshi= mid_threshold,
            ex_thresh=extreme_threshold,
            when_follow_pred=prediction_threshold,
        )

        if not (fico|scip_default):
            print('Neither fico or scip_default is specified. use one of them!')
            sys.exit(1)


    return akk, ess_geh_ehm


def hyper_hyper_tuner_tuner(main_regression=False, feat_reduction=False, plot_main=False, plot_reduction=False,
                                 forest=True, linear=True, out_thresh=350, directory='',
                                 fico=True, scip=False, feat_subset=None, scalerz=None, picture_save='', title=None,
                                 skalieren_des_labels=True, ex_threshold = None, pred_thresh=None, combinations=None,
                                 instances_split=None, fico_data_5='', fico_feats_5='', fico_data_6='', fico_feats_6='',
                                 scip_data_10='', scip_feats_10=''):

    if scalerz is None:
        scalerz = ['Quantile', 'NoScaling', 'Standard', 'MinMax', 'Robust', 'Yeo']

    if fico:
        data_sets = [(fico_data_5, fico_feats_5, "5"), (fico_data_6, fico_feats_6, "6")]
    else:
        data_sets = [(scip_data_10, scip_feats_10, "10")]
    if pred_thresh is None:
        pred_thresh = [0.0]
    if instances_split is None:
        instances_split = [2]
    if combinations is None:
        combinations = [(None, None), (5, None), (10, None)]
    if ex_threshold is None:
        ex_threshold = [4.0]

    for data_set in data_sets:
        for combination in combinations:
            for threshi in pred_thresh:
                for ex in ex_threshold:
                    for instance in instances_split:
                        if main_regression:
                            if fico:
                                fico_final = f'{directory}{data_set[2]}/depth{combination[0]}'
                                main(path_to_data=data_set[0], path_to_features=data_set[1], scip_default=False, fico=True,
                                     treffplusx=fico_final, feature_subset=feat_subset, models=None,
                                     label_scalen=skalieren_des_labels, feature_scaling=scalerz,
                                     kick_outliers=True, outlier_value=out_thresh, max_tiefe=combination[0],
                                     number_features_tree=combination[1], max_blatter=None,
                                     instances_per_split=instance, mid_threshold=0.1, extreme_threshold=ex,
                                     prediction_threshold=threshi)
                            if scip:
                                scip_final = f'{directory}/depth{combination[0]}'
                                main(path_to_data=data_set[0], path_to_features=data_set[1], scip_default=True,
                                     fico=False,
                                     treffplusx=scip_final, feature_subset=feat_subset, models=None,
                                     label_scalen=skalieren_des_labels, feature_scaling=scalerz,
                                     kick_outliers=True, outlier_value=out_thresh, max_tiefe=combination[0],
                                     number_features_tree=combination[1], max_blatter=None,
                                     instances_per_split=instance, mid_threshold=0.1, extreme_threshold=ex,
                                     prediction_threshold=threshi)

                        if plot_main:
                            if fico:
                                path = f"{directory}{data_set[2]}/depth{combination[0]}"
                                train_vs_test_acuracy(
                                    path,
                                    version= title,
                                    fico_or_scip='fico',
                                    save_to=f"{picture_save}/depth{combination[0]}")
                                train_vs_test_sgm(
                                    path,
                                    version=title,
                                    fico_or_scip='fico',
                                    save_to=f"{picture_save}/depth{combination[0]}")
                            if scip:
                                path = f"{directory}/depth{combination[0]}"
                                train_vs_test_acuracy(
                                    path,
                                    version=title,
                                    fico_or_scip='scip',
                                    save_to=f"{picture_save}/depth{combination[0]}")
                                train_vs_test_sgm(
                                    path,
                                    version=title,
                                    fico_or_scip='scip',
                                    save_to=f"{picture_save}/depth{combination[0]}")

                        if feat_reduction:
                            if fico:
                                fico_feat_reduction = f'{directory}/FICO9.{data_set[2]}/depth{combination[0]}/'
                                if linear:
                                    feature_reduction(data_path=data_set[0], feature_path=data_set[1], data_set='fico',
                                                      model='linear', treffmas=fico_feat_reduction, skalierer=scalerz,
                                                      remove_outlier=True, outlier_threshold=out_thresh, max_depp=combination[0],
                                                      max_blatt=None, instances_split=instance)
                                if forest:
                                    feature_reduction(data_path=data_set[0], feature_path=data_set[1], data_set='fico' , model='forest',
                                                      treffmas=fico_feat_reduction, skalierer=scalerz, remove_outlier=True,
                                                      outlier_threshold=out_thresh, num_features_tree=None, max_depp=combination[0],
                                                      max_blatt=None, instances_split=instance)
                            if scip:
                                scip_reduction = f'{directory}/SCIP/depth{combination[0]}'
                                if linear:
                                    feature_reduction(data_path=data_set[0], feature_path=data_set[1], data_set='scip',
                                                      model='linear', treffmas=scip_reduction,
                                                      skalierer=scalerz,
                                                      remove_outlier=True, outlier_threshold=out_thresh,
                                                      max_depp=combination[0],
                                                      max_blatt=None, instances_split=instance)
                                    if forest:
                                        feature_reduction(data_path=data_set[0], feature_path=data_set[1],
                                                          data_set='scip', model='forest',
                                                          treffmas=scip_reduction, skalierer=scalerz,
                                                          remove_outlier=True,
                                                          outlier_threshold=out_thresh, num_features_tree=None,
                                                          max_depp=combination[0],
                                                          max_blatt=None, instances_split=instance)

                        if plot_reduction:
                            if fico:
                                if linear:
                                    if data_set[2] == "5":
                                        thresh_linear = 14
                                    else:
                                        thresh_linear = 13
                                    plot_feature_reduction(
                                        directory=f'{directory}/FICO9.{data_set[2]}/depth{combination[0]}/FeatureReduction/fico',
                                        fico_or_scip='fico', base_data="", feature_ranking='linear',
                                        title_add_on="",
                                        threshold=thresh_linear,
                                        save_to=f'{picture_save}/FICO9.{data_set[2]}/FeatureReduction')
                                if forest:
                                    thresh_forest = 14
                                    plot_feature_reduction(
                                        directory=f'{directory}/FICO9.{data_set[2]}/depth{combination[0]}/FeatureReduction/fico',
                                        fico_or_scip='fico', base_data="", feature_ranking='forest',
                                        title_add_on="",
                                        threshold=thresh_forest,
                                        save_to=f'{picture_save}/FICO9.{data_set[2]}/FeatureReduction')
                            if scip:
                                if linear:
                                    thresh_linear = 12
                                    plot_feature_reduction(
                                        directory=f'{directory}/depth{combination[0]}/FeatureReduction/scip',
                                        fico_or_scip='scip', base_data="", feature_ranking='linear',
                                        title_add_on="",
                                        threshold=thresh_linear,
                                        save_to=f'{picture_save}/SCIP/FeatureReduction')
                                if forest:
                                    thresh_forest = 10
                                    plot_feature_reduction(
                                        directory=f'{directory}/depth{combination[0]}/FeatureReduction/scip',
                                        fico_or_scip='scip', base_data="", feature_ranking='forest',
                                        title_add_on="",
                                        threshold=thresh_forest,
                                        save_to=f'{picture_save}/SCIP/FeatureReduction')




