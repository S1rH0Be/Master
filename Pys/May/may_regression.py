import time
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
import logging


treffplustage = 'TreffenMasDiez/ScaledLabel'

# Setup logging configuration
os.makedirs(os.path.join(f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffplustage}'), exist_ok=True)
logging.basicConfig(
    filename=f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffplustage}/regression_log.txt',  # Log file name
    level=logging.INFO,             # Minimum level to log
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

def create_directory(parent_name):
    base_path = f'/Users/fritz/Downloads/ZIB/Master/Treffen/{parent_name}'
    subdirs = ['Prediction', 'Accuracy', 'Importance', 'RunTime']
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
        raise ValueError(f"Shift too small. Minimum value is {values.min()}, so shift must be > {-values.min()}")

    shifted_values = values + shift

    shifted_values_log = np.log(shifted_values)  # Step 1: Log of each element in shifted_values
    log_mean = np.mean(shifted_values_log)  # Step 2: Compute the mean of the log values
    geo_mean = np.exp(log_mean) - shift
    # geo_mean = np.round(geo_mean, 6)
    return geo_mean

def get_features_label(data_frame, feature_df, chosen_features):
    features = feature_df[chosen_features]
    label = data_frame['Cmp Final solution time (cumulative)']
    return features, label

def label_scaling(label):
    y_pos = label[label >= 0]
    y_neg = label[label < 0]
    y_pos_log = np.log(y_pos + 1)
    y_neg_log = np.log(abs(y_neg) + 1) * -1
    y_log = pd.concat([y_pos_log, y_neg_log]).sort_index()
    return y_log

def get_accuracy(prediction, actual, extreme_value):
    # Filter for nonzero labels
    nonzero_indices = actual != 0
    y_test_nonzero = actual[nonzero_indices]
    y_pred_nonzero = prediction[nonzero_indices]

    # Calculate percentage of correctly predicted signs
    correct_signs = np.sum(np.sign(y_test_nonzero) == np.sign(y_pred_nonzero))
    percent_correct_signs = correct_signs / len(y_test_nonzero) * 100 if len(y_test_nonzero) > 0 else np.nan

    # Filter for extreme labels
    extreme_indices = abs(actual) >= extreme_value


    y_test_extreme = actual[extreme_indices]
    number_extreme_signs = (len(y_test_extreme), len(y_test_nonzero))
    y_pred_extreme = prediction[extreme_indices]

    # Calculate percentage of correctly predicted signs
    correct_extreme_signs = np.sum(np.sign(y_test_extreme) == np.sign(y_pred_extreme))
    percent_correct_extreme_signs = correct_extreme_signs / len(y_test_extreme) * 100 if len(
        y_test_extreme) > 0 else np.nan

    return percent_correct_signs, percent_correct_extreme_signs, number_extreme_signs

def get_sgm_comparison(y_pred, y_test):
    pred_df = pd.DataFrame({'Prediction': y_pred, 'Actual': y_test},
                           index=y_pred.index)
    pred_df['Right or Wrong'] = (np.sign(pred_df['Prediction']) == np.sign(pred_df['Actual'])).astype(int)
    # add column containing the absolute difference in prediction and actual
    pred_df['Abs Time Diff'] = abs(pred_df['Prediction'] - pred_df['Actual'])
    return pred_df

def get_predicted_run_time_sgm(y_pred, data, shift):
    predicted_time = pd.Series(index=y_pred.index, name='Predicted Run Time')
    indices = y_pred.index
    for i in indices:
        if y_pred.loc[i] > 0:
            predicted_time.loc[i] = data.loc[i, 'Final solution time (cumulative) Mixed']
        else:
            predicted_time.loc[i] = data.loc[i, 'Final solution time (cumulative) Int']

    try:
        sgm_predicted = shifted_geometric_mean(predicted_time, shift)
        sgm_mixed = shifted_geometric_mean(data['Final solution time (cumulative) Mixed'].loc[indices], shift)
        sgm_int = shifted_geometric_mean(data['Final solution time (cumulative) Int'].loc[indices], shift)
        sgm_vbs = shifted_geometric_mean(data['Virtual Best'].loc[indices], shift)
    except ValueError as e:
        logging.error(f"SGM failed due to shift: {e}")
        return None, None, None, None  # or raise again if you want the pipeline to crash
    return sgm_predicted, sgm_mixed, sgm_int, sgm_vbs

def get_importance_col(importances, feature_names, model_name, imputation, scaler):
    importance_df = pd.DataFrame({f'{model_name}_{imputation}_{scaler}': importances}, index=feature_names)
    return importance_df

def get_prediction_df(dictionary):
    max_len = max(len(lst) for lst in dictionary.values())
    # Pad each list with zeros
    for key in dictionary:
        dictionary[key] += [0] * (max_len - len(dictionary[key]))
    return pd.DataFrame.from_dict(dictionary, orient='columns')

def trainer(imputation, scaler, model, model_name, X_train, y_train, seed, data_set):
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
    joblib.dump(model, f'models/{data_set}/{model_name}_{seed}.pkl')
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

def regression(data, data_set_name, features_df, feature_names, models, scalers, imputer, random_seeds, label_scale=False, extreme_threshold=4.0):
    """
    Gets a csv file as input
    trains a ml model
    outputs csv files: Accuracy, Time save/loss, Feature Importance
    """
    start_time = time.time()
    training_time = 0
    prediction_time = 0
    features, label = get_features_label(data, features_df, feature_names)

    if label_scale:
        label = label_scaling(label)
        extreme_threshold = np.log(extreme_threshold)

    accuracy_dictionary = {}
    run_time_dictionary = {}
    prediction_dictionary = {}
    importance_dictionary = {}
    logging.info(f"{'-' * 80}\n{data_set_name}\n{'-' * 80}")

    for model_name, model in models.items():
        if model_name not in ['LinearRegression', 'RandomForest']:
            logging.info(f'AHHHHHHHHHHHHHHHHHHHHHHHH. {model_name} is not a valid regressor!')
            continue
        for imputation in imputer:
            for scaler in scalers:
                for seed in random_seeds:
                    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2,
                                                                        random_state=seed)
                    # train the model
                    trained_model, tt = trainer(imputation, scaler, model, model_name, X_train, y_train, seed, data_set_name)
                    training_time += tt
                    # let the model make predictions
                    y_pred_relevant, y_test_relevant, pt = predict(trained_model, X_test, y_test)
                    prediction_time += pt
                    # get accuracy measure for the model
                    accuracy_dictionary[model_name+'_'+imputation+'_'+str(scaler)+'_'+str(seed)] = get_accuracy(y_pred_relevant, y_test_relevant, extreme_threshold)
                    # add sgm of run time for this setting to run_time_df
                    run_time_dictionary[model_name+'_'+imputation+'_'+str(scaler)+'_'+str(seed)] = get_predicted_run_time_sgm(y_pred_relevant, data, shift=50)
                    # return actual prediction
                    prediction_dictionary[model_name+'_'+imputation+'_'+str(scaler)+'_'+str(seed)] = y_pred_relevant.to_list()
                    # feature importance
                    if model_name == 'LinearRegression':
                        importances = trained_model.named_steps['model'].coef_
                    else:
                        importances = trained_model.named_steps['model'].feature_importances_
                    importance_dictionary[model_name+'_'+imputation+'_'+str(scaler)+'_'+str(seed)] = importances.tolist()

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

        accuracy_df = pd.DataFrame.from_dict(accuracy_dictionary, orient='index')
        accuracy_df.columns = ['Accuracy', 'Extreme Accuracy', 'Extreme Instances']
        accuracy_df.loc[:, ['Accuracy', 'Extreme Accuracy']] = accuracy_df.loc[:, ['Accuracy', 'Extreme Accuracy']].astype(float)

        prediction_df = get_prediction_df(prediction_dictionary).astype(float)

        run_time_df = pd.DataFrame.from_dict(run_time_dictionary, orient='index').astype(float)
        run_time_df.columns = ['Predicted', 'Mixed', 'Int', 'VBS']

    end_time = time.time()
    logging.info(f'Training time: {training_time}')
    logging.info(f'Prediction time: {prediction_time}')
    logging.info(f'Final time: {end_time - start_time}')
    print(f'{data_set_name} is done, after {end_time - start_time}!')
    return accuracy_df, run_time_df, prediction_df, feature_importance_df

def run_regression_pipeline(data_name, data_path, feats_path, is_excel, prefix, treffplusx, models, imputer, hundred_seeds, label_scale=False):
    # Load data
    if is_excel:
        data = pd.read_excel(data_path)
        features = pd.read_excel(feats_path).iloc[:, 1:]
    else:
        data = pd.read_csv(data_path)
        features = pd.read_csv(feats_path)

    # Set scalers
    scalers = [
        StandardScaler(),
        MinMaxScaler(),
        RobustScaler(),
        PowerTransformer(method='yeo-johnson'),
        QuantileTransformer(output_distribution='normal', n_quantiles=int(len(data) * 0.8))
    ]

    # Run regression
    acc_df, runtime_df, prediction_df, importance_df = regression(
        data, data_name, features, features.columns, models, scalers, imputer, hundred_seeds, label_scale
    )

    # Save results
    base_path = f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffplusx}'
    acc_df.to_csv(f'{base_path}/Accuracy/{prefix}_acc_df.csv', index=True)
    runtime_df.to_csv(f'{base_path}/RunTime/{prefix}_sgm_runtime.csv', index=True)
    prediction_df.to_csv(f'{base_path}/Prediction/{prefix}_prediction_df.csv')
    importance_df.to_csv(f'{base_path}/Importance/{prefix}_importance_df.csv', index=True)

def main(scip_default=False, scip_no_pseudo=False, fico=False, treffplusx='Wurm', label_scalen= False):
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
    }
    imputer = ['mean', 'median']
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
        run_regression_pipeline(
            data_name = 'scip_default',
            data_path=f'/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/cleaned_scip/scip_default_clean_data.csv',
            feats_path=f'/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/cleaned_scip/scip_default_clean_feats.csv',
            is_excel=False,
            prefix='scip',
            treffplusx=treffplusx,
            models=models,
            imputer=imputer,
            hundred_seeds=hundred_seeds,
            label_scale=label_scalen
        )

    if scip_no_pseudo:
        run_regression_pipeline(
            data_name='scip_no_pseudo',
            data_path=f'/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/cleaned_scip/scip_no_pseudocosts_clean_data.csv',
            feats_path=f'/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/cleaned_scip/scip_no_pseudocosts_clean_feats.csv',
            is_excel=False,
            prefix='scip_no_pseudo',
            treffplusx=treffplusx,
            models=models,
            imputer=imputer,
            hundred_seeds=hundred_seeds,
            label_scale=label_scalen
        )

    if fico:
        run_regression_pipeline(
            data_name='fico',
            data_path='/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/fico/clean_data_final_06_03.xlsx',
            feats_path='/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/fico/base_feats_no_cmp_918_24_01.xlsx',
            is_excel=True,
            prefix='fico',
            treffplusx=treffplusx,
            models=models,
            imputer=imputer,
            hundred_seeds=hundred_seeds,
            label_scale=label_scalen
        )

# main(scip_default=True, scip_no_pseudo=False, fico=False, treffplusx=treffplustage)
# main(scip_default=False, scip_no_pseudo=True, fico=True, treffplusx=treffplustage)
# main(scip_default=True, scip_no_pseudo=True, fico=False, treffplusx=treffplustage')
# main(scip_default=True, scip_no_pseudo=True, fico=True, treffplusx=treffplustage, label_scalen=False)
# main(scip_default=True, scip_no_pseudo=True, fico=True, treffplusx=treffplustage, label_scalen=True)
