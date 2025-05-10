import time
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer

def create_directory(parent_name):
    base_path = f'/Users/fritz/Downloads/ZIB/Master/Treffen/{parent_name}'
    subdirs = ['Prediction', 'Accuracy', 'Importance', 'RunTime']
    for subdir in subdirs:
        os.makedirs(os.path.join(base_path, subdir), exist_ok=True)
    return 0

def shifted_geometric_mean(values, shift):
    values = np.array(values)
    # Shift the values by the constant and check for any negative values after shifting
    shifted_values = values + shift
    if shifted_values.dtype == 'object':
        # Attempt to convert to float
        shifted_values = shifted_values.astype(float)

    shifted_values_log = np.log(shifted_values)  # Step 1: Log of each element in shifted_values
    log_mean = np.mean(shifted_values_log)  # Step 2: Compute the mean of the log values
    geo_mean = np.exp(log_mean) - shift
    geo_mean = np.round(geo_mean, 2)
    return geo_mean

def get_features_label(data_frame, feature_df, chosen_features):
    features = feature_df[chosen_features]
    label = data_frame['Cmp Final solution time (cumulative)']
    return features, label

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
    sgm_predicted = shifted_geometric_mean(predicted_time, shift)
    sgm_mixed = shifted_geometric_mean(data['Final solution time (cumulative) Mixed'].loc[indices], shift)
    sgm_int = shifted_geometric_mean(data['Final solution time (cumulative) Int'].loc[indices], shift)
    sgm_vbs = shifted_geometric_mean(data['Virtual Best'].loc[indices], shift)
    return sgm_predicted, sgm_mixed, sgm_int, sgm_vbs

def get_run_time_row(prediction, data, model_name, imputation, scaler, shift):
    sgm_pred, sgm_mixed, sgm_int, sgm_vbs = get_predicted_run_time_sgm(prediction, data, shift=50)
    new_row = pd.DataFrame([{'Model': model_name, 'Imputation': imputation,
                   'Scaler': scaler, 'SGM Mixed': sgm_mixed,
                   'SGM Int': sgm_int, 'SGM Prediction': sgm_pred,
                   'SGM VBS': sgm_vbs}])
    return new_row

def get_acc_row(pred, test, model_name, imputation, scaler, extreme_threshold):
    total_accuracy, extreme_accuracy, number_ex_instances = get_accuracy(pred, test, extreme_threshold)
    acc_df =  pd.DataFrame([{'Model': model_name, 'Imputation': imputation, 'Scaler': scaler,
                                                      'Accuracy': total_accuracy, 'Extreme Accuracy': extreme_accuracy,
                                                      'Number of extreme instances': number_ex_instances}])
    return acc_df

def get_importance_col(importances, feature_names, model_name, imputation, scaler):
    importance_df = pd.DataFrame({f'{model_name}_{imputation}_{scaler}': importances}, index=feature_names)
    return importance_df

def get_prediction_df(dictionary):
    max_len = max(len(lst) for lst in dictionary.values())
    # Pad each list with zeros
    for key in dictionary:
        dictionary[key] += [0] * (max_len - len(dictionary[key]))
    return pd.DataFrame.from_dict(dictionary, orient='columns')

def trainer(imputation, scaler, model, model_name, X_train, y_train, seed):

    # Build pipeline
    steps = []
    # Add imputation
    steps.append(('imputer', SimpleImputer(strategy=imputation)))
    # Add scaling if applicable
    if scaler:
        steps.append(('scaler', scaler))
    # Add model
    steps.append(('model', model))
    # Create pipeline
    pipeline = Pipeline(steps)
    # Update model-specific parameters
    if model_name == "RandomForest":
        model.random_state = seed
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    return pipeline

def predict(pipeline, X_test, y_test):
    # Evaluate on the test set
    relevant_indices = y_test[y_test != 0].index
    y_test_relevant = y_test.loc[relevant_indices]
    y_pred_relevant = pipeline.predict(X_test.loc[relevant_indices, :])
    y_pred_relevant = pd.Series(y_pred_relevant, index=relevant_indices, name='Prediction')
    return y_pred_relevant, y_test_relevant

def regression(data, features_df, feature_names, models, scalers, imputer, random_seeds, extreme_threshold=4.0):
    """
    Gets a csv file as input
    trains a ml model
    outputs csv files: Accuracy, Time save/loss, Feature Importance
    """
    start_time = time.time()
    features, label = get_features_label(data, features_df, feature_names)
    accuracy_df = pd.DataFrame()
    run_time_df = pd.DataFrame()
    feature_importance_df = pd.DataFrame(index=feature_names)
    prediction_dictionary = {}

    for model_name, model in models.items():
        if model_name not in ['LinearRegression', 'RandomForest']:
            print('AHHHHHHHHHHHHHHHHHHHHHHHH')
            break
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
                    new_acc_row = get_acc_row(y_pred_relevant, y_test_relevant, model_name, imputation, scaler, extreme_threshold)
                    accuracy_df = pd.concat([accuracy_df, new_acc_row])
                    # add sgm of run time for this setting to run_time_df
                    new_run_time_row = get_run_time_row(y_pred_relevant, data, model_name, imputation, scaler, shift=50)
                    run_time_df = pd.concat([run_time_df, new_run_time_row])
                    # return actual prediction
                    prediction_dictionary[model_name+'_'+imputation+'_'+str(scaler)+'_'+str(seed)] = y_pred_relevant.to_list()
                    # feature importance

                    if model_name == 'LinearRegression':
                        importances = trained_model.named_steps['model'].coef_
                    else:
                        importances = trained_model.named_steps['model'].feature_importances_
                    # TODO: Mach das als dictionary, weil is slow af
                    new_importance_col = get_importance_col(importances, feature_names, model_name, imputation, scaler)
                    feature_importance_df = pd.concat([feature_importance_df, new_importance_col], axis=1)

    prediction_df = get_prediction_df(prediction_dictionary)
    end_time = time.time()
    print(f'Final time: {end_time - start_time}')
    return accuracy_df, run_time_df, prediction_df, feature_importance_df

def main(scip_default=False, scip_no_pseudo=False, fico=False, treffplusx='Wurm'):

    models = {'LinearRegression': LinearRegression(), 'RandomForest': RandomForestRegressor(n_estimators=100, random_state=0)}
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
    # create directory for current run
    create_directory(f'{treffplusx}')
    # TODO make the next part cleaner
    if scip_default:
        # call regression
        scip_data = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_default_clean_data.csv')
        features_scip = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_default_clean_feats.csv')
        scalers = [StandardScaler(), MinMaxScaler(), RobustScaler(), PowerTransformer(method='yeo-johnson'),
                   QuantileTransformer(output_distribution='normal', n_quantiles=int(len(scip_data) * 0.8))]
        scip_acc_df, scip_sgm_runtime, prediction_df, feature_importance_df = regression(scip_data, features_scip,
                                                                                         features_scip.columns, models,
                                                                                         scalers, imputer, hundred_seeds)
        # to csv
        scip_acc_df.to_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffplusx}/Accuracy/scip_acc_df.csv', index=False)
        scip_sgm_runtime.to_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffplusx}/RunTime/scip_sgm_runtime.csv', index=False)
        prediction_df.to_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffplusx}/Prediction/scip_prediction_df.csv')
        feature_importance_df.to_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffplusx}/Importance/scip_importance_df.csv')

    if fico:
        # call regression
        fico_data = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/918/clean_data_final_06_03.xlsx')
        features_fico = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/918/base_feats_no_cmp_918_24_01.xlsx').iloc[:,1:]

        scalers = [StandardScaler(), MinMaxScaler(), RobustScaler(), PowerTransformer(method='yeo-johnson'),
                   QuantileTransformer(output_distribution='normal', n_quantiles=int(len(fico_data) * 0.8))]
        fico_acc_df, fico_sgm_runtime, prediction_df, feature_importance_df = regression(fico_data, features_fico,
                                                                                         features_fico.columns, models,
                                                                                         scalers, imputer, hundred_seeds)
        # to csv
        fico_acc_df.to_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffplusx}/Accuracy/fico_acc_df.csv', index=False)
        fico_sgm_runtime.to_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffplusx}/RunTime/fico_sgm_runtime.csv', index=False)
        prediction_df.to_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffplusx}/Prediction/fico_prediction_df.csv')
        feature_importance_df.to_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffplusx}/Importance/fico_importance_df.csv')

    if scip_no_pseudo:
        # call regression
        scip_data = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_no_pseudocosts_clean_data.csv')
        features_scip = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_no_pseudocosts_clean_feats.csv')
        scalers = [StandardScaler(), MinMaxScaler(), RobustScaler(), PowerTransformer(method='yeo-johnson'),
                   QuantileTransformer(output_distribution='normal', n_quantiles=int(len(scip_data) * 0.8))]
        scip_acc_df, scip_sgm_runtime, prediction_df, feature_importance_df = regression(scip_data, features_scip,
                                                                                         features_scip.columns, models,
                                                                                        scalers, imputer, hundred_seeds)
        # to csv
        scip_acc_df.to_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffplusx}/Accuracy/scip_no_pseudo_acc_df.csv', index=False)
        scip_sgm_runtime.to_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffplusx}/RunTime/scip_no_pseudo_sgm_runtime.csv',
                                index=False)
        prediction_df.to_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffplusx}/Prediction/scip_no_pseudo_prediction_df.csv')
        feature_importance_df.to_csv(
            f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffplusx}/Importance/scip_no_pseudo_importance_df.csv')

# main(scip_default=True, fico=False, treffplusx='TreffMasDos')
# main(scip_default=False, fico=True, treffplusx='TreffMasDos')
# main(scip_default=True, scip_no_pseudo=True, fico=True, treffplusx='TreffenMasDos')
# main(scip_default=True, scip_no_pseudo=True, fico=False, treffplusx='TreffenMasDos')
# main(scip_default=True, scip_no_pseudo=True, fico=False, treffplusx='TreffenMasDos')

