import numpy as np
import pandas as pd
from pandas import Series
import time
from datetime import datetime
# scikitlearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# plotting
from visualize_erfolg import shifted_geometric_mean

# Get the current date
current_date = datetime.now()
# Format it as a string
date_string = current_date.strftime("%d_%m")

def read_data():
    data = pd.read_excel(f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/NoCmpFeats/base_data_24_01.xlsx').drop(columns='Matrix Name')
    feats = pd.read_excel(f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/NoCmpFeats/base_feats_no_cmp_24_01.xlsx')
    # feats = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/NoCmpFeats/max_scaling/yeo_johnson.xlsx')
    # feats = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/NoCmpFeats/max_scaling/quantile.xlsx')

    # feats = feats/feats.max()
    label = data['Cmp Final solution time (cumulative)']
    return data, feats, label

def kick_outlier(feat_df, label_series, threshold : int):
    indices_to_keep = label_series[label_series.abs() <= threshold].index
    feats_to_keep = feat_df.loc[indices_to_keep, :]
    labels_to_keep = label_series.loc[indices_to_keep]
    return feats_to_keep, labels_to_keep

def scale_label(label):
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

def get_info(values:Series):

    print('Shape: ', values.shape)
    print('Min: ', values.min())
    print('Max: ', values.max())
    print('Mean: ', values.mean())
    print('Std: ', values.std())
    print('Median: ', values.median())
    print('Variance: ', values.var())
    print('Negative r2: ', len(values[values<0]))

def predicted_time(time_df, prediction_df):
    pred_index = prediction_df.index
    pred_df = time_df.loc[pred_index,:].copy()

    pred_df['Predicted Time'] = pred_df['Virtual Best']

    for index, row in prediction_df.iterrows():
        if row['Right or Wrong'] != 1:
            pred_df.loc[index, 'Predicted Time'] = max(pred_df.loc[index, 'Final solution time (cumulative) Mixed'],
                                                       pred_df.loc[index, 'Final solution time (cumulative) Int'])
    pred_df['Time Save/Loss'] = abs(pred_df['Predicted Time'] - pred_df['Virtual Best'])

    mixed_mean = shifted_geometric_mean(pred_df['Final solution time (cumulative) Mixed'], 0.5)
    predicted_time_mean = shifted_geometric_mean(pred_df['Predicted Time'], 0.5)
    vbs_mean = shifted_geometric_mean(pred_df['Virtual Best'], 0.5)
    relative_to_mixed_mean = [1.0, predicted_time_mean/mixed_mean, vbs_mean/mixed_mean]
    return relative_to_mixed_mean

def regression(full_data, features, label, scalers, imputations, models, random_seeds, extreme_threshold=4.0):

    results = []

    # Initialize dictionaries to collect feature importance data
    linear_feature_importance_data = {}
    forest_feature_importance_data = {}

    linear_feature_importance_df = pd.DataFrame({'Feature': features.columns})
    forest_feature_importance_df = linear_feature_importance_df.copy()

    time_mixed_int_vbs = full_data[['Final solution time (cumulative) Mixed', 'Final solution time (cumulative) Int', 'Virtual Best']].copy()
    columns_for_collected_sgm = {}

    count=0
    start_time = time.time()
    for model_name, model in models.items():
        for imputation in imputations:
            for scaler in scalers:
                for seed in random_seeds:
                    count+=1
                    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2,random_state=seed)
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
                    # Evaluate on the test set
                    relevant_indices = y_test[y_test != 0].index
                    y_test_relevant = y_test.loc[relevant_indices]

                    y_pred_relevant = pipeline.predict(X_test.loc[relevant_indices, :])
                    # get the accuracy of predicted signs
                    total_accuracy, extreme_accuracy, number_ex_instances = get_accuracy(y_pred_relevant, y_test.loc[relevant_indices], extreme_threshold)
                    # sgm of predicted time
                    pred_df = pd.DataFrame({'Prediction': y_pred_relevant, 'Actual': y_test_relevant},
                                           index=relevant_indices)
                    pred_df['Right or Wrong'] = (np.sign(pred_df['Prediction']) == np.sign(pred_df['Actual'])).astype(int)
                    # add column containing the absolute difference in prediction and actual
                    pred_df['Abs Time Diff'] = abs(pred_df['Prediction'] - pred_df['Actual'])

                    # feature importance
                    if model_name == "LinearRegression":
                        coefficients = model.coef_
                        # linear_feature_importance_df[str(model_name)+str(imputation)+str(scaler)+str(count)] = coefficients
                        linear_feature_importance_data[f"{model_name}_{imputation}_{scaler}_{count}"] = coefficients
                        mean_to_mixed = predicted_time(time_mixed_int_vbs, pred_df)
                        columns_for_collected_sgm['Linear: ' + str(count)] = mean_to_mixed

                    elif model_name == "RandomForest":
                        importance = model.feature_importances_
                        # Collect data in the dictionary
                        forest_feature_importance_data[f"{model_name}_{imputation}_{scaler}_{count}"] = importance
                        # forest_feature_importance_df[str(model_name)+str(imputation)+str(scaler)+str(count)] = importance
                        mean_to_mixed = predicted_time(time_mixed_int_vbs, pred_df)
                        columns_for_collected_sgm['Forest: ' + str(count)] = mean_to_mixed

                    results.append({
                        "Model": model_name,
                        "Imputation": imputation,
                        "Scaling": scaler.__class__.__name__ if scaler else None,
                        "Accuracy": total_accuracy,
                        "Extreme Accuracy": extreme_accuracy,
                        "Number Extreme Instances": number_ex_instances
                    })

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    # Convert the dictionaries into DataFrames after the loop
    linear_feature_importance_df = pd.concat([linear_feature_importance_df, pd.DataFrame.from_dict(linear_feature_importance_data, orient='columns')], axis=1)
    forest_feature_importance_df = pd.concat([forest_feature_importance_df, pd.DataFrame.from_dict(forest_feature_importance_data, orient='columns')], axis=1)
    collected_sgm_df = pd.DataFrame({'TimeShiftedGeoMean': ['Mixed', 'Predicted', 'Virtual Best']})
    sgm_run_time_df = pd.concat([collected_sgm_df, pd.DataFrame(columns_for_collected_sgm)], axis=1)
    print(f"Execution time: {elapsed_time:.2f} seconds")
    return results_df, linear_feature_importance_df, forest_feature_importance_df, sgm_run_time_df

def regress_on_different_sets_based_on_label_magnitude(number_of_seeds:str, regressors, scalers, imputations, feature_names,
                                                       outlier_threshold=1000,  log_label=False, to_excel=False, sgm=False,
                                                       directory_for_excels='/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/NoCmpFeats/Tester/'):
    d, feat, target = read_data()
    feat = feat[feature_names]
    if len(regressors) == 2:
        regressor_names_for_excel = 'both'
    elif len(regressors) == 1:
        regressor_names_for_excel = next(iter(regressors.keys()))
    else:
        print('Either no or too many regressors selected')
        return 1

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
    one_seed = hundred_seeds[46:47]
    ten_seeds = [1024137296, 4024648401, 2912980295, 568995048, 362704412, 1684864875, 1282240360, 829894663,
                 1341861657,
                 3626048517]
    twenty_seeds = [1507732364, 666405231, 1024137296, 4218641677, 1684864875, 362704412, 4013370079, 3143265894,
                    2324915710, 1387819803, 3118826909, 1341861657, 1210988828, 2270465462, 1640670982, 537517556,
                    2237947797, 2942245439, 882010483, 744281271]
    fifty_seeds = [563163990, 3495225726, 1684864875, 263097927, 829894663, 958203997, 1396620602, 4218641677,
                   3308693507,
                   362704412, 738323230, 537517556, 3049281880, 2093310729, 1784663289, 4052173232, 1280346069,
                   1210988828, 2207168494, 1174739769, 429011752, 1693665289, 698121968, 4033267948, 325283101,
                   744281271, 1417846724, 2478344316, 2033245880, 3118826909, 1203175353, 1024137296, 1665201999,
                   1891799436, 3691808733, 2872252636, 3094311947, 1387819803, 289901203, 2112194978, 1023587314,
                   1341861657, 923999093, 2942245439, 3898118442, 1023133507, 572356937, 1398432260, 3925818254,
                   2912980295]
    seed_dict = {'one': one_seed, 'ten': ten_seeds, 'twenty': twenty_seeds, 'fifty': fifty_seeds,
                 'hundred': hundred_seeds}

    if log_label:
        feat_below_threshold, target_below_threshold = kick_outlier(feat, target, outlier_threshold)
        target_below_threshold = scale_label(target_below_threshold)
        result_below_threshold_df, linear_importance, forest_importance, sgm_df = regression(d, feat_below_threshold, target_below_threshold, scalers, imputations, regressors, random_seeds=seed_dict[number_of_seeds], extreme_threshold=1.6)
        if sgm:
            sgm_df.to_excel(directory_for_excels+f'/Logged/SGM/sgm_logged_t{len(feature_names)}_{regressor_names_for_excel}_below_{outlier_threshold}_{number_of_seeds}_seeds_{len(scalers)}_{date_string}.xlsx', index=False)

        if to_excel:
            result_below_threshold_df.to_excel(
                directory_for_excels + f'/Logged/Accuracy/logged_t{len(feature_names)}_{regressor_names_for_excel}_below_{outlier_threshold}_{number_of_seeds}_seeds_{len(scalers)}_{date_string}.xlsx',
                index=False)
            linear_importance.to_excel(
                directory_for_excels + f'/Logged/Importance/Linear/logged_lin_impo_t{len(feature_names)}_below_{outlier_threshold}_{number_of_seeds}_seeds_{len(scalers)}_{date_string}.xlsx',
                index=False)
            forest_importance.to_excel(
                directory_for_excels + f'/Logged/Importance/Forest/logged_forest_impo_t{len(feature_names)}_below_{outlier_threshold}_{number_of_seeds}_seeds_{len(scalers)}_{date_string}.xlsx',
                index=False)

    else:
        feat_below_threshold, target_below_threshold = kick_outlier(feat, target, outlier_threshold)
        result_below_threshold_df, linear_importance, forest_importance, sgm_df = regression(d, feat_below_threshold, target_below_threshold, scalers, imputations, regressors, random_seeds=seed_dict[number_of_seeds])
        if sgm:
            # sgm_unscaled.to_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/PräsiTristan/Präsi/CSV/t3_unscaled_sgm.xlsx', index=False)
            sgm_df.to_excel(directory_for_excels+'/Unscaled/SGM/sgm_unscaled'+f'_t{len(feature_names)}_{regressor_names_for_excel}_below_{outlier_threshold}_{number_of_seeds}_seeds_{len(scalers)}_{date_string}.xlsx', index=False)

        if to_excel:
            result_below_threshold_df.to_excel(
                directory_for_excels+f'/Unscaled/Accuracy/unscaled_t{len(feature_names)}_{regressor_names_for_excel}_below_{outlier_threshold}_{number_of_seeds}_seeds_{len(scalers)}_{date_string}.xlsx',
                index=False)
            linear_importance.to_excel(
                directory_for_excels+f'/Unscaled/Importance/Linear/unscaled_lin_impo_t{len(feature_names)}_below_{outlier_threshold}_{number_of_seeds}_seeds_{len(scalers)}_{date_string}.xlsx',
                index=False)
            forest_importance.to_excel(
                directory_for_excels+f'/Unscaled/Importance/Forest/unscaled_forest_impo_t{len(feature_names)}_below_{outlier_threshold}_{number_of_seeds}_seeds_{len(scalers)}_{date_string}.xlsx',
                index=False)


imputators = ['constant', 'median', 'mean']
scaling = [PowerTransformer('yeo-johnson'),
           QuantileTransformer(n_quantiles=100,output_distribution="normal",random_state=42)]
regression_models = {"LinearRegression": LinearRegression(),
                     "RandomForest": RandomForestRegressor(n_estimators=100, random_state=729154)}

all_features = ['Matrix Equality Constraints', 'Matrix Quadratic Elements',
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

integer_feature = ['Matrix Equality Constraints', 'Matrix Quadratic Elements',
       'Matrix NLP Formula', 'Presolve Columns', 'Presolve Global Entities',
       '#nodes in DAG', '#integer violations at root',
       '#nonlinear violations at root', '#spatial branching entities fixed (at the root) Mixed']

float_feature = ['% vars in DAG (out of all vars)',
       '% vars in DAG unbounded (out of vars in DAG)',
       '% vars in DAG integer (out of vars in DAG)',
       '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)',
       'Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
       'Avg ticks for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
       'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Mixed',
       'Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
       'Avg coefficient spread for convexification cuts Mixed']

t3_feats_combined = ['Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
            '#nodes in DAG',
            'Avg ticks for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed']

t6_linear = ['Avg ticks for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
             '#nodes in DAG',
             '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)',
             '% vars in DAG integer (out of vars in DAG)',
             'Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
             'Presolve Columns']

t4_linear = ['Avg ticks for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
             '#nodes in DAG',
             '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)',
             '% vars in DAG integer (out of vars in DAG)']

t1_forest = ['#integer violations at root']
t3_forest = ['#integer violations at root',
             'Presolve Global Entities',
             'Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed']


df, feat, label  = read_data()

preset_already_scaled = ['hundred', regression_models, [None], imputators, all_features, 1000]

preset_everything = ['hundred', regression_models, scaling, imputators, all_features, 1000]

preset_only_scaled = ['hundred', regression_models, [PowerTransformer('yeo-johnson'),
                      QuantileTransformer(n_quantiles=100,output_distribution="normal",random_state=42)],
                      imputators, all_features, 1000]

preset_only_quantile = ['hundred', regression_models,
                        [QuantileTransformer(n_quantiles=100,output_distribution="normal",random_state=42)],
                        imputators, all_features, 1000]

preset_combined_t3 = ['hundred', regression_models, scaling, imputators, t3_feats_combined, 1000]



regress_on_different_sets_based_on_label_magnitude(preset_only_scaled[0], preset_only_scaled[1], preset_only_scaled[2],
                                                   preset_only_scaled[3], preset_only_scaled[4], preset_only_scaled[5],
                                                   directory_for_excels='/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra',
                                                   log_label=True, to_excel=True, sgm=True)
regress_on_different_sets_based_on_label_magnitude(preset_only_scaled[0], preset_only_scaled[1], preset_only_scaled[2],
                                                   preset_only_scaled[3], preset_only_scaled[4], preset_only_scaled[5],
                                                   directory_for_excels='/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra',
                                                   log_label=False, to_excel=True, sgm=True)
