import pandas as pd
import numpy as np
import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# general functions
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
    return geo_mean

# RUNTIME
def get_predicted_run_time_sgm(y_pred, data, shift, default_rule:str):
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
    sgms = [sgm_predicted, sgm_mixed, sgm_int, sgm_vbs]
    if default_rule == "Mixed":
        default = sgm_mixed
    elif default_rule == "Int":
        default = sgm_int
    else:
        print(f"Default rule {default_rule} not supported.")
        sys.exit(1)
    relative_to_default = [value/default for value in sgms]
    return relative_to_default

# FEATURE IMPORTANCE
# TODO: adapt to swapped columns and rows
def ranking_feature_importance(importance_df, feature_names):

    ranking_df = pd.DataFrame(index=feature_names, columns=['Feature', 'Linear Score', 'Forest Score'])

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

    ranking_df['Feature'] = feature_names
    ranking_df['Linear Score'] = lin_scores.sum(axis=1)
    ranking_df['Forest Score'] = for_scores.sum(axis=1)
    ranking_df['Combined'] = ranking_df['Linear Score']+ranking_df['Forest Score']
    ranking_df.sort_values(by=['Combined'], ascending=True, inplace=True)

    return ranking_df

def get_feature_importances(fitted_model, feature_names):
    # feature importance
    if isinstance(fitted_model, RandomForestRegressor):
        feature_importance = fitted_model.feature_importances_
    elif isinstance(fitted_model, LinearRegression):
        feature_importance = fitted_model.coef_
    else:
        print(f"{fitted_model} is not a valid model!")
        sys.exit(1)
    # Todo: Check if pd.Series(feature_importance, index=feature_names) is better
    return feature_importance

# ACCURACY
def threshold_accuracy(y_pred, y_true, relevant_threshold):
    # Filter for labels larger threshold
    relevant_indices = abs(y_true) >= relevant_threshold
    y_true_relevant = y_true[relevant_indices]
    number_relevant_instances = (len(y_true_relevant), len(y_true))
    y_pred_relevant = y_pred[relevant_indices]

    # Calculate percentage of correctly predicted signs
    number_correct_predictions = np.sum(np.sign(y_true_relevant) == np.sign(y_pred_relevant))
    accuracy_threshold = (number_correct_predictions/number_relevant_instances[0])*100 \
        if number_relevant_instances[0] > 0 else np.nan
    accuracy_threshold = np.round(accuracy_threshold, decimals=2)
    return accuracy_threshold, number_relevant_instances

def get_accuracy(y_pred, y_true, mid_threshold, extreme_threshold):
    nonzero_indices = y_true != 0
    y_true_nonzero = y_true[nonzero_indices]
    y_pred_nonzero = y_pred[nonzero_indices]

    overall_acc, number_instances = threshold_accuracy(y_pred_nonzero, y_true_nonzero, 0)
    mid_acc, number_mid_instances = threshold_accuracy(y_pred_nonzero, y_true_nonzero, mid_threshold)
    extreme_acc, number_extreme_instances = threshold_accuracy(y_pred_nonzero, y_true_nonzero, extreme_threshold)

    return overall_acc, mid_acc, number_mid_instances, extreme_acc, number_extreme_instances

# SHARES
def get_shares(prediction, label):
    actual_mixed = len(label[label>0])
    predicted_mixed = len(prediction[prediction>0])

    actual_int = len(label[label<0])
    predicted_int = len(prediction[prediction<0])

    total_actual = actual_mixed + actual_int
    total_predicted = predicted_mixed + predicted_int

    actual_shares_mixed_int = [np.round(actual_mixed/total_actual, 2), np.round(actual_int/total_actual, 2)]
    predicted_shares_mixed_int = [np.round(predicted_mixed/total_predicted, 2), np.round(predicted_int/total_predicted, 2)]

    return actual_shares_mixed_int, predicted_shares_mixed_int
