
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DEBUG=True

# TODO:
# Stats i need:
        # 1. Accuracy
        # 2. Feature Importance
        # 3. Shares
        # 4. run time sgm
# Information in these stats:
# 1. Model
# 2. Scaling
# 3. Imputation
# 4. Data Set




def create_directory(path:str):
    os.makedirs(os.path.join(path), exist_ok=True)
    subdirs = ['Prediction', 'Accuracy', 'Importance', 'SGM']
    for subdir in subdirs:
        os.makedirs(os.path.join(path, subdir), exist_ok=True)
    return 0

def plot_histo(data, number_bins, title):

    plt.hist(data, bins=number_bins, color='purple', alpha=1)
    plt.title(title)

    # Adjust layout
    plt.tight_layout()
    # Show the plots once all are created
    plt.show()
    # Close the plot to free up memory
    plt.close()

def get_stats(prediction:pd.Series, label:pd.Series, data:pd.DataFrame, mid_threshold=0.1, extreme_threshold=4):

    def get_accuracy(prediction_series, labels, mid_thresh=0.1, extreme_thresh=4):
        # Filter for nonzero labels
        nonzero_indices = label != 0
        actual_nonzero = labels[nonzero_indices]
        pred_nonzero = prediction_series[nonzero_indices]

        def complete_accuracy(predicted_labels, actual_labels):
            # Calculate percentage of correctly predicted signs
            correct_signs = np.sum(np.sign(actual_labels) == np.sign(predicted_labels))
            percent_correct_signs = correct_signs / len(actual_labels) * 100 if len(actual_labels) > 0 else np.nan
            return percent_correct_signs
        # TODO: maybe range for instances with label between 0.1 and extreme_threshold
        # TODO: Frage: Why is mid acc worse than overall acc
        def threshold_accuracy(predicted, actual, threshold):
            # Filter for threshold labels
            threshold_indices = abs(actual) >= threshold

            y_test_threshold = actual[threshold_indices]
            number_threshold_signs = (len(y_test_threshold), len(actual)) # number of instances >= threshold
            y_pred_threshold = predicted[threshold_indices]

            # Calculate percentage of correctly predicted signs
            correct_threshold_signs = np.sum(np.sign(y_test_threshold) == np.sign(y_pred_threshold))
            percent_correct_threshold_signs = correct_threshold_signs / len(y_test_threshold) * 100 if len(
                y_test_threshold) > 0 else np.nan
            return percent_correct_threshold_signs, number_threshold_signs

        overall_accuracy = complete_accuracy(pred_nonzero, actual_nonzero)
        mid_accuracy, number_mid_instances = threshold_accuracy(pred_nonzero, actual_nonzero, threshold=mid_thresh)
        extreme_accuracy, number_extreme_instances = threshold_accuracy(pred_nonzero, actual_nonzero, threshold=extreme_thresh)

        accuracy_dict = {'Overall Accuracy': overall_accuracy, 'Mid Accuracy': (mid_accuracy, number_mid_instances),
                         'Extreme Accuracy': (extreme_accuracy, number_extreme_instances)}

        return accuracy_dict

    def get_feature_importance():
        pass

    def get_feature_score(feature_importance):
        pass

    def get_run_time_sgm(predicted_labels, time_df):
        """
            prediction:
            time_df: columns=[Mixed Time, Int Time, VBS]
        """
        pass

    def get_shares_of_rules(predicted_labels, actual_labels):
        """
        Returns shares of mixed and int according to the prediction and the actual label on test set
        """
        predicted_mixed_count = 0
        predicted_int_count = 0
        predicted_zero_count = 0
        actual_mixed_count = 0
        actual_int_count = 0
        actual_zero_count = 0

        for i in predicted_labels:
            if i < 0:
                predicted_int_count += 1
            elif i > 0:
                predicted_mixed_count += 1
            else:
                predicted_zero_count += 1

        for i in actual_labels:
            if i < 0:
                actual_int_count += 1
            elif i > 0:
                actual_mixed_count += 1
            else:
                actual_zero_count += 1

        order = ['Mixed', 'Int', 'Zero']
        predicted_shares = [predicted_mixed_count, predicted_int_count, predicted_zero_count]
        actual_shares = [actual_mixed_count, actual_int_count, actual_zero_count]
        return order, predicted_shares, actual_shares

    # TODO:
    accuracy = get_accuracy(prediction_series=prediction, labels=label, mid_thresh=mid_threshold, extreme_thresh=extreme_threshold)
    # TODO:
    feat_impo = get_feature_importance()
    # TODO:
    feat_score = get_feature_score(feat_impo)
    # TODO:
    run_time = get_run_time_sgm(prediction, data)
    order, pred_shares, actual_shares = get_shares_of_rules(prediction, label)
    shares = [order, pred_shares, actual_shares]

    return accuracy, feat_impo, feat_score, run_time, shares

# TODO: think about kicking label outlier
def kick_outlier():
    pass

def get_feature_label(df, label_loggen=False):
    # TODO: feature ist platzhalter
    feature = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/fico/base_feats_no_cmp_918_24_01.xlsx')
    label = df.loc[:, 'Cmp Final solution time (cumulative)']
    if label_loggen:
        label_pos = label[label >= 0]
        label_neg = label[label < 0]
        # log1p calculates log(1+x) numerically stable
        label_pos_log = np.log1p(label_pos)/np.log(10.0) # this is equivalent to log10
        label_neg_log = (np.log1p(abs(label_neg))/np.log(10.0)) * -1 # this is equivalent to log10
        label = pd.concat([label_pos_log, label_neg_log]).sort_index()
    return feature, label

def get_predicted_label_df(df, predicted_labels):
    """
        Returns df with columns ['Matrix Name', 'Predicted Label', 'Actual Label']
    """
    label_df = df.copy()
    label_df['Predicted Label'] = predicted_labels
    label_df = label_df.loc[:,['Matrix Name', 'Predicted Label', 'Cmp Final solution time (cumulative)']]
    return label_df

def training(imputation:str, scaler:str, model, model_name:str, training_feature, training_label, random_seed:int):
    # Build pipeline
    # Update model-specific parameters
    if model_name == "RandomForest":
        model.random_state = random_seed
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
    pipeline.fit(training_feature, training_label)
    return pipeline

def make_prediction(model, X_test):
    prediction = model.predict(X_test)
    prediction_series = pd.Series(prediction, index=X_test.index, name='Prediction')
    return prediction_series

def log_the_label(label):
    y_pos = label[label >= 0]
    y_neg = label[label < 0]
    # log1p calculates log(1+x) numerically stable
    y_pos_log = np.log1p(y_pos)/np.log(10.0) # gives log10
    y_neg_log = (np.log1p(abs(y_neg))/np.log(10.0)) * -1 # gives log10
    y_log = pd.concat([y_pos_log, y_neg_log]).sort_index()
    return y_log

def create_stat_csvs(data, prediction_label, test_labels, extreme_threshold, mid_threshold, logged_label=True):
    if logged_label:
        extreme_threshold = np.log(extreme_threshold)
        mid_threshold = np.log(mid_threshold)
    # get stats
    acc_dict, feature_importance, feature_scores, run_time, shares = get_stats(prediction_label, test_labels, data,
                                                                               extreme_threshold, mid_threshold)
    print(acc_dict)
    print(feature_importance)
    print(feature_scores)
    print(run_time)
    print(shares)



def regression(data_set:pd.DataFrame, features=None, model_name=None, model=None, random_seed=None, imputation=None, scaler=None,
               logged_label=True, mid_threshold=0.1, extreme_threshold=4.0):
    features, label = get_feature_label(data_set)
    # TODO: just for debugging
    if DEBUG:
        plot_histo(label, 20, 'Label pre Log')

    if logged_label:
        label = log_the_label(label)

    if DEBUG:
        plot_histo(label, 20, 'Label Post Log')

    X_train, X_test, y_train, y_test = train_test_split(features, label, random_state=random_seed)
    # train the model
    trained_model = training(imputation=imputation, scaler=scaler, model=model, model_name=model_name,
                             training_feature=X_train, training_label=y_train, random_seed=random_seed)
    # use trained model to make predictions
    predicted_labels = make_prediction(model=trained_model, X_test=X_test)

    return predicted_labels, y_test



def main(subdirectory:str, scip=False, fico=False, label_scalen=True,
         feature_scaling=None, prescaled=False, feature_subset=None, models=None, log_label=True,
         mid_threshold=0.1, extreme_threshold=4.0):

    # DAS IST ALLES PLATZHALTER ERSTMAL
    fico_df = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/fico/clean_data_final_06_03.xlsx')
    # start regression
    prediction, test_label = regression(fico_df, model_name='linear', model=LinearRegression(), random_seed=42, imputation='median',
               logged_label=log_label)
    # TODO fico_df platzhalter
    create_stat_csvs(data=fico_df, prediction_label=prediction, test_labels=test_label, logged_label=log_label,
                     extreme_threshold=extreme_threshold, mid_threshold=mid_threshold)

main('Wurm')