import numpy as np
import pandas as pd

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
        accuracy_in_threshold = number_correct_predictions / len(y_test_relevant) * 100 if len(
            y_test_relevant) > 0 else np.nan
        return accuracy_in_threshold, number_relevant_instances


    overall_acc = overall_accuracy()
    mid_acc, number_mid_instances = threshold_accuracy(mid_threshold)
    extreme_acc, number_extreme_instances = threshold_accuracy(extreme_threshold)


    return overall_acc, mid_acc, number_mid_instances, extreme_acc, number_extreme_instances



acc_test = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/JulyTry/Runs/Iteration1/FICO/ScaledLabel/Accuracy/fico_acc_df.csv')
acc_train = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/JulyTry/Runs/Iteration1/FICO/ScaledLabel/Accuracy/fico_acc_trainset.csv')
for col in acc_train.columns:
    print(type(acc_train[col].values[0]))