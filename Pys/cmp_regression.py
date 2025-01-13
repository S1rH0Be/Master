#general
from datetime import datetime
import pandas as pd
import numpy as np
from hyperframe.frame import DataFrame
from setuptools.command.rotate import rotate
# regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# visualize
import matplotlib.pyplot as plt
from visualize_erfolg import shifted_geometric_mean
from bar_plot_accuracy import by_intervall
from create_and_scale_cmp_df import feature_df

# Get the current date
current_date = datetime.now()
# Format it as a string
date_string = current_date.strftime("%d_%m")

def accuracy(prediction_df):
    # instances where one rule is by a factor of more then 0.5 faster are considered "extreme"
    extreme_cases_df = prediction_df[np.abs(prediction_df['Actual']) >= 0.5]
    mid_cases_df = prediction_df.loc[~prediction_df.index.isin(extreme_cases_df.index)]
    # number of cases
    number_of_relevant_cases = len(prediction_df.index)
    number_of_mid_cases = len(mid_cases_df)
    number_of_extreme_cases = len(extreme_cases_df)
    # accuracy of model differentiated by cases
    total_accuracy = np.round((prediction_df['Right or Wrong'].sum() / number_of_relevant_cases) * 100, 2)
    mid_accuracy = np.round((mid_cases_df['Right or Wrong'].sum() / number_of_mid_cases) * 100, 2)
    extreme_accuracy = np.round((extreme_cases_df['Right or Wrong'].sum() / number_of_extreme_cases) * 100, 2)
    # create a df which contains the accuracies of the different intervalls
    acc_df = pd.DataFrame({'Intervall': ['Complete', '(0,0.5)', '[0.5, inf)'],
                           'Accuracy': [total_accuracy, mid_accuracy, extreme_accuracy]})
    return acc_df

def bar_plot(df):
    ax = df.plot(kind='bar', legend=False)
    plt.xticks(ticks=range(len(df)), labels=df['Intervall'], rotation=0)
    # Add labels and title
    plt.ylabel('Values')
    plt.title('Bar Plot with Index as Custom x-tick Labels')
    plt.show()
    plt.close()

def forrest_regression(features:DataFrame, label:DataFrame, show_accuracy=False):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)
    # only instances with actual label of unequal 0 are from interest
    relevant_indices = y_test['Cmp Final solution time (cumulative)'][
        y_test['Cmp Final solution time (cumulative)'] != 0.0].index
    x_test_relevant = x_test.loc[relevant_indices, :]
    y_test_relevant = y_test['Cmp Final solution time (cumulative)'].loc[relevant_indices]

    # fit model and predict label
    rf.fit(x_train, y_train.values.ravel())
    y_pred_relevant = rf.predict(x_test_relevant)
    # create df which contains the predicted and actual values and if the sign is correct
    pred_df = pd.DataFrame({'Prediction': y_pred_relevant, 'Actual': y_test_relevant}, index=relevant_indices)
    pred_df['Right or Wrong'] = (np.sign(pred_df['Prediction']) == np.sign(pred_df['Actual'])).astype(int)
    # add column containing the absolute difference in prediction and actual
    pred_df['Abs Time Diff'] = abs(pred_df['Prediction'] - pred_df['Actual'])
    # add column to check by what factor the prediction is off
    pred_df['Factor'] = pred_df['Prediction'] / pred_df['Actual']

    accuracy_df = accuracy(pred_df)
    if show_accuracy:
        bar_plot(accuracy_df)

    # Evaluate the model
    mse = mean_squared_error(y_test_relevant, y_pred_relevant)
    r2 = r2_score(y_test_relevant, y_pred_relevant)

    return pred_df, accuracy_df, mse, r2

cmp_feature_df = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/ZwischenPräsi_Januar/Data/Features/cmp_features_07_01.xlsx')
label_series = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/ZwischenPräsi_Januar/Data/Features/label_series_07_01.xlsx')

p, a, m, r = forrest_regression(cmp_feature_df, label_series)

# p.to_excel(f'/Users/fritz/Downloads/ZIB/Master/ZwischenPräsi_Januar/Predictions/Cmp/cmp_prediction_df_{date_string}.xlsx',
#            index=False)

print(shifted_geometric_mean(p['Abs Time Diff'], 0.5))
print(shifted_geometric_mean(abs(p['Factor']), 0.5))

"""
Next:
1. If abs diff <= 1%, then factor shouldnt matter
2. Calculate shifted geometric mean of abs diff and factor
3. Do the same with linreg
"""