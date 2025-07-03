#general
from datetime import datetime
import pandas as pd
import numpy as np
import random
import time
from hyperframe.frame import DataFrame
# regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# visualize
import matplotlib.pyplot as plt
from visualize_erfolg import shifted_geometric_mean

# Get the current date
current_date = datetime.now()
# Format it as a string
date_string = current_date.strftime("%d_%m")

lin_max_min_feat_impo_tupel = []
for_max_min_feat_impo_tupel = []
linear_feature_importance_df = pd.DataFrame({'Feature': ['Matrix Equality Constraints', 'Matrix Quadratic Elements',
       'Matrix NLP Formula', 'Presolve Columns', 'Presolve Global Entities',
       '#nodes in DAG', '#integer violations at root',
       '#nonlinear violations at root', '% vars in DAG (out of all vars)',
       '% vars in DAG unbounded (out of vars in DAG)',
       '% vars in DAG integer (out of vars in DAG)',
       '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)',
       'Avg work for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
       'Avg work for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
       'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Mixed',
       'Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
       'Cmp #spatial branching entities fixed (at the root)',
       'Cmp Avg coefficient spread for convexification cuts']})
forest_feature_importance_df = linear_feature_importance_df.copy()

def read_data(version='16_01'):
    data = pd.read_excel(f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/clean_data_final_{version}.xlsx').drop(columns='Matrix Name')
    feats = pd.read_excel(f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/final_features_{version}.xlsx')
    label = data['Cmp Final solution time (cumulative)']
    return data, feats, label

def bar_plot(df, title : str):
    colors = ['green' if x >= 80 else 'red' if x < 50 else 'blue' for x in df['Accuracy']]

    ax = plt.bar(df['Intervall'], df['Accuracy'], color=colors)
    plt.xticks(ticks=range(len(df)), labels=df['Intervall'], rotation=0)
    # Add labels and title
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.show()
    plt.close()

def feature_histo(df, columns: list, number_bins=10):
    """
    Create histograms for specified columns in a DataFrame, focusing only on values in (0, 1).
    Args:
        df: The DataFrame containing data.
        columns: List of column names to plot histograms for.
        number_bins: Number of bins for the histograms.
    """
    # Create a figure with subplots for each column dynamically
    fig, axs = plt.subplots(len(columns), 1, figsize=(8, 4 * len(columns)))  # n rows, 1 column

    # If there's only one column, axs is not an array, so we handle it separately
    if len(columns) == 1:
        axs = [axs]

    for i, col in enumerate(columns):
        # Filter values in (0, 1)
        filtered_data = df[col]#[(df[col] != 0)]
        # Plot histogram with color distinction
        color = 'red' if 'Mixed' in col else ('magenta' if 'Int' in col else 'orange')
        axs[i].hist(filtered_data, bins=number_bins, color=color, alpha=1)#, label=f'Filtered ({len(filtered_data)} points)')
        axs[i].set_title(f'{col}')

    # Adjust layout
    plt.tight_layout()
    # Show the plots once all are created
    plt.show()
    # Close the plot to free up memory
    plt.close()

def impute(dataframe, columns_to_impute: list, imputation):
    df = dataframe.copy()
    for col in columns_to_impute:
        if col in df.columns:
            if imputation == 'Median':
                # Calculate the median ignoring -1
                median_value = df.loc[df[col] != -1, col].median()
                # Replace -1 with the calculated median
                df[col] = df[col].apply(lambda x: median_value if x == -1 else x)
            elif imputation == 'Mean':
                # Calculate the mean ignoring -1
                mean_value = df.loc[df[col] != -1, col].mean()
                # Replace -1 with the calculated mean
                df[col] = df[col].apply(lambda x: mean_value if x == -1 else x)
            else:
                # if concrete numeric value is given as imputation value  use it
                df[col] = df[col].apply(lambda x: imputation if x == -1 else x)
    return df

def yeo_johnson(df, histo=False):
    # yeo transformer
    pt = PowerTransformer(method='yeo-johnson')
    yeo_data = pt.fit_transform(df)
    yeo_df = pd.DataFrame(yeo_data, columns=df.columns)
    yeo_df_normalized = yeo_df.apply(lambda x: x / abs(x).max())
    if histo:
        feature_histo(yeo_df_normalized, yeo_df.columns)
    # print(yeo_df.isin([np.inf, -np.inf]).sum().sum())
    return yeo_df

def standardscaler(df, histo=False):
    # yeo transformer
    st = StandardScaler()
    stanni_data = st.fit_transform(df)
    stanni_df = pd.DataFrame(stanni_data, columns=df.columns)

    stanni_df_normalized = stanni_df.apply(lambda x: x / abs(x).max())
    if histo:
        feature_histo(stanni_df_normalized, stanni_df_normalized.columns)
    return stanni_df_normalized

def filter_existing_columns(column_list, df):
    return [col for col in column_list if col in df.columns]

def scaling_by_hand(feature):
    feature = feature.astype('float')
    already_fine = ['% vars in DAG (out of all vars']
    quizas = ['% vars in DAG unbounded (out of vars in DAG']

    """NONNEG COLUMNS"""
    cols_log_plus_one = ['Matrix Equality Constraints', 'Matrix Quadratic Elements', 'Matrix NLP Formula',
                         'Presolve Columns', 'Presolve Global Entities', '#nodes in DAG',
                         '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)']

    log_then_root = ['#nonlinear violations at root']

    cols_sqrt =[]

    cols_cbrt = ['% vars in DAG integer (out of vars in DAG)']

    cols_to_4throot_scale = ['#integer violations at root']

    cols_to_8root_scale = ['Avg work for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
                           'Avg work for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed']

    cols_to_10root_scale = ['Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
                            'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Mixed']

    """NegCols"""

    non_pos = ['Cmp #spatial branching entities fixed (at the root)']
    neg_pos = ['Cmp Avg coefficient spread for convexification cuts']

    # feature[neg_pos] = feature[neg_pos].map(lambda x: np.sign(x) * np.log(abs(x)) if x != 0 else x)

    cols_cbrt += non_pos
    cols_cbrt += neg_pos

    # Filter each list of columns to include only those present in X_scaled
    log_then_root = filter_existing_columns(log_then_root, feature)
    cols_log_plus_one = filter_existing_columns(cols_log_plus_one, feature)
    cols_sqrt = filter_existing_columns(cols_sqrt, feature)
    cols_cbrt = filter_existing_columns(cols_cbrt, feature)
    cols_to_4throot_scale = filter_existing_columns(cols_to_4throot_scale, feature)
    cols_to_8root_scale = filter_existing_columns(cols_to_8root_scale, feature)
    cols_to_10root_scale = filter_existing_columns(cols_to_10root_scale, feature)
    # adding 1 before logging results in 0 being zero and the rest greater than zero, because log(1)=0
    feature.loc[:, cols_log_plus_one] = feature.loc[:, cols_log_plus_one] + 1
    feature.loc[:, cols_log_plus_one] = feature.loc[:, cols_log_plus_one].map(lambda x: np.log(x) if x > 10 ** (-6) else 0)

    feature.loc[:, log_then_root] = feature.loc[:, log_then_root].map(lambda x: np.log(x) if x > 10 ** (-6) else 0)
    cols_sqrt += log_then_root
    feature.loc[:, cols_sqrt] = feature.loc[:, cols_sqrt].map(np.sqrt)

    feature.loc[:, cols_cbrt] = feature.loc[:, cols_cbrt].map(np.cbrt)

    feature.loc[:, cols_to_4throot_scale] = feature.loc[:, cols_to_4throot_scale].map(lambda x: np.power(x, 0.25))
    feature.loc[:, cols_to_8root_scale] = feature.loc[:, cols_to_8root_scale].map(lambda x: np.power(x, 0.125))
    feature.loc[:, cols_to_10root_scale] = feature.loc[:, cols_to_10root_scale].map(lambda x: np.power(x, 0.1))

    feature = feature.apply(lambda x: x / abs(x).max())
    return feature

def create_relevant_df(df, label):
    rel_indices = label[label != 0].index
    rel_df = df.loc[rel_indices, ['Virtual Best', 'Pot Time Save']].copy()
    rel_df['Label'] = label[rel_indices]
    return rel_df

def predicted_time(time_df, prediction_df):
    pred_df = time_df.copy()
    pred_df['Predicted Time'] = pred_df['Virtual Best']
    for index, row in prediction_df.iterrows():
        if row['Right or Wrong'] != 1:
            pred_df.loc[index, 'Predicted Time'] = max(pred_df.loc[index, 'Final solution time (cumulative) Mixed'], pred_df.loc[index, 'Final solution time (cumulative) Int'])
    pred_df['Time Save/Loss'] = abs(pred_df['Predicted Time'] - pred_df['Virtual Best'])

    mixed_mean = shifted_geometric_mean(pred_df['Final solution time (cumulative) Mixed'], 0.5)
    predicted_time_mean = shifted_geometric_mean(pred_df['Predicted Time'], 0.5)
    vbs_mean = shifted_geometric_mean(pred_df['Virtual Best'], 0.5)
    relative_to_mixed_mean = [1.0, predicted_time_mean/mixed_mean, vbs_mean/mixed_mean]
    return relative_to_mixed_mean

def accuracy(prediction_df):
    # instances where one rule is by a factor of more then 0.5 faster are considered "extreme"
    extreme_cases_df = prediction_df[np.abs(prediction_df['Actual']) >= 2]
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
    acc_df = pd.DataFrame({'Intervall': ['Complete', '(0,0.5)', '[2, inf)'],
                           'Accuracy': [total_accuracy, mid_accuracy, extreme_accuracy]})
    return acc_df

def get_importances(importance_dict, model:str, to_excel=False):
    importance_df = pd.DataFrame(list(importance_dict.items()), columns=["Feature", "LinScore"])
    importance_df['ForScore'] = importance_df['Feature'].map(importance_dict)
    importance_df['CombinedScore'] = importance_df['Feature'].map(combined_impo)

    importance_df_sorted = importance_df.sort_values(by="CombinedScore", ascending=True)
    if to_excel:
        importance_df_sorted.to_excel(
            f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/feature_importance_1_seeds{date_string}.xlsx',
            index=False)
    return importance_df_sorted

def forest_regression(features:DataFrame, label:DataFrame, seed, title_idea:str,  show_accuracy=False):
    show_accuracy = False
    rf = RandomForestRegressor(n_estimators=100, random_state=seed)
    x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=seed)
    # only instances with actual label of unequal 0 are from interest
    relevant_indices = y_test[y_test != 0.0].index
    x_test_relevant = x_test.loc[relevant_indices, :]
    y_test_relevant = y_test.loc[relevant_indices]
    # fit model and predict label
    rf.fit(x_train, y_train.values.ravel())
    # Get feature importance
    importance = rf.feature_importances_
    # Display feature importance
    feature_importance_df = pd.DataFrame({'Feature': features.columns,'Importance': importance})
    feature_importance_sorted_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    forest_feature_importance_df[title_idea] = importance

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
        bar_plot(accuracy_df, title_idea)
    # Evaluate the model
    mse = mean_squared_error(y_test_relevant, y_pred_relevant)
    r2 = r2_score(y_test_relevant, y_pred_relevant)
    # print('ForMSE: ', mse, 'ForR2: ', r2)
    return pred_df, accuracy_df, mse, r2, feature_importance_sorted_df

def linear_regression(features:DataFrame, label:DataFrame, seed, title_idea:str, show_accuracy=False):
    show_accuracy = False
    linreg = LinearRegression()

    x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=seed)
    # only instances with actual label of unequal 0 are from interest
    relevant_indices = y_test[y_test != 0.0].index
    x_test_relevant = x_test.loc[relevant_indices, :]
    y_test_relevant = y_test.loc[relevant_indices]

    # fit model and predict label
    linreg.fit(x_train, y_train.values)

    # Extract coefficients
    coefficients = linreg.coef_
    # Create a DataFrame for better visualization
    feature_importance = pd.DataFrame(
        {"Feature": features.columns, "Importance": coefficients}).sort_values(by="Importance", key=abs, ascending=False)

    linear_feature_importance_df[title_idea] = coefficients

    y_pred_relevant = linreg.predict(x_test_relevant)
    # create df which contains the predicted and actual values and if the sign is correct
    pred_df = pd.DataFrame({'Prediction': y_pred_relevant, 'Actual': y_test_relevant}, index=relevant_indices)
    pred_df['Right or Wrong'] = (np.sign(pred_df['Prediction']) == np.sign(pred_df['Actual'])).astype(int)
    # add column containing the absolute difference in prediction and actual
    pred_df['Abs Time Diff'] = abs(pred_df['Prediction'] - pred_df['Actual'])
    # add column to check by what factor the prediction is off
    pred_df['Factor'] = pred_df['Prediction'] / pred_df['Actual']

    accuracy_df = accuracy(pred_df)
    if show_accuracy:
        bar_plot(accuracy_df, title_idea)
    # Evaluate the model
    mse = mean_squared_error(y_test_relevant, y_pred_relevant)
    r2 = r2_score(y_test_relevant, y_pred_relevant)
    # print('LinMSE: ', mse, 'LinR2: ', r2)
    return pred_df, accuracy_df, mse, r2, feature_importance

def bar(values, bar_names, title):
    plt.bar(bar_names, values, align='center')
    plt.title(title)
    plt.xticks(range(len(bar_names)), bar_names)
    plt.show()
    plt.close()

def pot_time_save_per_intervall():
    everything, x, y = read_data()
    relevant_time_df = create_relevant_df(everything, y)
    # maybe remove outlier
    # relevant_time_df = relevant_time_df[abs(relevant_time_df['Label'])<=50]

    close_to_each_other_indices = relevant_time_df[abs(relevant_time_df['Label'])<0.5].index
    half_indices = relevant_time_df[abs(relevant_time_df['Label'])>=0.5].index
    double_indices = relevant_time_df[abs(relevant_time_df['Label'])>=1].index
    threehalfs_indices = relevant_time_df[abs(relevant_time_df['Label'])>=1.5].index
    threetimes_indices = relevant_time_df[abs(relevant_time_df['Label'])>=2].index


    best_time = relevant_time_df['Pot Time Save'].sum()
    close_time = relevant_time_df.loc[close_to_each_other_indices, 'Pot Time Save'].sum()
    half_time = relevant_time_df.loc[half_indices, 'Pot Time Save'].sum()
    double_time = relevant_time_df.loc[double_indices, 'Pot Time Save'].sum()
    threehalfs_time = relevant_time_df.loc[threehalfs_indices, 'Pot Time Save'].sum()
    threetimes_time = relevant_time_df.loc[threetimes_indices, 'Pot Time Save'].sum()
    print(best_time, threetimes_time)
    bar([best_time, close_time, half_time, double_time, threehalfs_time, threetimes_time],
        ['Total\n 298 Inst', '(0,0.5)\n 140 Inst', '[0.5, inf)\n158 Inst', '[1, inf)\n121 Inst',
         '[1.5, inf)\n94 Inst', '[2, inf)\n80 Inst'],
        'Potential Time Save per Intervall')

def main(rand_seeds, acc_to_ex=False, sgm_to_excel=False):
    full_data, feature_df, label_series = read_data()
    # feature_df = feature_df[['Avg work for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
    #                         '% vars in DAG integer (out of vars in DAG)', '% vars in DAG unbounded (out of vars in DAG)',
    #                         'Presolve Global Entities']]

    time_mixed_int_vbs = full_data[['Final solution time (cumulative) Mixed', 'Final solution time (cumulative) Int', 'Virtual Best']].copy()

    feat_importance_lin = {key: 0 for key in feature_df.columns}
    feat_importance_for = {key: 0 for key in feature_df.columns}

    columns_to_be_imputed = [
        'Avg work for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
        'Avg work for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
        'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Mixed',
        'Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed']

    imputations = [0, 'Median', 'Mean']
    scaler = ['NoScaling', 'byHand', 'Yeo-Johnson', 'StandardScaler']
    regressors = ['LinearRegression', 'RandomForestRegression']

    collected_accuracies_df = pd.DataFrame({'Intervall': ['Complete', '(0,0.5)', '[0.5, inf)']})

    columns_for_collected_sgm = {}

    # loop over all combinations of imputation, scaling and regressor
    count = 0
    start_time = time.time()
    for rand_seed in rand_seeds:
        print('RandomSeed number: ', count)
        for imputation in imputations:
            imputed_data = impute(feature_df, columns_to_be_imputed, imputation)
            for scaling in scaler:
                if scaling == 'NoScaling':
                    # do not scale the features
                    for regressor in regressors:
                        if regressor == 'LinearRegression':
                            count += 1
                            p, a, m, r, importance = linear_regression(imputed_data, label_series, rand_seed, title_idea=str(imputation)+scaling+regressor+str(count), show_accuracy=True)
                            mean_to_mixed = predicted_time(time_mixed_int_vbs, p)
                            mean_to_mixed.extend(a['Accuracy'].values)
                            columns_for_collected_sgm['Linear: '+str(count)] = mean_to_mixed
                            for i in range(len(importance)):
                                feat_importance_lin[importance['Feature'].iloc[i]] += i
                        elif regressor == 'RandomForestRegression':
                            count += 1
                            p, a, m, r, importance = forest_regression(imputed_data, label_series, rand_seed,  title_idea=str(imputation)+scaling+regressor+str(count), show_accuracy=True)
                            mean_to_mixed = predicted_time(time_mixed_int_vbs, p)
                            mean_to_mixed.extend(a['Accuracy'].values)
                            columns_for_collected_sgm['Forest: '+str(count)] = mean_to_mixed
                            for i in range(len(importance)):
                                feat_importance_for[importance['Feature'].iloc[i]] += i
                        else:
                            print('Not a valid Regressor')
                            return 1
                        collected_accuracies_df[str(imputation) + scaling + regressor] = a['Accuracy']

                elif scaling == 'byHand':
                    scaled_features = scaling_by_hand(imputed_data)
                    for regressor in regressors:
                        if regressor == 'LinearRegression':
                            count += 1
                            p, a, m, r, importance = linear_regression(scaled_features, label_series, rand_seed, title_idea=str(imputation)+scaling+regressor+str(count), show_accuracy=True)
                            mean_to_mixed = predicted_time(time_mixed_int_vbs, p)
                            mean_to_mixed.extend(a['Accuracy'].values)
                            columns_for_collected_sgm['Linear: '+str(count)] = mean_to_mixed
                            for i in range(len(importance)):
                                feat_importance_lin[importance['Feature'].iloc[i]] += i
                        elif regressor == 'RandomForestRegression':
                            count += 1
                            p, a, m, r, importance = forest_regression(scaled_features, label_series, rand_seed, title_idea=str(imputation)+scaling+regressor+str(count), show_accuracy=True)
                            mean_to_mixed = predicted_time(time_mixed_int_vbs, p)
                            mean_to_mixed.extend(a['Accuracy'].values)
                            columns_for_collected_sgm['Forest: '+str(count)] = mean_to_mixed
                            for i in range(len(importance)):
                                feat_importance_for[importance['Feature'].iloc[i]] += i
                        else:
                            print('Not a valid Regressor')
                            return 1
                        collected_accuracies_df[str(imputation) + scaling + regressor] = a['Accuracy']

                elif scaling == 'Yeo-Johnson':
                    scaled_features = yeo_johnson(imputed_data)
                    for regressor in regressors:
                        if regressor == 'LinearRegression':
                            count += 1
                            p, a, m, r, importance = linear_regression(scaled_features, label_series, rand_seed, title_idea=str(imputation)+scaling+regressor+str(count), show_accuracy = True)
                            mean_to_mixed = predicted_time(time_mixed_int_vbs, p)
                            mean_to_mixed.extend(a['Accuracy'].values)
                            columns_for_collected_sgm['Linear: '+str(count)] = mean_to_mixed
                            for i in range(len(importance)):
                                feat_importance_lin[importance['Feature'].iloc[i]] += i
                        elif regressor == 'RandomForestRegression':
                            count += 1
                            p, a, m, r, importance = forest_regression(scaled_features, label_series, rand_seed, title_idea=str(imputation)+scaling+regressor+str(count), show_accuracy=True)
                            mean_to_mixed = predicted_time(time_mixed_int_vbs, p)
                            mean_to_mixed.extend(a['Accuracy'].values)
                            columns_for_collected_sgm['Forest: '+str(count)] = mean_to_mixed
                            for i in range(len(importance)):
                                feat_importance_for[importance['Feature'].iloc[i]] += i
                        else:
                            print('Not a valid Regressor')
                            return 1
                        collected_accuracies_df[str(imputation) + scaling + regressor] = a['Accuracy']

                elif scaling == 'StandardScaler':
                    scaled_features = standardscaler(imputed_data)
                    for regressor in regressors:
                        if regressor == 'LinearRegression':
                            count += 1
                            p, a, m, r, importance = linear_regression(scaled_features, label_series, rand_seed,
                                                                       title_idea=str(imputation) + scaling + regressor + str(count), show_accuracy=True)
                            mean_to_mixed = predicted_time(time_mixed_int_vbs, p)
                            mean_to_mixed.extend(a['Accuracy'].values)
                            columns_for_collected_sgm['Linear: ' + str(count)] = mean_to_mixed
                            for i in range(len(importance)):
                                feat_importance_lin[importance['Feature'].iloc[i]] += i
                        elif regressor == 'RandomForestRegression':
                            count += 1
                            p, a, m, r, importance = forest_regression(scaled_features, label_series, rand_seed,
                                                                       title_idea=str(imputation) + scaling + regressor + str(count), show_accuracy=True)
                            mean_to_mixed = predicted_time(time_mixed_int_vbs, p)
                            mean_to_mixed.extend(a['Accuracy'].values)
                            columns_for_collected_sgm['Forest: ' + str(count)] = mean_to_mixed
                            for i in range(len(importance)):
                                feat_importance_for[importance['Feature'].iloc[i]] += i
                        else:
                            print('Not a valid Regressor')
                            return 1
                        collected_accuracies_df[str(imputation)+scaling+regressor] = a['Accuracy']

                else:
                    print('Not a valid scaler.')
                    return 1

    collected_sgm_df = pd.DataFrame({'TimeShiftedGeoMean': ['Mixed', 'Predicted', 'Virtual Best', 'Complete', '(0-0.5)', '[0.5,inf)', '[2,inf)']})
    sgm_with_accuracies_df = pd.concat([collected_sgm_df, pd.DataFrame(columns_for_collected_sgm)], axis=1)
    if acc_to_ex:
        collected_accuracies_df.to_excel(f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/collected_accuracies_{date_string}.xlsx',
                      index=False)
    if sgm_to_excel:
        sgm_with_accuracies_df.to_excel(f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/50_collected_shifted_geo_means_with_accuracies_extreme_extreme_{date_string}.xlsx',
                  index=False)

    combined_importance = {key: feat_importance_lin[key] + feat_importance_for[key] for key in feat_importance_lin}
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    return feat_importance_lin, feat_importance_for, combined_importance, collected_accuracies_df



hundred_seeds = [2207168494, 288314836, 1280346069, 1968903417, 1417846724, 2942245439, 2177268096, 571870743, 1396620602, 3691808733, 4033267948, 3898118442, 24464804, 882010483, 2324915710, 316013333, 3516440788, 535561664, 1398432260, 572356937, 398674085, 4189070509, 429011752, 2112194978, 3234121722, 2237947797, 738323230, 3626048517, 733189883, 4126737387, 2399898734, 1856620775, 829894663, 3495225726, 1844165574, 1282240360, 2872252636, 1134263538, 1174739769, 2128738069, 1900004914, 3146722243, 3308693507, 4218641677, 563163990, 568995048, 263097927, 1693665289, 1341861657, 1387819803, 157390416, 2921975935, 1640670982, 4226248960, 698121968, 1750369715, 3843330071, 2093310729, 1822225600, 958203997, 2478344316, 3925818254, 2912980295, 1684864875, 362704412, 859117595, 2625349598, 3108382227, 1891799436, 1512739996, 1533327828, 1210988828, 3504138071, 1665201999, 1023133507, 4024648401, 1024137296, 3118826909, 4052173232, 3143265894, 1584118652, 1023587314, 666405231, 2782652704, 744281271, 3094311947, 3882962880, 325283101, 923999093, 4013370079, 2033245880, 289901203, 3049281880, 1507732364, 698625891, 1203175353, 1784663289, 2270465462, 537517556, 2411126429]
one_seed = hundred_seeds[46:47]
ten_seeds = [1024137296, 4024648401, 2912980295, 568995048, 362704412, 1684864875, 1282240360, 829894663, 1341861657, 3626048517]
twenty_seeds = [1507732364, 666405231, 1024137296, 4218641677, 1684864875, 362704412, 4013370079, 3143265894, 2324915710, 1387819803, 3118826909, 1341861657, 1210988828, 2270465462, 1640670982, 537517556, 2237947797, 2942245439, 882010483, 744281271]
fifty_seeds = [563163990, 3495225726, 1684864875, 263097927, 829894663, 958203997, 1396620602, 4218641677, 3308693507, 362704412, 738323230, 537517556, 3049281880, 2093310729, 1784663289, 4052173232, 1280346069, 1210988828, 2207168494, 1174739769, 429011752, 1693665289, 698121968, 4033267948, 325283101, 744281271, 1417846724, 2478344316, 2033245880, 3118826909, 1203175353, 1024137296, 1665201999, 1891799436, 3691808733, 2872252636, 3094311947, 1387819803, 289901203, 2112194978, 1023587314, 1341861657, 923999093, 2942245439, 3898118442, 1023133507, 572356937, 1398432260, 3925818254, 2912980295]

impo_lin, impo_for, combined_impo, accs = main(ten_seeds, acc_to_ex=False, sgm_to_excel=True)
# linear_feature_importance_df.to_excel(f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/linear_importances_yeo_stanni_{date_string}.xlsx', index=False)
# forest_feature_importance_df.to_excel(f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/forest_importances_{date_string}.xlsx', index=False)