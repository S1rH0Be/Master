from itertools import count

import numpy as np
import pandas as pd
from pandas import Series
import time
from datetime import datetime
from pandas.core.interchange.dataframe_protocol import DataFrame
# scikitlearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
# plotting
import matplotlib.pyplot as plt
from feature_distribution import feature_histo

# Get the current date
current_date = datetime.now()
# Format it as a string
date_string = current_date.strftime("%d_%m")

def read_data(version='23_01'):
    data = pd.read_excel(f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/clean_data_final_{version}.xlsx').drop(columns='Matrix Name')
    feats = pd.read_excel(f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/final_features_{version}.xlsx')
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
    y_log = pd.concat([y_pos_log, y_neg_log])
    return y_log

def box_plot(values:Series, title:str):
    plt.boxplot(values, vert=True, patch_artist=True)
    plt.title(title)
    plt.ylabel("Values")
    plt.show()
    plt.close()

def bar_plot(values:Series, title:str):
    return 0

def get_accuracy(prediction, actual):
    # Filter for nonzero labels
    nonzero_indices = actual != 0
    y_test_nonzero = actual[nonzero_indices]
    y_pred_nonzero = prediction[nonzero_indices]

    # Calculate percentage of correctly predicted signs
    correct_signs = np.sum(np.sign(y_test_nonzero) == np.sign(y_pred_nonzero))
    percent_correct_signs = correct_signs / len(y_test_nonzero) * 100 if len(y_test_nonzero) > 0 else np.nan

    # Filter for extreme labels
    extreme_indices = abs(actual) >= 1.5
    y_test_extreme = actual[extreme_indices]
    y_pred_extreme = prediction[extreme_indices]

    # Calculate percentage of correctly predicted signs
    correct_extreme_signs = np.sum(np.sign(y_test_extreme) == np.sign(y_pred_extreme))
    percent_correct_extreme_signs = correct_extreme_signs / len(y_test_extreme) * 100 if len(
        y_test_extreme) > 0 else np.nan

    return percent_correct_signs, percent_correct_extreme_signs

def get_info(values:Series):

    print('Shape: ', values.shape)
    print('Min: ', values.min())
    print('Max: ', values.max())
    print('Mean: ', values.mean())
    print('Std: ', values.std())
    print('Median: ', values.median())
    print('Variance: ', values.var())
    print('Negative r2: ', len(values[values<0]))

def regression(features, label, random_seeds, cross_val=False):

    imputations = ['constant', 'median', 'mean']
    scalers = [None, PowerTransformer('yeo-johnson'), StandardScaler()] #'byHand',
    models = {"LinearRegression": LinearRegression(),
              "RandomForest": RandomForestRegressor()}

    results = []

    linear_feature_importance_df = pd.DataFrame({'Feature': ['Matrix Equality Constraints', 'Matrix Quadratic Elements',
                                                             'Matrix NLP Formula', 'Presolve Columns',
                                                             'Presolve Global Entities',
                                                             '#nodes in DAG', '#integer violations at root',
                                                             '#nonlinear violations at root',
                                                             '% vars in DAG (out of all vars)',
                                                             '% vars in DAG unbounded (out of vars in DAG)',
                                                             '% vars in DAG integer (out of vars in DAG)',
                                                             '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)',
                                                             'Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
                                                             'Avg ticks for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
                                                             'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Mixed',
                                                             'Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
                                                             'Cmp #spatial branching entities fixed (at the root)',
                                                             'Cmp Avg coefficient spread for convexification cuts']})
    forest_feature_importance_df = linear_feature_importance_df.copy()

    count=0
    start_time = time.time()
    for model_name, model in models.items():
        for imputation in imputations:
            for scaler in scalers:
                for seed in random_seeds:
                    count+=1
                    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3,random_state=seed)
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
                    y_pred = pipeline.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    # get the accuracy of predicted signs
                    total_accuracy, extreme_accuracy = get_accuracy(y_pred, y_test)
                    # feature importance
                    if model_name == "LinearRegression":
                        coefficients = model.coef_
                        linear_feature_importance_df[str(model_name)+str(imputation)+str(scaler)+str(count)] = coefficients
                    elif model_name == "RandomForest":
                        importance = model.feature_importances_
                        forest_feature_importance_df[str(model_name)+str(imputation)+str(scaler)+str(count)] = importance

                    if cross_val:
                        # Perform cross-validation
                        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
                        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2')
                        # Store results
                        results.append({
                            "Model": model_name,
                            "Imputation": imputation,
                            "Scaling": scaler.__class__.__name__ if scaler else None,
                            "CV Mean R2": cv_scores.mean(),
                            "CV Std R2": cv_scores.std(),
                            "Test R2": r2,
                            "Accuracy": total_accuracy,
                            "Extreme Accuracy": extreme_accuracy
                        })
                    else:
                        results.append({
                            "Model": model_name,
                            "Imputation": imputation,
                            "Scaling": scaler.__class__.__name__ if scaler else None,
                            "r2": r2,
                            "Accuracy": total_accuracy,
                            "Extreme Accuracy": extreme_accuracy
                        })

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    print(f"Execution time: {elapsed_time:.2f} seconds")
    return results_df, linear_feature_importance_df, forest_feature_importance_df

def regress_on_different_sets_based_on_label_magnitude(log_label=False):
    d, feat, target = read_data()

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

    if log_label:
        feat_below_thousand, target_below_thousand = kick_outlier(feat, target, 1000)
        target_below_thousand = scale_label(target_below_thousand)
        result_below_thousand_df, linear_importance, forest_importance = regression(feat_below_thousand, target_below_thousand, random_seeds=fifty_seeds)
        result_below_thousand_df.to_excel(
            f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/TestCombinations/label_logged_below_1000_50s_v1601_{date_string}.xlsx',
            index=False)
        return linear_importance, forest_importance

    else:
        result_df = regression(feat, target, random_seeds=ten_seeds)
        result_df.to_excel(f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/TestCombinations/10s_v1601_{date_string}.xlsx', index=False)

        feat_below_thousand, target_below_thousand = kick_outlier(feat, target, 1000)
        result_below_thousand_df = regression(feat_below_thousand, target_below_thousand, random_seeds=ten_seeds)
        result_below_thousand_df.to_excel(f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/TestCombinations/below_1000_10s_v1601_{date_string}.xlsx', index=False)

        feat_below_twohundred, target_below_twohundred = kick_outlier(feat, target, 200)
        result_below_twohundred_df = regression(feat_below_twohundred, target_below_twohundred, random_seeds=ten_seeds)
        result_below_twohundred_df.to_excel(f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/TestCombinations/below_200_10s_v1601_{date_string}.xlsx', index=False)

extreme_all = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/TestCombinations/10s_v1601.xlsx')
extreme_thousand = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/TestCombinations/label_below_1000_10s_v1601.xlsx')
extreme_twohundred = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/TestCombinations/label_below_200_10s_v1601.xlsx')

data_sets = [extreme_all, extreme_thousand, extreme_twohundred]
linear_data_sets = [df[df['Model'] == 'LinearRegression'] for df in data_sets]
forest_data_sets = [df[df['Model'] == 'RandomForest'] for df in data_sets]


# lin_impo, for_impo = regress_on_different_sets_based_on_label_magnitude(log_label=True)
# lin_impo.to_excel(
#     f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/TestCombinations/lin_importance_label_logged_below_1000_50s_v1601_{date_string}.xlsx',
#     index=False)
# for_impo.to_excel(
#     f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/TestCombinations/for_importance_label_logged_below_1000_50s_v1601_{date_string}.xlsx',
#     index=False)

from scipy.stats import gmean
def plot_column_variance(df, model_name:str):
    """
    Calculates the variance of each column in a DataFrame and plots it in a bar graph.

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    None
    """
    # Calculate geometric mean for each column
    geometric_means = df.apply(lambda col: gmean(col[col > 0]), axis=0)

    # Plot the variance as a bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(geometric_means.index, geometric_means.values, color='skyblue')

    # Add labels and title
    plt.xlabel('Runs')
    plt.ylabel('Variance')
    plt.title(f'GeoMean of Feature Importance for each {model_name} run')
    plt.tight_layout()
    # Remove x-axis ticks
    plt.xticks([])
    # Show the plot
    plt.show()

# lin = pd.read_excel(f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/TestCombinations/lin_importance_label_logged_below_1000_50s_v1601_{date_string}.xlsx').drop('Feature', axis=1)
# forest = pd.read_excel(f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/TestCombinations/for_importance_label_logged_below_1000_50s_v1601_{date_string}.xlsx').drop('Feature', axis=1)
# plot_column_variance(abs(lin), 'Linear')
# plot_column_variance(forest, 'Forest')