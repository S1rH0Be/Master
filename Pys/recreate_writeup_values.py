from typing import List, Union, Literal
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime

# plotting
import matplotlib.pyplot as plt

from create_and_scale_cmp_df import yeo_johnson
from visualize_erfolg import shifted_geometric_mean
from visualizer import feature_histo
from bar_plot_accuracy import by_intervall
# scaling
from feature_distribution import scale_by_hand  # , scale_cmp
from sklearn.preprocessing import PowerTransformer, StandardScaler
# regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

# Get the current date
current_date = datetime.now()
# Format it as a string
date_string = current_date.strftime("%d_%m")

top_8_global = ['% vars in DAG integer (out of vars in DAG) Mixed', '% vars in DAG integer (out of vars in DAG) Int',
                '#MIP nodes Int', '#MIP nodes Mixed',
                'Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
                '#nodes in DAG Int', '#nodes in DAG Mixed', 'Avg coefficient spread for convexification cuts Mixed']
top_8_lin = [
    'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Mixed',
    '% vars in DAG unbounded (out of vars in DAG) Int', '% vars in DAG unbounded (out of vars in DAG) Mixed',
    '% vars in DAG integer (out of vars in DAG) Mixed', '% vars in DAG integer (out of vars in DAG) Int',
    '#MIP nodes Int', '#MIP nodes Mixed', 'Avg coefficient spread for convexification cuts Mixed']
top_8_for = ['#MIP nodes Mixed', '#MIP nodes Int', '% vars in DAG integer (out of vars in DAG) Mixed',
             '#nodes in DAG Mixed', '% vars in DAG integer (out of vars in DAG) Int', '#nodes in DAG Int',
             'Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
             'Avg coefficient spread for convexification cuts Mixed']


def read_in_data():
    df = pd.read_excel(
        '/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/Master_Excel/pointfive_is_zero_df.xlsx')
    time = df[['Final solution time (cumulative) Mixed', 'Final solution time (cumulative) Int', 'Virtual Best']]
    # label y
    target = df['Cmp Final solution time (cumulative)']
    # features
    feats = pd.read_excel(
        '/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/Master_Excel/features_pointfive_is_zero_df.xlsx')
    return feats, target, time


def kick_outlier(feat_df, label_series, threshold: int):
    indices_to_keep = label_series[label_series.abs() <= threshold].index

    feats_to_keep = feat_df.loc[indices_to_keep, :]
    labels_to_keep = label_series.loc[indices_to_keep]
    return feats_to_keep, labels_to_keep


def imputer_scaler_regressor():
    imputer = ['Mean']#, 'Median']
    scaler = [None]  # , 'byHand', PowerTransformer(method='yeo-johnson')]
    regressor = [LinearRegression(), RandomForestRegressor(n_estimators=100, random_state=42)]
    return regressor, imputer, scaler


def visualizer(data: pd.DataFrame, title: str, y_lims: List[float], x_ticks: List[str], sort_data=False, colors=None,
               colors_by_goodness=False, title_size=12):
    """Input: DataFrame with first column contains name of value, and second column the values
    Output: Bar for each column"""
    x_tick = x_ticks

    if sort_data:
        if len(x_ticks) > 0:
            # such that each bar keeps its label
            data['x ticks'] = x_ticks
        data.sort_values(by=data.columns[1], ascending=False, inplace=True)
        x_tick = data['x ticks']

    names = data.iloc[:, 0]
    values = data.iloc[:, 1]

    if colors_by_goodness:
        farben = ['green' if v >= 80 else 'red' if v < 70 else 'magenta' for v in values]
        plt.bar(names, values, color=farben)
    elif colors is None:
        plt.bar(names, values)
    else:
        plt.bar(names, values, color=colors)
    plt.title(title, fontsize=title_size)
    plt.ylim(y_lims[0], y_lims[1])  # Set y-axis limits for visibility
    plt.xticks(ticks=names, labels=x_tick)

    plt.show()
    plt.close()


def compare_sgm_relative_to_mixed(time_df: pd.DataFrame, shift: float):
    shifted_mean_mixed = shifted_geometric_mean(time_df['Final solution time (cumulative) Mixed'], shift)
    shifted_means_relative_to_mixed = [np.round(shifted_geometric_mean(time_df[col], shift) / shifted_mean_mixed, 2) for
                                       col in time_df.columns]
    shifted_means_df = pd.DataFrame(
        {'Rule': time_df.columns, 'Shifted Geometric Mean': shifted_means_relative_to_mixed})

    visualizer(shifted_means_df, 'Shifted Geometric Means of RunTime of BranchRule relative to Mixed',
               [0, max(shifted_means_relative_to_mixed) * 1.1], ['Mixed', 'Int', 'Best', 'Predicted'], sort_data=True,
               colors=['green', 'magenta'])
    return shifted_means_df


def get_top_8(top8: list[list]):
    flattened = [value for sublist in top8 for value in sublist]
    # Count occurrences of each item in the list
    counter = Counter(flattened)
    most_common = counter.most_common(8)
    top_8_feats = [feat[0] for feat in most_common]
    top_8_scores = [feat[1] for feat in most_common]
    top_eight_df = pd.DataFrame({'Feature': top_8_feats, 'Appearances': top_8_scores})
    return top_eight_df


class FeaturesAndLabel:
    rand_state = 75475
    test_size = 0.2

    def __init__(self, features: pd.DataFrame, label: pd.Series, name: str):
        self.name = name
        self.label = label
        self.features = features
        self.feature_names = features.columns
        self.feat_train, self.feat_test, self.label_train, self.label_test = train_test_split(self.features,
                                                                                              self.label,
                                                                                              test_size=0.2,
                                                                                              random_state=75475)
        self.feat_train_zero_imputed = self.feat_train.fillna(0)
        self.feat_test_zero_imputed = self.feat_test.fillna(0)
        self.feat_train_mean_imputed = self.feat_train.fillna(self.feat_train.mean())
        self.feat_test_mean_imputed = self.feat_test.fillna(self.feat_test.mean())
        self.feat_train_median_imputed = self.feat_train.fillna(self.feat_train.median())
        self.feat_test_median_imputed = self.feat_test.fillna(self.feat_test.median())

    def __repr__(self):
        return f"Features({self.name})"


class Regressor:
    extreme_threshold = 1.5

    def __init__(self, model: Union[LinearRegression(), RandomForestRegressor()], imputation_name: str,
                 train_features: pd.DataFrame, test_features: pd.DataFrame, train_label: pd.Series,
                 test_label: pd.Series, scaler: Union[Literal['byHand'], PowerTransformer, StandardScaler, None],
                 time_data: pd.DataFrame):
        self.model = model
        self.imputation_name = imputation_name
        self.scaler = scaler
        self.train_label = train_label
        self.test_label = test_label
        self.feature_names = train_features.columns
        self.feat_train = self.scale(train_features)
        self.feat_test = self.scale(test_features)
        self.fitted = self.model.fit(self.feat_train, self.train_label)
        self.feature_importance = self.feature_importance_df()
        self.relevant_indices = self.test_label[self.test_label != 1.0].index
        self.prediction = self.model.predict(self.feat_test.loc[self.relevant_indices, :])
        self.prediction_df = pd.DataFrame(
            {'Prediction': self.prediction, 'Actual': self.test_label.loc[self.relevant_indices]},
            index=self.relevant_indices)
        self.time_data = time_data.loc[self.relevant_indices, :]
        self.time_data['Predicted Time'] = np.where(self.prediction_df['Prediction'] >= 0,
                                                    self.time_data['Final solution time (cumulative) Mixed'],
                                                    self.time_data['Final solution time (cumulative) Int'])

    def scale(self, feat_df: pd.DataFrame):
        if self.scaler is None:
            return feat_df
        elif self.scaler == 'byHand':
            return scale_by_hand(feat_df)
        else:
            yeo_john = self.scaler.fit_transform(feat_df)
            return pd.DataFrame(columns=feat_df.columns, data=yeo_john, index=feat_df.index)

    def visualize_features(self, on_train_set=True, on_test_set=False, bins=10):
        if on_train_set:
            feature_histo(self.feat_train, self.feature_names, number_bins=bins)
        if on_test_set:
            feature_histo(self.feat_test, self.feature_names, number_bins=bins)

    def feature_importance_df(self):
        if isinstance(self.model, RandomForestRegressor):
            return pd.DataFrame({'Feature': self.feature_names, 'Score': np.round(self.fitted.feature_importances_, 2)})

        elif isinstance(self.model, (LinearRegression, Ridge)):
            return pd.DataFrame({'Feature': self.feature_names, 'Score': np.round(self.fitted.coef_, 7)})

        else:
            print('Not a valid model')

    def feature_selector(self, select_by: tuple):
        if select_by[0] == 'Top':
            selected_features_df = self.feature_importance.loc[
                self.feature_importance['Score'].abs().sort_values(ascending=False).index
            ].head(select_by[1])
            return selected_features_df

        elif select_by[0] == 'Threshold':
            selected_features_df = self.feature_importance[self.feature_importance['Score'].abs() >= select_by[1]]
            return selected_features_df

        else:
            print('Not a valid criterion! Try ("Top",int) or ("Threshold",float)')

    @staticmethod
    def visualize_feature_importance(score_df: pd.DataFrame):
        visualizer(score_df, 'Importance of Features', [min(score_df['Score']) * 1.1, max(score_df['Score']) * 1.1],
                   score_df['Feature'], sort_data=False)

    def accuracy(self, title: str):
        self.prediction_df['Right or Wrong'] = (
                    np.sign(self.prediction_df['Prediction']) == np.sign(self.prediction_df['Actual'])).astype(int)
        extreme_cases_df = self.prediction_df[np.abs(self.prediction_df['Actual']) >= self.extreme_threshold]
        mid_cases_df = self.prediction_df.loc[~self.prediction_df.index.isin(extreme_cases_df.index)]
        # number of cases
        number_of_relevant_cases = len(self.prediction_df.index)
        number_of_mid_cases = len(mid_cases_df)
        number_of_extreme_cases = len(extreme_cases_df)
        # accuracy of model differentiated by cases
        total_accuracy = np.round((self.prediction_df['Right or Wrong'].sum() / number_of_relevant_cases) * 100, 2)
        mid_accuracy = np.round((mid_cases_df['Right or Wrong'].sum() / number_of_mid_cases) * 100, 2)
        extreme_accuracy = np.round((extreme_cases_df['Right or Wrong'].sum() / number_of_extreme_cases) * 100, 2)

        # print('Total: ', total_accuracy, 'Mid: ', mid_accuracy, 'Extreme: ', extreme_accuracy)
        acc_df = pd.DataFrame({'Intervall': ['Complete', '[1,1.5)', '[1.5, inf)'],
                               'Accuracy': [total_accuracy, mid_accuracy, extreme_accuracy]})
        # visualizer(acc_df, title, [45, 100], acc_df['Intervall'], colors_by_goodness=True)
        return acc_df


def get_top_features(glob, linear, forrest):
    eight_global = get_top_8(glob)
    eight_lin = get_top_8(linear)
    eight_for = get_top_8(forrest)
    return eight_global, eight_for, eight_lin


t8glob = []
t8lin = []
t8for = []


def predict_all_combinations(regressoren, imputations, scalers, feature_space, label, t_df, compare_sgm=False,
                             show_accuracy=True, show_feature_histogram=False):
    accuracy_df = pd.DataFrame({'Intervall': ['Complete', '[1,1.5)', '[1.5, inf)']})
    importance_df = pd.DataFrame({'Feature': feature_space[0][0].columns})
    for regressor in regressoren:
        for imp in imputations:
            for scal in scalers:
                # print(regressor, imp, scal)
                for feat in feature_space:
                    feats = FeaturesAndLabel(feat[0], label,
                                             'X and y')  # watch out need to change feature space as well
                    if imp == 'Median':
                        model = Regressor(regressor, imp, feats.feat_train_median_imputed,
                                          feats.feat_test_median_imputed,
                                          feats.label_train, feats.label_test, scal, t_df)
                        if compare_sgm:
                            compare_sgm_relative_to_mixed(model.time_data, 0.5)
                        if show_accuracy:
                            accuracy_df[str(regressor) + " " + imp + " " + str(scal) + " " + feat[1]] = \
                            model.accuracy(str(regressor) + imp + str(scal) + feat[1])['Accuracy']
                        if show_feature_histogram:
                            model.visualize_features(bins=15)

                        if model.feature_importance['Score'].max() > 0:
                            importance_df[str(regressor) + imp + str(scal) + feat[1]] = model.feature_importance[
                                'Score']
                            t8glob.append(model.feature_selector(('Top', 8))['Feature'])
                            if isinstance(model.model, RandomForestRegressor):
                                t8for.append(model.feature_selector(('Top', 8))['Feature'])
                            elif isinstance(model.model, LinearRegression):
                                t8lin.append(model.feature_selector(('Top', 8))['Feature'])

                    elif imp == 'Mean':
                        model = Regressor(regressor, imp, feats.feat_train_mean_imputed, feats.feat_test_mean_imputed,
                                          feats.label_train, feats.label_test, scal, t_df)
                        if compare_sgm:
                            compare_sgm_relative_to_mixed(model.time_data, 0.5)
                        if show_accuracy:
                            accuracy_df[str(regressor) + " " + imp + " " + str(scal) + " " + feat[1]] = \
                            model.accuracy(str(regressor) + imp + str(scal) + feat[1])['Accuracy']
                        if show_feature_histogram:
                            model.visualize_features(bins=15)
                        if model.feature_importance['Score'].max() > 0:
                            importance_df[str(regressor) + imp + str(scal) + feat[1]] = model.feature_importance[
                                'Score']
                            t8glob.append(model.feature_selector(('Top', 8))['Feature'])
                            if isinstance(model.model, RandomForestRegressor):
                                t8for.append(model.feature_selector(('Top', 8))['Feature'])
                            elif isinstance(model.model, LinearRegression):
                                t8lin.append(model.feature_selector(('Top', 8))['Feature'])
                                # visualizer(model.feature_importance, 'FeatImp: '+str(regressor)+str(imp)+str(scal), [min(0, model.feature_importance['Score'].min()*1.1), max(0, model.feature_importance['Score'].max()*1.1)], [], sort_data=False, colors=['orange', 'orange', 'orange', 'magenta', 'magenta', 'turquoise', 'turquoise',
                                #                             'magenta', 'magenta', 'turquoise', 'turquoise', 'magenta', 'magenta', 'turquoise', 'turquoise',
                                #                             'magenta', 'magenta', 'turquoise', 'turquoise', 'magenta', 'magenta', 'turquoise', 'turquoise',
                                #                             'magenta', 'magenta', 'turquoise', 'turquoise', 'magenta', 'magenta', 'turquoise', 'turquoise',
                                #                             'magenta', 'magenta'], title_size=10)
    return accuracy_df, importance_df


def main():
    regressoren, imputations, scalers = imputer_scaler_regressor()
    x, y, time = read_in_data()
    print(y)
    y.plot.box()
    # Replace NaNs with the median of each column
    x = x.apply(lambda col: col.fillna(col.median()), axis=0)
    #scale by hand
    # x = scale_by_hand(x)
    # sclae by yeo-john
    yeo_john = PowerTransformer(method='yeo-johnson')
    scaled_values = yeo_john.fit_transform(x)
    x =  pd.DataFrame(columns=x.columns, data=scaled_values, index=x.index)

    x_filtered, y_filtered = kick_outlier(x, y, 50)
    y_filtered_normalized = y_filtered / y_filtered.abs().max()
    y_normalized = y / y.abs().max()
    labels = [y, y_normalized, y_filtered, y_filtered_normalized]
    feature_space_filtered = [(x_filtered, 'All'), (x_filtered[top_8_global], 'Glob8'), (x_filtered[top_8_lin], 'Lin8'),
                              (x_filtered[top_8_for], 'For8')]
    feature_space_unfiltered = [(x, 'All'), (x[top_8_global], 'Glob8'), (x[top_8_lin], 'Lin8'), (x[top_8_for], 'For8')]

    acc_df, importance_df = predict_all_combinations(regressoren, imputations, scalers, feature_space_unfiltered, y,
                                                     time, show_accuracy=True)

    # by_intervall(acc_df)
    acc_df.to_excel(f'/Users/fritz/Downloads/ZIB/Master/ZwischenPräsi_Januar/Predictions/NonCmp/AccuraciesNonCmp/komplett_median_imputed_then_scaled_yeo_john_acc_unfiltered_point_five_is_zero_{date_string}.xlsx',
        index=False)
    # acc_df.to_excel(f'/Users/fritz/Downloads/ZIB/Master/ZwischenPräsi_Januar/Accuracies/acc_filtered_point_five_is_zero_{date_string}.xlsx',
    #     index=False)
    # for column in acc_df.columns[1:]:
    #     values = acc_df[column]
    #     colors = ['red' if v <= 60 else 'green' if v >= 85 else 'blue' for v in
        #           values]  # Assign colors based on conditions
        #
        # # Create the bar plot
        # plt.figure(figsize=(8, 5))
        # plt.bar(range(len(values)), values, color=colors)
        #
        # # Add labels and title
        # plt.xticks(ticks=range(len(values)), labels=acc_df.index, rotation=45)
        # plt.ylabel(column)
        # plt.title(f"Bar Plot for {column}")
        #
        # # Display the plot
        # plt.tight_layout()
        # # plt.show()

    # Iterate through each row of the DataFrame
    for idx, row in acc_df.iloc[:1, 1:].iterrows():
        # Extract values and corresponding column names
        values = row.values  # Values for the row
        columns = acc_df.columns[1:]  # Column names for x-axis labels
        # Assign colors based on conditions
        colors = ['red' if v <= 60 else 'green' if v >= 80 else 'blue' for v in values]

        # Create the bar plot
        plt.figure(figsize=(8, 5))  # Set figure size
        plt.bar(columns, values, color=colors)  # Bar plot with conditional coloring

        # Add plot labels and title
        plt.title(f"Median Yeo-John")  # Title with row index
        plt.ylabel('Value')  # Label for y-axis
        # plt.xlabel('Columns')  # Label for x-axis
        # plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

        # Display the plot
        plt.tight_layout()
        plt.show()


main()

# data = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/ZwischenPräsi_Januar/Data/Features/lean_data_05_01.xlsx')
# cmp_features = data.drop(columns=['Cmp Final solution time (cumulative)'])
# label = data['Cmp Final solution time (cumulative)']
# feature_histo(data, data.columns)
# scaler = StandardScaler()
# standard_scaled_features = scaler.fit_transform(cmp_features)
# standard_scaled_features_df = pd.DataFrame(standard_scaled_features, columns=cmp_features.columns)
# feats = FeaturesAndLabel(standard_scaled_features_df, label, 'X and y')
#
# feature_histo(standard_scaled_features_df, standard_scaled_features_df.columns)
#
# model = Regressor(LinearRegression(), 'Median', feats.feat_train_median_imputed, feats.feat_test_median_imputed,
#                   feats.label_train, feats.label_test, None, t_df)
# accuracy_df[str(LinearRegression())+" "+'Median'+" "+str(scaler)+" "] = model.accuracy(str(LinearRegression())+'Median'+str(scaler))['Accuracy']
# accuracy_df.to_excel(f'/Users/fritz/Downloads/ZIB/Master/ZwischenPräsi_Januar/Accuracies/acc_cmp_{str(scaler)}_{date_string}.xlsx', index=False)


"""
To Show FeatImportance:
1. AllFeatImportance.pdf ##
2. DF with feature names and feature importance score
3. top_eights (Names of selected Features) ##
4. Accuracy on Intervalls depending on Feature Selection (All, global, forrest, lin) 
"""

"""TO-DO:"""
"""4. ResidualGraph"""

""""
Funktionen:
1. Main:
    1.1 Input Data(cleaned, not imputated), Imputation and scaling  
2. Regressor##
3. Imputation##
4. Scaling##
5. Calc Accuracy##
6. Calculate FeatImportance##
7. Visualize Accuracy:##
    5.1 Bar graph with Total, Mid, Extreme##
8. Analyze Model:
    6.1 Create a Time df where only relevant columns for analysis are stored
    6.2 Feature Importance: (Training data, test data) for now just training data
    6.3 Residual Graph
    6.4 Cmp shiftedGeoMean  
"""
