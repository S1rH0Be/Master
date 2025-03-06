import pandas as pd
import numpy as np

from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from clean_regression import read_data, get_accuracy, predicted_time


def linear_regression(random_seed, used_feature_names='All', extreme_threshold=1.6):
    data, feat, target = read_data()
    if used_feature_names != 'All':
        feat = feat[used_feature_names]



    def make_prediction(features, label, seed):
        # split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=seed)
        # build pipeline
        steps = []
        # add imputer
        steps.append(('imputer', SimpleImputer(strategy='median')))
        # Add scaler
        steps.append(('scaler', QuantileTransformer(n_quantiles=100,output_distribution="normal",random_state=42)))
        # Add model
        linear_regressor = LinearRegression()
        steps.append(('model', linear_regressor))
        # create pipeline
        pipeline = Pipeline(steps)
        # train model
        pipeline.fit(X_train, y_train)

        # Get index of instances with nonzero label
        relevant_indices = y_test[y_test != 0].index
        y_test_relevant = y_test.loc[relevant_indices]
        # predict only on instances with nonzero label
        y_pred_relevant = pipeline.predict(X_test.loc[relevant_indices, :])
        return y_pred, y_test_relevant, linear_regressor.coef_

    y_pred, y_test, feat_coefficients = make_prediction(feat, target, random_seed)

    # get the accuracy of predicted signs
    total_accuracy, extreme_accuracy, number_ex_instances = get_accuracy(y_pred, y_test, extreme_threshold)
    
    def create_prediction_dataframe():
        # get mixed, predicted and best run time on test set
        pred_df = pd.DataFrame({'Prediction': y_pred, 'Actual': y_test},
                               index=y_test.index)
        pred_df['Right or Wrong'] = (np.sign(pred_df['Prediction']) == np.sign(pred_df['Actual'])).astype(int)
        # add column containing the absolute difference in prediction and actual
        pred_df['Abs Time Diff'] = abs(pred_df['Prediction'] - pred_df['Actual'])
        #
        columns_for_collected_sgm = {}
        time_mixed_vbs = data[['Final solution time (cumulative) Mixed', 'Virtual Best']].copy()
        mean_to_mixed = predicted_time(time_mixed_vbs, pred_df)
        columns_for_collected_sgm['Linear'] = mean_to_mixed
        """HIERR IJDBPIBDPIBDPIUBDPIBDPIBIHDBIHJDBHIJDBIJHBDIHBDHIBDHBDHIBDHJDJBJKDBKJDJKDJKDBJKDBJKDBHIJB"""
        return pred_df, columns_for_collected_sgm

    def create_importance_df():
        # create a df to store feature importances
        linear_feature_importance_df = pd.DataFrame({'Feature': feat.columns})
        linear_feature_importance_data = {}
        # store the importance of each feature
        linear_feature_importance_data[f"Linear Regression"] = feat_coefficients
        return linear_feature_importance_

