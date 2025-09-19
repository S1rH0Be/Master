import time
import pandas as pd
import numpy as np
import os
import sys

from stats_per_combi_september import ranking_feature_importance

from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer

def create_directory(parent_name):

    base_path = f'{parent_name}'
    subdirs = ['Prediction', 'Accuracy', 'Importance', 'SGM']
    for subdir in subdirs:
        os.makedirs(os.path.join(base_path, subdir), exist_ok=True)
    return 0

def save_stats(acc_df, runtime_df, prediction_df, importance_df, impo_ranking, acc_train, sgm_train, base_path, solver)
    acc_df.to_csv(f'{base_path}/Accuracy/{solver}_acc_df.csv', index=True)
    runtime_df.to_csv(f'{base_path}/SGM/{solver}_sgm_runtime.csv', index=True)
    prediction_df.to_csv(f'{base_path}/Prediction/{solver}_prediction_df.csv')
    importance_df.to_csv(f'{base_path}/Importance/{solver}_importance_df.csv', index=True)
    impo_ranking.to_csv(f'{base_path}/Importance/{solver}_importance_ranking.csv', index=False)
    # TrainSet
    acc_train.to_csv(f'{base_path}/Accuracy/{solver}_acc_trainset.csv', index=True)
    sgm_train.to_csv(f'{base_path}/SGM/{solver}_sgm_trainset.csv', index=True)

def log_label(label_series):
    y_pos = label_series[label_series >= 0]
    y_neg = label_series[label_series < 0]
    # log1p calculates log(1+x) numerically stable
    y_pos_log = np.log1p(y_pos) / np.log(10)
    y_neg_log = -(np.log1p(abs(y_neg)) / np.log(10))
    y_log = pd.concat([y_pos_log, y_neg_log]).sort_index()
    return y_log


def regression(data_name, label, features, feature_names, models, scalers, imputer, seeds,
                   label_scalen, mid_threshold=0.1, extreme_threshold=4.0):

    if label_scalen:
        label = log_label(label)
        mid_threshold = np.log10(mid_threshold)
        extreme_threshold = np.log10(extreme_threshold)
    # Store Stats on TestSet
    accuracy_dictionary = {}
    run_time_dictionary = {}
    prediction_dictionary = {}
    importance_dictionary = {}
    # Store Stats on TrainSet
    accuracy_dictionary_train = {}
    run_time_dictionary_train = {}

    for model_name, model in models.items():
        # print(f'Training {model_name}')
        for imputation in imputer:
            for scaler in scalers:
                for seed in seeds:


    return 0,0,0,0,0,0,0

def run_regression_pipeline(data_name, label, feats, storage, models, imputer, scalers, seeds, label_scale):
    features = feats.replace([-1, -1.0], np.nan)

    # Set scalers
    if scalers is not None:
        scaler_dict = {'NoScaling': None,
                       'Standard': StandardScaler(),
                       'MinMax': MinMaxScaler(),
                       'Robust': RobustScaler(),
                       'Yeo': PowerTransformer(method='yeo-johnson'),
                       'Quantile': QuantileTransformer(output_distribution='normal',
                                                       n_quantiles=int(len(features) * 0.8))
                       }
        scalers = [scaler_dict[scaler] for scaler in scalers]
    else:
        scalers = [None]

    # Run Regression
    accuracy, runtime, prediction, importance, impo_ranking, accuracy_trainset, sgm_trainset = (
        regression(data_name, label, features, features.columns, models, scalers, imputer, seeds,
                   label_scalen=label_scale))
    save_stats(accuracy, runtime, prediction, importance, impo_ranking, accuracy_trainset, sgm_trainset, storage, data_name)

def main(fico_or_scip, data:str, features:str, models, imputer, scaler, save_to_directory:str, label_loggen:bool, outlier:bool):
    data = pd.read_csv(data)
    label_series = data['Cmp Final solution time (cumulative)']

    # setup directory
    os.makedirs(os.path.join(f'{save_to_directory}'), exist_ok=True)

    if label_loggen:
        directory = save_to_directory+"/LoggedLabel"
    else:
        directory = save_to_directory+"/UnscaledLabel"

    # initialize new directory with everytinng needed for stats storage
    create_directory(directory)

    hundred_random_seeds = [2207168494, 288314836, 1280346069, 1968903417, 1417846724, 2942245439, 2177268096, 571870743,
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

    if fico_or_scip.lower() == "fico":
        run_regression_pipeline(
            data_name='fico',
            label=label_series,
            feats=features,
            storage=directory,
            models=models,
            imputer=imputer,
            scalers=scaler,
            seeds=hundred_random_seeds,
            label_scale=label_loggen,
        )
    if fico_or_scip.lower() == "scip":
        run_regression_pipeline(
            data_name='scip',
            label=label_series,
            feats=features,
            storage=directory,
            models=models,
            imputer=imputer,
            scalers=scaler,
            seeds=hundred_random_seeds,
            label_scale=label_loggen,
        )













