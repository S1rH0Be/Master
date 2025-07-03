import pandas as pd
import numpy as np

import os
import sys

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from typing import List

from june_stats import ranking_feature_importance

from june_regression_v2 import run_regression
from june_label_processing import get_label
from june_stats import get_predicted_run_time_sgm, get_feature_importances, get_accuracy, get_shares

def setup_directory(path, new_directory_name):
    # Setup directory
    os.makedirs(os.path.join(path, new_directory_name), exist_ok=True)

def dict_to_csv(path:str, stat_name:str, filename:str, dictionary:dict, column_names:list) -> None:
    setup_directory(path, stat_name)

    df = pd.DataFrame.from_dict(dictionary, orient='index', columns=column_names)
    df.to_csv(os.path.join(path, stat_name, filename), index=True, index_label='Run')
    return df

def proper_log_ten(value):
    if not isinstance(value, (int, float)):
        print(f"Value {value} not numeric")
        sys.exit(1)
    if value < 1:
        logged_value = np.log1p(value)/np.log(10)
    else:
        logged_value = np.log10(value)
    return logged_value



def run_fico_experiment(fico_data, fico_feats, path_to_storage:str, run_name:str, models:List[str], imputation:List[str], scaling:List[str], seeds='Hundred', logged_label=True,
                        remove_outlier=False, threshold_outlier=50, size_of_test_set=0.2,mid_threshold=0.1, extreme_threshold=4):
    """
    This should call each function seperately if i need it. Maybe for looking at stats or something.
    """

    if logged_label:
        mid_threshold = proper_log_ten(mid_threshold)
        extreme_threshold = proper_log_ten(extreme_threshold)

    if seeds.lower() == 'hundred':
        seeds = [2207168494, 288314836, 1280346069, 1968903417, 1417846724, 2942245439, 2177268096, 571870743,
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

    models_dict = {'LinearRegression': LinearRegression(),
              'RandomForest': RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)}

    label = get_label(fico_data, log_label=logged_label, kick_outlier=remove_outlier, outlier_threshold=threshold_outlier)

    # store stats in dictionaries for csv later
    accuracy_dict = {}
    importance_dict = {}
    sgm_dict = {}
    shares_dict = {}


    for model in models:
        if model in models_dict.keys():
            model = models_dict[model]
        else:
            print(f"Model {model} not supported.")
            pass
        for seed in seeds:
            for imputer in imputation:
                for scaler in scaling:
                    prediction, trained_model = run_regression(fico_feats, label, model, imputer=imputer, scaler=scaler,
                                                                    random_seed=seed, test_set_size=size_of_test_set)
                    test_indices = prediction.index

                    # get all stats
                    # importance
                    feature_importance = get_feature_importances(trained_model, fico_feats.columns)
                    # accuracy
                    overall_acc, mid_acc, number_mid_instances, extreme_acc, number_extreme_instances = get_accuracy(
                        prediction, label.loc[test_indices], mid_threshold=mid_threshold, extreme_threshold=extreme_threshold)
                    # shares
                    actual_shares, predicted_shares = get_shares(prediction, label.loc[test_indices])
                    # runtime: Predicted, Mixed, Int, VBS
                    sgms = get_predicted_run_time_sgm(prediction, fico_data, 1, 'Mixed')

                    # stats in dict
                    key = f"{model}_{imputer}_{scaler}_{seed}"
                    accuracy_dict[key] = [overall_acc, mid_acc, number_mid_instances, extreme_acc, number_extreme_instances]
                    importance_dict[key] = feature_importance
                    sgm_dict[key] = sgms
                    shares_dict[key] = actual_shares+predicted_shares

    # write stats to csv
    store_in = os.path.join(path_to_storage, run_name)
    # ACCURACY
    dict_to_csv(store_in, "Accuracy", f"accuracy_fico_{run_name}.csv",
                accuracy_dict, column_names=['Accuracy', 'Mid Accuracy', 'Mid Instances', 'Extreme Accuracy', 'Extreme Instances'])
    # TODO: Check if mapping importance to feature is correct
    #FEATUREIMPORTANCE
    dict_to_csv(store_in, "FeatureImportance", f"feature_importance_fico_{run_name}.csv",
                               importance_dict, column_names=fico_feats.columns.tolist())
    # SGM
    dict_to_csv(store_in, "SGM", f"sgm_fico_{run_name}.csv", sgm_dict,
                column_names=['Predicted', 'Mixed', 'Int', 'VBS'])
    # SHARES
    dict_to_csv(store_in, "Shares", f"shares_fico_{run_name}.csv", shares_dict,
                column_names=['Mixed Actual', 'Int Actual', 'Mixed Predicted', 'IntPredicted'])

def main():
    fico_data = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/June/Bases/FICO/fico_clean_data.csv')
    fico_feats = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/June/Bases/FICO/fico_feats_918_ready_to_ml.csv')

    path = f'/Users/fritz/Downloads/ZIB/Master/June/Runs'
    run = 'Uno'

    run_fico_experiment(fico_data=fico_data,
                        fico_feats=fico_feats,
                        path_to_storage=path,
                        run_name=run,
                        models=['LinearRegression', 'RandomForest'],
                        imputation=['median'],
                        scaling=['Quantile'],
                        logged_label=True)

main()