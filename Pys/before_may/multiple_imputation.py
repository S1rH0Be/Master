import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

N_SPLITS = 5
#
# def get_scores_for_imputer(imputer, regressor, features : pd.DataFrame, label : pd.Series):
#     estimator = make_pipeline(imputer, regressor)
#     impute_scores = cross_val_score(
#         estimator, features, label, scoring="neg_mean_squared_error", cv=N_SPLITS
#     )
#     return impute_scores

def get_impute_iterative(features : pd.DataFrame, label : pd.Series):
    imputer = IterativeImputer(
        missing_values=np.nan,
        add_indicator=True,
        random_state=0,
        n_nearest_features=3,
        max_iter=1,
        sample_posterior=True,
    )
    return iterative_impute_scores.mean(), iterative_impute_scores.std()