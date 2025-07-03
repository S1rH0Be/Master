import pandas as pd
import numpy as np
from typing import Union

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer


# TODO Call funciton AFTER train test split

def get_scaler(scaler_name:str, number_of_instances:int):
    scaler_dict = {'NoScaling': None,
                   'Standard': StandardScaler(),
                   'MinMax': MinMaxScaler(),
                   'Robust': RobustScaler(),
                   'Yeo': PowerTransformer(method='yeo-johnson'),
                   'Quantile': QuantileTransformer(output_distribution='normal', n_quantiles=int(number_of_instances*0.8))
                   }

    if scaler_name not in scaler_dict.keys():
        raise ValueError('Invalid scaler name: {}'.format(scaler_name))
    scaler = scaler_dict[scaler_name]
    return scaler

# TODO add debug messages
def trainer(imputation:str, scaler:str, model:Union[LinearRegression, RandomForestRegressor],
            features:Union[pd.DataFrame, pd.Series], label:pd.Series, seed:int):
    # treat -1 as missing value
    features = features.replace([-1, -1.0], np.nan)
    # Build pipeline
    # Update model-specific parameters
    if isinstance(model, RandomForestRegressor) == "RandomForest":
        model.random_state = seed
    # Add imputation
    steps = [('imputer', SimpleImputer(strategy=imputation))]
    # Add scaling if applicable
    if scaler:
        scaler = get_scaler(scaler, len(features))
        steps.append(('scaler', scaler))
    # Add model
    steps.append(('model', model))
    # Create pipeline
    pipeline = Pipeline(steps)
    # Train the pipeline
    pipeline.fit(features, label)
    return pipeline
