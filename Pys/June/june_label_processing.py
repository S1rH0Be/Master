import numpy as np
import pandas as pd

import sys

from typing import Union

def remove_outlier(data_series:pd.Series, threshold:Union[float, int]):
    # TODO return indices of outlier/not outlier. Or can i just use
    outlier_removed = data_series[data_series<threshold]
    return outlier_removed

def get_logged_label(labels:pd.Series):
    nonneg_labe = labels[labels >= 0]
    negative_label = labels[labels < 0]
    # log1p calculates log(1+x) numerically stable
    y_pos_log = np.log1p(nonneg_labe) / np.log(10)
    y_neg_log = (np.log1p(abs(negative_label)) / np.log(10)) * -1
    y_log = pd.concat([y_pos_log, y_neg_log]).sort_index()
    return y_log

def get_label(data_set:pd.DataFrame, log_label=False, kick_outlier=False, outlier_threshold=None):
    label = data_set['Cmp Final solution time (cumulative)']

    if kick_outlier:
        if not (isinstance(outlier_threshold, float) or isinstance(outlier_threshold, int)):
            print(f'Outlier threshold {outlier_threshold} is a {type(outlier_threshold)} not a float or int.')
            sys.exit(1)
        else:
            label = remove_outlier(data_series=label, threshold=outlier_threshold)

    if log_label:
        label = get_logged_label(labels=label)

    return label
