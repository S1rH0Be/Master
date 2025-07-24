import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

def compute_accuracy(model, X_test, y_test):
    return r2_score(y_test, model.predict(X_test))

def compute_sgm(model, X_test, y_test):
    residuals = y_test - model.predict(X_test)
    return np.std(residuals)

def compute_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
    else:
        return pd.DataFrame({"feature": feature_names, "importance": [None]*len(feature_names)})