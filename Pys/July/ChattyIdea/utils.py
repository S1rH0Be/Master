import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess_data(X, imputation="mean", scaling="standard"):
    imputer = SimpleImputer(strategy=imputation)
    X_imputed = imputer.fit_transform(X)

    if scaling == "standard":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
    elif scaling == "minmax":
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_imputed)
    else:
        return pd.DataFrame(X_imputed, columns=X.columns)

    return pd.DataFrame(X_scaled, columns=X.columns)