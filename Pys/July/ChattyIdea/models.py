from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def get_model(model_type, random_state=42):
    if model_type == "linear":
        return LinearRegression()
    elif model_type == "randomforest":
        return RandomForestRegressor(random_state=random_state)
    else:
        raise ValueError("Unsupported model type")