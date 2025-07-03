import pandas as pd
from typing import Union
import sys
from sklearn.model_selection import train_test_split


from june_trainer import trainer



# TODO: Check if i can automatically check if all type hints are enforced
def run_regression(features:Union[pd.DataFrame, pd.Series], labels:pd.Series,
               regression_model, imputer, scaler,  random_seed:int, test_set_size:float=0.2):

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_set_size,
                                                        random_state=random_seed)
    # I need them for creating prediction pd.Series
    if (y_test.index == X_test.index).all():
        instance_indices = y_test.index
    else:
        print(f"y_test, {y_test.index}, and X_test, {X_test.index}, do not have the same index! Train test split is broken")
        sys.exit(1)

    # train regressor
    trained_pipeline = trainer(imputation=imputer, scaler=scaler, model=regression_model, features=X_train,
                                label=y_train, seed=random_seed)
    trained_model = trained_pipeline.named_steps['model']
    # make predictions
    prediction = pd.Series(trained_pipeline.predict(X_test), index=instance_indices)
    return prediction, trained_model