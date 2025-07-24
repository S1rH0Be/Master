import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from utils import preprocess_data
from models import get_model
from metrics import compute_accuracy, compute_sgm, compute_feature_importance

def run_pipeline(df, target_column, imputation, scaling, model_type, random_seed):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_processed = preprocess_data(X, imputation=imputation, scaling=scaling)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=random_seed)

    model = get_model(model_type, random_state=random_seed)
    model.fit(X_train, y_train)

    # Create run directory
    run_time = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = os.path.join("runs", run_time)
    os.makedirs(output_dir, exist_ok=True)

    # Compute & save metrics
    accuracy = compute_accuracy(model, X_test, y_test)
    sgm = compute_sgm(model, X_test, y_test)
    importance = compute_feature_importance(model, X_train.columns)

    pd.DataFrame({"accuracy": [accuracy]}).to_csv(os.path.join(output_dir, "accuracy.csv"), index=False)
    pd.DataFrame({"sgm": [sgm]}).to_csv(os.path.join(output_dir, "sgm.csv"), index=False)
    importance.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)