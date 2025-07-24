from pipeline import run_pipeline
from sklearn.datasets import load_diabetes

if __name__ == "__main__":
    data = load_diabetes(as_frame=True)
    df = data.frame
    target = "target"

    run_pipeline(
        df=df,
        target_column=target,
        imputation="mean",       # Options: 'mean', 'median', etc.
        scaling="standard",      # Options: 'standard', 'minmax', None
        model_type="randomforest",  # Options: 'linear', 'randomforest'
        random_seed=42
    )