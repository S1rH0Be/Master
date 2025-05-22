import pandas as pd
import os


def main(stat_version:str, stats_be_filtered:str):
    scalers = ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'PowerTransformer', 'QuantileTransformer']
    imputers = ['median', 'mean']
    models = ['LinearRegression', 'RandomForest']

    os.makedirs(os.path.join(f'{stat_version}/{stats_be_filtered}/SplitUp'), exist_ok=True)


    acc = [] # model_imputation_scaler <- 0-th column: row names
    imp = [] # model_imputation_scaler <- col names
    pred = [] # model_imputation_scaler <- col names
    runtime = [] # model_imputation_scaler <- 0-th column: row names

    subdirs = {'Accuracy': acc, 'Importance': imp, 'Prediction': pred, 'RunTime': runtime}
    for subdir in subdirs.keys():
        for root, _, files in os.walk(f'{stat_version}/{subdir}'):
            if not root.endswith('SplitUp'):
                for file in files:
                    if file.endswith('.csv'):
                        file_path = os.path.join(root, file)
                        try:
                            df = pd.read_csv(file_path)
                            subdirs[subdir].append(df)
                        except Exception as e:
                            print(f"Failed to read {file_path}: {e}")

    filter_by_row = ['Accuracy', 'RunTime']
    filter_by_col = ['Importance', 'Prediction']

    partial_dfs = []

    for model in models:
        for imputer in imputers:
            for scaler in scalers:
                if stats_be_filtered in filter_by_row:
                    for df in subdirs[stats_be_filtered]:
                        first_col = df.columns[0]
                        df = df[df[first_col].astype(str).str.contains(f'{model}_{imputer}_{scaler}', na=False)]
                        partial_dfs.append(df)
                        df.to_csv(f'{stat_version}/{stats_be_filtered}/SplitUp/{model}_{imputer}_{scaler}.csv', index=False)
                elif stats_be_filtered in filter_by_col:
                    for df in subdirs[stats_be_filtered]:
                        relevant_col_names = [col_name for col_name in df.columns if f'{model}_{imputer}_{scaler}' in col_name]
                        if stats_be_filtered == 'Importance':
                            relevant_col_names.insert(0, df.columns[0])
                        df = df[relevant_col_names]
                        partial_dfs.append(df)
                        df.to_csv(f'{stat_version}/{stats_be_filtered}/SplitUp/{stats_be_filtered}_{model}_{imputer}_{scaler}.csv',
                                  index=False)

# main('/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasDiez/UnscaledLabel', 'Accuracy')
# main('/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasDiez/UnscaledLabel', 'Importance')
# main('/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasDiez/UnscaledLabel', 'Prediction')
# main('/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasDiez/UnscaledLabel', 'RunTime')
