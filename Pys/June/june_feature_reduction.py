import pandas as pd

def get_top_x_by_regressor(impo_rank:pd.DataFrame, sort_by:str, x:int):
    impo_rank.sort_values(by=sort_by, ascending=True, inplace=True)
    top_x = impo_rank['Feature'].head(x)
    return top_x
# TODO write feature reduction
def feature_reduction_fico(path_to_store:str):
    fico_data = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/June/Bases/FICO/fico_clean_data.csv')
    fico_feats = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/June/Bases/FICO/fico_feats_918_ready_to_ml.csv')


