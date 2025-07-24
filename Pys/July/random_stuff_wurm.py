import pandas as pd
from sklearn.preprocessing import QuantileTransformer


fico_logged = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/JulyTry/Bases/FICO/Scaled/logged_fico_clean_data.csv')
fico_basic = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/JulyTry/Bases/FICO/Cleaned/fico_clean_data_753.csv')

feats_logged = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/JulyTry/Bases/FICO/Scaled/logged_fico_feats.csv')
feats_basic = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/JulyTry/Bases/FICO/Cleaned/fico_features_753.csv')


# Initialize the transformer
qt = QuantileTransformer(output_distribution='normal', n_quantiles=int(len(feats_logged) * 0.8))

# Apply quantile transform
feats_basic_transformed = pd.DataFrame(qt.fit_transform(feats_basic), columns=feats_basic.columns)
feats_logged_transformed = pd.DataFrame(qt.fit_transform(feats_logged), columns=feats_logged.columns)



for i in feats_basic.columns:
    print(i)
    print(feats_logged_transformed[i].min())
    print(feats_logged_transformed[i].max())
