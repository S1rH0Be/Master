import pandas as pd

from may_regression import shifted_geometric_mean


def best_accuracy(accuracy_df:pd.DataFrame, filter_by=''):
    indices_list = accuracy_df.index.tolist()
    indices_to_keep = [index for index in indices_list if filter_by in index]

    accuracy_df = accuracy_df.loc[indices_to_keep,:]

    acc_sgm = shifted_geometric_mean(accuracy_df['Accuracy'], 0.0)
    return acc_sgm


scip_acc_df = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasVeinteUno/ScaledLabel/Accuracy/scip_acc_df.csv',
                          index_col=0)
fico_acc_df = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasVeinteUno/ScaledLabel/Accuracy/fico_acc_df.csv',
                          index_col=0)
scaler_names = ['None', 'Standard', 'MinMax', 'Robust', 'Power', 'Quantile']
imputer = ['mean', 'median']
models = ['RandomForest']#, 'RandomForest']
cmp_list = []
for imp in imputer:
    for scaler_name in scaler_names:
        for model in models:
            filter = f'{model}_{imp}_{scaler_name}'
            value_scip = best_accuracy(scip_acc_df, f'{filter}')
            value_fico = best_accuracy(fico_acc_df, f'{filter}')
            cmp_list.append((value_scip, value_fico, filter))
            # print(f'SCIP:{filter}: {value_scip}')
            # print(f'FICO:{filter}: {value_fico}')
            # print('')

best_combis_lin = [('SCIP', 'median_Power+3%'), ('FICO', 'median_Quantile+3%')]
best_combis_for = [('SCIP', 'egal'), ('FICO', 'mean_None')]


# best_accuracy(fico_acc_df,'mean')
# best_accuracy(fico_acc_df,'median')


