import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler

from Pys.data_cleaning_fico import features
from data_cleaning_fico import date_string


def update_cmp(df, col_list):

    cmp_column_names = [col_list[i] for i in range(0, len(col_list), 3)]
    feature_couples = [(col_list[i], col_list[i+1]) for i in range(1, len(col_list), 3)]

    for i in range(len(cmp_column_names)):
        for index, row in df.iterrows():
            mixed_value = df[feature_couples[i][0]].loc[index]
            int_value = df[feature_couples[i][1]].loc[index]

            if mixed_value == np.inf and abs(int_value) != np.inf:
                df.loc[index, cmp_column_names[i]] = -np.inf
            elif mixed_value == -np.inf and abs(int_value) != np.inf:
                df.loc[index, cmp_column_names[i]] = np.inf
            elif int_value == np.inf and abs(mixed_value) != np.inf:
                df.loc[index, cmp_column_names[i]] = np.inf
            elif int_value == -np.inf and abs(mixed_value) != np.inf:
                df.loc[index, cmp_column_names[i]] = -np.inf
            elif mixed_value == int_value:
                df.loc[index, cmp_column_names[i]] = 0
            elif abs(mixed_value-int_value)<10**(-6):
                df.loc[index, cmp_column_names[i]] = 0
            #Brauch noch nen Wert der das vernünftig repräsentiert, probably max after scaling
            elif (mixed_value==0)|(int_value==0):
                if mixed_value==0:
                    # print('Mixed=0, Int=', int_value)
                    df.loc[index, cmp_column_names[i]] = 100#np.inf ist von Tristan
                else:
                    # print('Int=0, Mixed=', mixed_value)
                    df.loc[index, cmp_column_names[i]] = -100#-np.inf ist von Tristan
            else:
                if mixed_value>int_value:
                    df.loc[index, cmp_column_names[i]] = (1-(mixed_value/int_value))*100
                else:
                    df.loc[index, cmp_column_names[i]] = -(1-(int_value/mixed_value))*100
    return df

# def create_cmp_df(dataframe, excel_it=False):
#     imputed_and_updated_df = impute(dataframe, dataframe.columns, imputation='Median')
#     #get all columns where we compare features
#     cmp_col_names = imputed_and_updated_df.filter(like='Cmp').columns
#     # divide cmp columns by 100 to get 'actual' factor
#     imputed_and_updated_df[cmp_col_names] = imputed_and_updated_df[cmp_col_names] / 100
#     #get all columns with features which are same for both branching rules
#     static_col_names = imputed_and_updated_df.filter(regex='^(?!.*(Mixed|Int|Cmp)).*$', axis=1).columns
#     #only keep the above columns
#     cmp_df = imputed_and_updated_df.loc[:, list(static_col_names)+list(cmp_col_names)]
#     #drop non valid features
#     lean_df = cmp_df.drop(columns=['Cmp Final Objective', 'Pot Time Save', 'permutation seed', 'Virtual Best'])
#     #drop cmp cols for deleted features
#     lean_df = lean_df.drop(columns=['Cmp Avg ticks for propagation + cutting / entity / rootLPticks',
#                                        'Cmp #non-spatial branch entities fixed (at the root)'])
#     if excel_it:
#         lean_df.to_excel(
#         f'/Users/fritz/Downloads/ZIB/Master/ZwischenPräsi_Januar/Data/WholeDataSet/lean_cmp_data_updated_{date_string}.xlsx',
#         index=False)
#
#     return lean_df


