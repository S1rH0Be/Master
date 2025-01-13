import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler

from data_cleaning import date_string


def feature_histo(df, columns: list, number_bins=10):
    """
    Create histograms for specified columns in a DataFrame, focusing only on values in (0, 1).
    Args:
        df: The DataFrame containing data.
        columns: List of column names to plot histograms for.
        number_bins: Number of bins for the histograms.
    """
    # Create a figure with subplots for each column dynamically
    fig, axs = plt.subplots(len(columns), 1, figsize=(8, 4 * len(columns)))  # n rows, 1 column

    # If there's only one column, axs is not an array, so we handle it separately
    if len(columns) == 1:
        axs = [axs]

    for i, col in enumerate(columns):
        # Filter values in (0, 1)
        # if df.values.min().min()>=0 and df.values.max().max()<=1:
        #     filtered_data = df[col][(df[col] >= 0) & (df[col] <= 1)]
        # else:
        filtered_data = df[col]
        # Plot histogram with color distinction
        color = 'red' if 'Mixed' in col else ('magenta' if 'Int' in col else 'orange')
        axs[i].hist(filtered_data, bins=number_bins, color=color, alpha=1)#, label=f'Filtered ({len(filtered_data)} points)')
        axs[i].set_title(f'{col}')

    # Adjust layout
    plt.tight_layout()
    # Show the plots once all are created
    plt.show()
    # Close the plot to free up memory
    plt.close()

def yeo_johnson(df, histo=False):
    # yeo transformer
    pt = PowerTransformer(method='yeo-johnson')
    yeo_data = pt.fit_transform(df)
    yeo_df = pd.DataFrame(yeo_data, columns=df.columns)
    yeo_df_normalized = yeo_df.apply(lambda x: x / abs(x).max())
    if histo:
        feature_histo(yeo_df_normalized, yeo_df.columns)

def box_cox(df, histo= False):
    make_strictly_pos_df = df+1

    b_c = PowerTransformer(method='box-cox')
    box_cox_data = b_c.fit_transform(make_strictly_pos_df)
    box_cox_df = pd.DataFrame(box_cox_data, columns=df.columns)
    box_cox_df = box_cox_df.apply(lambda x: x / abs(x).max())
    if histo:
        feature_histo(box_cox_df, box_cox_df.columns)

def replace_neg_ones_with_median(df, cmp_col_names):
    """
    For each column in `cmp_col_names`, find the two related columns (without 'Cmp' at the beginning),
    and replace all occurrences of -1 in those columns with the column's median (ignoring -1).

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - cmp_col_names (list): List of column names with 'Cmp' at the beginning.

    Returns:
    - pd.DataFrame: Modified DataFrame with -1 values replaced.
    """
    changed_cols = []
    for cmp_col in cmp_col_names:
        # Derive the related column names by removing 'Cmp' from the beginning and adding rule to the end
        related_col_name = cmp_col.replace("Cmp ", "")
        matching_columns = [related_col_name+" Mixed", related_col_name+" Int"]
        for col in matching_columns:

            if len(df.loc[df[col] == -1, col])>0:
                # Calculate the median ignoring -1
                median_value = df.loc[df[col] != -1, col].median()
                # Replace -1 with the calculated median
                df[col] = df[col].apply(lambda x: median_value if x == -1 else x)
                if cmp_col not in changed_cols:
                    changed_cols.append(cmp_col)
                changed_cols.append(col)
    return df, changed_cols

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

def impute_and_update_cmp(clean_data_df):
    imputed_data_df = clean_data_df.copy()
    cmp_names = [col for col in imputed_data_df.columns if 'Cmp' in col]
    imputed_data_df, changed_cols = replace_neg_ones_with_median(imputed_data_df, cmp_col_names=cmp_names)
    imputed_updated_data_df = update_cmp(imputed_data_df, changed_cols)
    return imputed_updated_data_df

def create_cmp_df(dataframe, excel_it=False):
    imputed_and_updated_df = impute_and_update_cmp(dataframe)
    #get all columns where we compare features
    cmp_col_names = imputed_and_updated_df.filter(like='Cmp').columns
    # divide cmp columns by 100 to get 'actual' factor
    imputed_and_updated_df[cmp_col_names] = imputed_and_updated_df[cmp_col_names] / 100
    #get all columns with features which are same for both branching rules
    static_col_names = imputed_and_updated_df.filter(regex='^(?!.*(Mixed|Int|Cmp)).*$', axis=1).columns
    #only keep the above columns
    cmp_df = imputed_and_updated_df.loc[:, list(static_col_names)+list(cmp_col_names)]
    #drop non valid features
    lean_df = cmp_df.drop(columns=['Cmp Final Objective', 'Pot Time Save', 'permutation seed', 'Virtual Best'])
    #drop cmp cols for deleted features
    lean_df = lean_df.drop(columns=['Cmp Avg ticks for propagation + cutting / entity / rootLPticks',
                                       'Cmp #non-spatial branch entities fixed (at the root)'])



    if excel_it:
        lean_df.to_excel(
        f'/Users/fritz/Downloads/ZIB/Master/ZwischenPräsi_Januar/Data/WholeDataSet/lean_cmp_data_updated_{date_string}.xlsx',
        index=False)

    return lean_df

def read_data(version='06_01'):
    data = pd.read_excel(f'/Users/fritz/Downloads/ZIB/Master/ZwischenPräsi_Januar/Data/WholeDataSet/clean_data_og_cmp_{version}.xlsx').drop(columns='Matrix Name')
    data = create_cmp_df(data)
    features = data.drop(columns='Cmp Final solution time (cumulative)')
    label = data['Cmp Final solution time (cumulative)']
    return data, features, label

full_data, feature_df, label_series = read_data()
cmp_cols = [col for col in feature_df.columns if 'Cmp' in col]
nonneg_cols = [col for col in feature_df.columns if 'Cmp' not in col]
#
# feature_df.to_excel(f'/Users/fritz/Downloads/ZIB/Master/ZwischenPräsi_Januar/Data/Features/cmp_features_{date_string}.xlsx', index=False)
# label_series.to_excel(f'/Users/fritz/Downloads/ZIB/Master/ZwischenPräsi_Januar/Data/Features/label_series_{date_string}.xlsx', index=False)