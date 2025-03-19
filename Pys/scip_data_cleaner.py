from typing import List
import pandas as pd
import numpy as np
from ugly import check_col_consistency



def find_broken_instances(df:pd.DataFrame, requirements_df:pd.DataFrame)->List:
    """ requirement_dict has as keys column names and as values a list of requirements the instances have to
     fulfill in this column """
    # only take columns where there are requirements for
    columns_to_check = []
    requirements_df = requirements_df.drop(columns=[col for col in requirements_df.columns if 'Cmp' in col])
    df.columns = df.columns.str.replace(' Mixed', '', regex=False)

    for col_name in df.columns:
        if col_name in requirements_df.columns:
            columns_to_check.append(col_name)

    dataframe, broken_instances = check_col_consistency(df[columns_to_check], requirements_df[columns_to_check], SCIP=True)


    return broken_instances

def read_data(file_path_data, file_path_requirements):
    data = pd.read_excel(file_path_data)
    requirement_df = pd.read_excel(file_path_requirements, header=0)

    return data, requirement_df

def main(file_path_dataset, file_path_requirements_xlsx):
    data, requirements = read_data(file_path_dataset, file_path_requirements_xlsx)

    broken_instance_names = find_broken_instances(data, requirements)

    print('Broken instances found:', set(broken_instance_names))
    print(len(data))

main("/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/ready_to_ml/all_with_feature/scip_data_reduced_columns_no_nan.xlsx",
          "/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/ready_to_ml/all_with_feature/scip_requirements.xlsx")