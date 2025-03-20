from typing import List
import pandas as pd
from ugly import check_col_consistency


def find_broken_instances(df:pd.DataFrame, requirements_df:pd.DataFrame)->List:
    """ requirement_dict has as keys column names and as values a list of requirements the instances have to
     fulfill in this column """
    # only take columns where there are requirements for
    columns_to_check = []
    requirements_df = requirements_df.drop(columns=[col for col in requirements_df.columns if 'Cmp' in col])

    for col_name in df.columns:
        if col_name in requirements_df.columns:
            columns_to_check.append(col_name)

    dataframe, broken_instances = check_col_consistency(df[columns_to_check], requirements_df[columns_to_check], SCIP=True)


    return broken_instances

def read_data(file_path_data, file_path_requirements):
    data = pd.read_excel(file_path_data)
    requirement_df = pd.read_excel(file_path_requirements, header=0)

    return data, requirement_df

def main(file_path_dataset, file_path_requirements_xlsx, to_excel=False):
    data, requirements = read_data(file_path_dataset, file_path_requirements_xlsx)

    broken_instances_and_reason = find_broken_instances(data, requirements)
    broken_names = [name[0] for name in broken_instances_and_reason]
    clean_data = data[~data['Matrix Name'].isin(broken_names)]

    clean_feats = clean_data[['#integer violations at root Mixed',
                               '#nodes in DAG Mixed',
                               'Avg coefficient spread for convexification cuts Mixed',
                               'Presolve Global Entities Mixed', 'Presolve Columns Mixed',
                               '#nonlinear violations at root Mixed', 'Matrix Equality Constraints',
                               'Matrix NLP Formula', '% vars in DAG (out of all vars) Mixed',
                               '% vars in DAG integer (out of vars in DAG) Mixed',
                               'Matrix Quadratic Elements',
                               '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG) Mixed',
                               '% vars in DAG unbounded (out of vars in DAG) Mixed']].copy()

    if to_excel:
        clean_data.to_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/ready_to_ml/all_with_feature/clean_stefan.xlsx',
                            index=False)
        clean_feats.to_excel(
            '/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/ready_to_ml/all_with_feature/clean_feats_stefan.xlsx',
            index=False)


    return clean_data, broken_instances_and_reason









