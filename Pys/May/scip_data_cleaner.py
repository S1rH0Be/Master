from typing import List
import pandas as pd
from ugly import check_col_consistency


def find_broken_instances(df:pd.DataFrame, requirements_df:pd.DataFrame)->List:
    """ requirement_dict has as keys column names and as values a list of requirements the instances have to
     fulfill in this column """
    # only take columns where there are requirements for
    columns_to_check = []

    for col_name in df.columns:
        if col_name in requirements_df.columns:
            columns_to_check.append(col_name)

    dataframe, broken_instances = check_col_consistency(df[columns_to_check], requirements_df[columns_to_check], SCIP=True)
    return broken_instances

def read_data(file_path_data, file_path_requirements):
    data = pd.read_excel(file_path_data)
    requirement_df = pd.read_excel(file_path_requirements, header=0)

    return data, requirement_df

def create_label(data_frame):
    df = data_frame.copy()
    df['Cmp Final solution time (cumulative)'] = 0.0

    int_times = df['Final solution time (cumulative) Int']
    mixed_times = df['Final solution time (cumulative) Mixed']
    diff = (int_times - mixed_times).abs()

    # Condition: nearly equal times
    condition_equal = diff <= 0.5

    # Condition: Mixed ≤ Int
    condition_mixed_better = mixed_times <= int_times
    ratio_mixed_better = (int_times / mixed_times)
    change_mixed_better = (~condition_equal) & condition_mixed_better & (abs(-1 + ratio_mixed_better) > 0.01)
    df.loc[change_mixed_better, 'Cmp Final solution time (cumulative)'] = -1 + ratio_mixed_better[change_mixed_better]

    # Condition: Int < Mixed
    condition_int_better = int_times < mixed_times
    ratio_int_better = (mixed_times / int_times)
    change_int_better = (~condition_equal) & condition_int_better & (abs(1 - ratio_int_better) > 0.01)
    df.loc[change_int_better, 'Cmp Final solution time (cumulative)'] = 1 - ratio_int_better[change_int_better]

    return df

def main(file_path_dataset, file_path_requirements_xlsx, treffmasx, dataset_name, to_csv=True):
    data, requirements = read_data(file_path_dataset, file_path_requirements_xlsx)

    broken_instances_and_reason = find_broken_instances(data, requirements)
    broken_names = [name[0] for name in broken_instances_and_reason]
    clean_data = data[~data['Matrix Name'].isin(broken_names)]
    clean_data = create_label(clean_data)
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
                               # , 'Avg pseudocosts of integer variables Mixed']].copy()

    if to_csv:
        clean_data.to_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffmasx}/CSVs/{dataset_name}_clean_data.csv',
                            index=False)
        clean_feats.to_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/{treffmasx}/CSVs/{dataset_name}_clean_feats.csv',
                            index=False)

    return clean_data, broken_instances_and_reason


main('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/standard_scip/ready_to_ml/all_instances/scip_to_fic.xlsx',
         '/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/standard_scip/ready_to_ml/all_with_feature/scip_requirements.xlsx',
          treffmasx='TreffenMasCinco',
          dataset_name='scip_default',
          to_csv=True)






