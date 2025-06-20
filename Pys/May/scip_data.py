import os
import re
import pandas as pd
import numpy as np
import json

'''
This script reads in the .out files for different runs in SCIP and merges them into one.
The runs are differentiated by the branching rule which is used: 
1. minlp(Default, always branch on integer variables first)
2. mix0 
3. mix1
and by the permutation seed of the instances, to get a bigger data set
'''
def find_no_feature_instances(df):
    no_feat_instances = {}
    for index, row in df.iterrows():
        if row.isna().sum() > 0:
            if row['Matrix Name'] in no_feat_instances:
                no_feat_instances[row['Matrix Name']].append(row['Random Seed Shift'])
            else:
                no_feat_instances[row['Matrix Name']] = [row['Random Seed Shift']]
    all_runs_no_feats = []
    s0_no_feats = []
    s1_no_feats = []
    s2_no_feats = []
    for key, value in no_feat_instances.items():
        if len(value) == 3:
            all_runs_no_feats.append(key)
        else:
            for i in value:
                if i == 0:
                    s0_no_feats.append(key)
                elif i == 1:
                    s1_no_feats.append(key)
                elif i == 2:
                    s2_no_feats.append(key)

    return all_runs_no_feats, s0_no_feats, s1_no_feats, s2_no_feats

def map_raw_scip_to_feature(df, compper_df, operation_dict):
    # Get the list of columns in compper_df
    df_columns = df.columns.tolist()

    for new_col, operation in operation_dict.items():
        # Check if all columns in the operation exist in compper_df
        columns_in_operation = [col for col in df if col in operation]

        for col in columns_in_operation:
            if col not in df_columns:
                raise ValueError(f"Column '{col}' not found in scip raw data")

        # Replace column names in the operation with actual references to the columns in df

        for col in df_columns:
            # Use \b to ensure we're replacing whole words only
            operation = re.sub(rf'\b{col}\b', f"df['{col}']", operation)

        # print('map_raw_scip_to_feature', compper_df['Matrix Equality Constraints'].min())
        try:
            # Evaluate the expression and create the new column in df
            compper_df[new_col] = eval(operation)
            if compper_df[new_col].isna().sum() > 0:
                compper_df[new_col] = compper_df[new_col].replace(np.nan, -1)
        except Exception as e:
            print(f"Error while evaluating operation for {new_col}: {e}")
            compper_df[new_col] = None  # If there's an error, assign NaN or None

    return compper_df

def create_compatible_dataframe(df, fico_only=False):

    name_mapping_fico_only = {'#integer violations at root': 'nintpseudocost',
                    '#nodes in DAG': 'nnonlinearvars+nauxvars',#weiß nicht warum ich nur nauxvars bisher hatte #aber vielleicht sind auch nnonlinearexpr oder nnonconvexexpr interessant
                    'Avg coefficient spread for convexification cuts Mixed': 'sumcoefspreadnonlinrows / nnonlinrows',#aber vielleicht ist auch sumcoefspreadactnonlinrows / nactnonlinrows interessant.
                    'Presolve Global Entities': 'nintegervars',
                    'Presolve Columns': 'ncontinuousvars + nbinaryvars + nintegervars',
                    '#nonlinear violations at root': 'nviolconss', #'nnlviolcands waeren die anzahl der branching candidates fuers spatial branching, also die anzahl von variables in nichtkonvexen termen in verletzen nichtlinear constraints',
                    'Matrix Equality Constraints': 'nlinearequconss + nnonlinearequconss',
                    'Matrix NLP Formula': 'nnonlinearconss',
                    '% vars in DAG (out of all vars)': '(nnonlinearvars + nauxvars) / (ncontinuousvars+nbinaryvars+nintegervars+nauxvars)',
                    '% vars in DAG integer (out of vars in DAG)': '(nnonlinearbinvars + nnonlinearintvars + nintauxvars) / (nnonlinearvars+nauxvars)',
                    '% vars in DAG unbounded (out of vars in DAG)': '(nnonlinearunboundedvars + nunboundedauxvars) / (nnonlinearvars+nauxvars)',
                    '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)': 'nquadexpr / (nquadexpr + nsuperquadexpr)',
                    'Matrix Quadratic Elements': 'nquadcons',
                    'Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed': 'sumintpseudocost / nintpseudocost',
                    }

    name_mapping_with_more_scip_features = {'#integer violations at root': 'nintpseudocost',
                                            '#nodes in DAG': 'nnonlinearvars+nauxvars',
                                            'Avg coefficient spread for convexification cuts Mixed': 'sumcoefspreadnonlinrows / nnonlinrows',
                                            'Presolve Global Entities': 'nintegervars',
                                            'Presolve Columns': 'ncontinuousvars + nbinaryvars + nintegervars',
                                            '#nonlinear violations at root': 'nviolconss',
                                            'Avg strong branching iterations in root': 'avgstrongbranchrootiter',
                                            'Matrix Equality Constraints': 'nlinearequconss + nnonlinearequconss',
                                            'Matrix NLP Formula': 'nnonlinearconss',
                                            'Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed': 'sumintpseudocost / nintpseudocost',
                                            '% vars in DAG (out of all vars)': '(nnonlinearvars + nauxvars) / (ncontinuousvars+nbinaryvars+nintegervars+nauxvars)',
                                            '% vars in DAG integer (out of vars in DAG)': '(nnonlinearbinvars + nnonlinearintvars + nintauxvars) / (nnonlinearvars+nauxvars)',
                                            '% vars in DAG unbounded (out of vars in DAG)': '(nnonlinearunboundedvars + nunboundedauxvars) / (nnonlinearvars+nauxvars)',
                                            '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)': 'nquadexpr / (nquadexpr + nsuperquadexpr)',
                                            'Matrix Quadratic Elements': 'nquadcons'
                                            }

    if fico_only:
        compa_df = pd.DataFrame(columns= ['Matrix Name', 'Random Seed Shift']+
                                         [name for name in name_mapping_fico_only.keys()]+
                                         ['Status Mixed', 'Status Int', 'Final solution time (cumulative) Mixed',
                                          'Final solution time (cumulative) Int', 'Cmp Final solution time (cumulative)',
                                          'Virtual Best'], index=df.index)

        for col_name in compa_df.columns:
            if col_name not in name_mapping_fico_only:
                compa_df[col_name] = df[col_name]
            elif name_mapping_fico_only[col_name] in df.columns:
                compa_df[col_name] = df[name_mapping_fico_only[col_name]]
        compa_df = map_raw_scip_to_feature(df, compa_df, name_mapping_fico_only)
    else:
        compa_df = pd.DataFrame(columns=['Matrix Name', 'Random Seed Shift'] +
                                        [name for name in name_mapping_with_more_scip_features.keys()] +
                                        ['Status Mixed', 'Status Int', 'Final solution time (cumulative) Mixed',
                                         'Final solution time (cumulative) Int', 'Cmp Final solution time (cumulative)',
                                         'Virtual Best'], index=df.index)
        for col_name in compa_df.columns:
            if col_name not in name_mapping_with_more_scip_features:
                compa_df[col_name] = df[col_name]
            elif name_mapping_with_more_scip_features[col_name] in df.columns:
                compa_df[col_name] = df[name_mapping_with_more_scip_features[col_name]]
        compa_df = map_raw_scip_to_feature(df, compa_df, name_mapping_with_more_scip_features)
    return compa_df

def get_name(string: str) -> str|None:
    matrix_name_match = re.search(r'/([^/]+?)\.[^.]+', string)
    if matrix_name_match:
        matrix_name = matrix_name_match.group(1)
        return matrix_name
    return None

def get_status(string: str) -> str|None:
    status = string.split(':')[1].strip()
    if 'optimal' in status:
        return 'optimal'
    elif 'infeasible' in status:
        return 'infeasible'
    elif 'time limit' in status:
        return 'timeout'
    elif 'gap limit' in status:
        return 'gap limit'
    return None

def get_randomseed(string: str) -> int|None:
    match = re.search(r'=\s*([\d.]+)', string)  # Extracts value after '='
    if match:
        return int(match.group(1))
    return None

def get_solving_time(string: str) -> str:
    return string.split(':')[1].strip()

def extract_instance_data_mix0(file_path):
    rows = []

    with open(file_path, 'r') as file:

        matrix_name = None
        features = None
        status = None
        solving_time = None
        random_seed_shift = None


        for line in file:
            line = line.strip()

            # Extract matrix name (the string between the last slash and the next dot)
            if line.startswith('SCIP> read'):
                matrix_name = get_name(line)

            # Extract the features dictionary after 'FEATURES:'
            if line.startswith('FEATURES:'):
                try:
                    features = json.loads(line.split('FEATURES:')[1].strip())
                except json.JSONDecodeError:
                    features = None  # Keep as None if JSON is invalid

            # Extract Status
            if line.startswith('SCIP Status'):
                status = get_status(line)

            # Extract solving time
            if line.startswith('Solving Time (sec)'):
                solving_time = get_solving_time(line)

            # Extract Random Seed Shift
            if "randomization/randomseedshift" in line:
                random_seed_shift = get_randomseed(line)

            # If "SCIP> quit" is found, store the instance data
            if line.startswith("SCIP> quit"):
                # Ensure default values for missing data
                if random_seed_shift is None:
                    random_seed_shift = 0

                row = {'Matrix Name': matrix_name or "Unknown", 'Random Seed Shift': random_seed_shift}
                row.update(features if features else {})
                row['Status Mixed'] = status or "Unknown"
                row['Final solution time (cumulative) Mixed'] = float(solving_time) or "Unknown"

                rows.append(row)

                # Reset for next instance
                matrix_name = None
                features = None
                status = None
                solving_time = None
                random_seed_shift = None

    # Create the pandas DataFrame
    df = pd.DataFrame(rows)
    return df

def extract_instance_data_minlp(file_path):
    # Create an empty list to store the rows of the dataframe
    rows = []

    # Open the .out file
    with open(file_path, 'r') as file:
      # Read the whole file line by line
        matrix_name = None
        status = None
        solving_time = None
        random_seed_shift = None  # New column

        # Iterate over each line in the file
        for line in file:
            line = line.strip()

            # Extract matrix name (the string between the last slash and the next dot)
            if line.startswith('SCIP> read'):
                matrix_name = get_name(line)

            # Extract Status
            if line.startswith('SCIP Status'):
                status = get_status(line)

            # Extract solving time
            if line.startswith('Solving Time (sec)'):
                solving_time = get_solving_time(line)

            # Extract Random Seed Shift
            if "randomization/randomseedshift" in line:
                random_seed_shift = get_randomseed(line)

            # If "SCIP> quit" is found, store the instance data
            if line.startswith("SCIP> quit"):
                # Ensure default values for missing data
                if random_seed_shift is None:
                    random_seed_shift = 0

                row = {'Matrix Name': matrix_name, 'Random Seed Shift': random_seed_shift,
                       'Status Int': status,
                       'Final solution time (cumulative) Int': float(solving_time)}

                rows.append(row)

                # Reset for next instance
                matrix_name = None
                status = None
                solving_time = None
                random_seed_shift = None

    # Create the pandas DataFrame
    df = pd.DataFrame(rows)
    return df

def calculate_label(df):
    df.insert(len(df.columns) - 1, 'Virtual Best', "")
    df.insert(len(df.columns) - 1, 'Cmp Final solution time (cumulative)', 0.0)


    for index, row in df.iterrows():
        if row['Final solution time (cumulative) Mixed'] >= row['Final solution time (cumulative) Int']:
            df.loc[index, 'Cmp Final solution time (cumulative)'] = 1-(row['Final solution time (cumulative) Mixed']/row['Final solution time (cumulative) Int'])
            df.loc[index, 'Virtual Best'] = row['Final solution time (cumulative) Int']
        else:
            df.loc[index, 'Cmp Final solution time (cumulative)'] = (row['Final solution time (cumulative) Int']/row['Final solution time (cumulative) Mixed'])-1
            df.loc[index, 'Virtual Best'] = row['Final solution time (cumulative) Mixed']

    for index, row in df.iterrows():
        if abs(row['Final solution time (cumulative) Mixed'] - row['Final solution time (cumulative) Int']) <= 0.5:
            df.loc[index, 'Cmp Final solution time (cumulative)'] = 0.0

        if abs(row['Cmp Final solution time (cumulative)']) <= 0.01:
            # 1% doesnt matter as well
            df.loc[index, 'Cmp Final solution time (cumulative)'] = 0.0
    return df

def process_directory(directory):
    file_groups = {}
    merge_seeds_list = []

    for filename in os.listdir(directory):
        if filename.endswith(".out"):
            match = re.search(r'(mix0|minlp)[_-]s(\d+)\.out$', filename)
            if match:
                category, group_id = match.groups()
                file_groups.setdefault(group_id, {}).update({category: os.path.join(directory, filename)})

    for group_id, files in file_groups.items():
        #read in data for the minlp and mix0_ runs
        mix0_df = extract_instance_data_mix0(files.get('mix0')) if 'mix0' in files else None

        if mix0_df is not None:
            mix0_df = mix0_df.loc[:, ~mix0_df.columns.str.endswith('_descr')]
            mix0_df = mix0_df.drop('Random Seed Shift', axis=1)

        mix0_df.to_csv(f'{directory}/mixo_s{group_id}.csv', index=False)
        minlp_df = extract_instance_data_minlp(files.get('minlp')) if 'minlp' in files else None


        merged_df = pd.merge(mix0_df, minlp_df, on="Matrix Name")
        columns_list = merged_df.columns.tolist()
        reordered_columns = columns_list[:-4]+['Status Int', 'Final solution time (cumulative) Mixed', 'Final solution time (cumulative) Int', 'Random Seed Shift']
        merged_df = merged_df[reordered_columns]
        merged_df = calculate_label(merged_df)
        merge_seeds_list.append(merged_df)
    # merge the two dataframes for minlp and mix0
    complete_df = pd.concat(merge_seeds_list, ignore_index=True).sort_values(by=['Matrix Name', 'Random Seed Shift'])

    return complete_df

def read_in_and_call_process(data_set: str, fico=True, to_csv=True):
    directory_path = f"/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/{data_set}/Outs"
    if data_set == '345':
        directory_path = "/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/345/Outs"

    stefans_data_merged = process_directory(directory_path)

    # delete all instances and their permutations if they have np.nan values
    # Identify matrix names that contain NaN values
    matrices_with_nan = stefans_data_merged[stefans_data_merged.isna().any(axis=1)]['Matrix Name'].unique()

    # Filter out all rows with those matrix names
    stefans_data_merged_all_have_features = stefans_data_merged[~stefans_data_merged['Matrix Name'].isin(matrices_with_nan)]

    if to_csv:
        # save only df where all instances without features are deleted
        stefans_data_merged_all_have_features.to_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/{data_set}/complete/scip_{data_set}_raw.csv',
                               index=False)
        stefan_data_reduced_cols_no_nan = create_compatible_dataframe(stefans_data_merged_all_have_features)
        stefan_data_reduced_cols_no_nan.to_csv(
            f"/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/{data_set}/complete/scip_{data_set}_ready_to_ml.csv",
            index=False)

        scip_features_df = stefan_data_reduced_cols_no_nan.drop(['Matrix Name', 'Random Seed Shift', 'Status Mixed',
                                                                 'Status Int', 'Final solution time (cumulative) Mixed',
                                                                 'Final solution time (cumulative) Int',
                                                                 'Cmp Final solution time (cumulative)',
                                                                 'Virtual Best'], axis=1)
        scip_features_df.to_csv(
            f"/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/{data_set}/only_features/scip_{data_set}_features.csv",
            index=False)

    if fico:
        # all instances
        scip_to_fic_df = create_compatible_dataframe(stefans_data_merged_all_have_features, fico_only=True)
        scip_to_fic_df.to_csv(f"/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/{data_set}/complete/scip_{data_set}_fico_compatible.csv",
                                index=False)

        stefans_feats = scip_to_fic_df[['#integer violations at root',
               '#nodes in DAG',
               'Avg coefficient spread for convexification cuts Mixed',
               'Presolve Global Entities', 'Presolve Columns',
               '#nonlinear violations at root', 'Matrix Equality Constraints',
               'Matrix NLP Formula', '% vars in DAG (out of all vars)',
               '% vars in DAG integer (out of vars in DAG)',
               'Matrix Quadratic Elements',
               '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)',
               '% vars in DAG unbounded (out of vars in DAG)']].copy()

        stefans_feats.to_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/{data_set}/only_features/scip_{data_set}_fico_features.csv',
                                   index=False)



# read_in_and_call_process(data_set='default')
read_in_and_call_process(data_set='345')

# TODO: Wie geh uich mit GAOP limit um? ist das optimal oder timeout oder was ganz anderes? Ne schon optimal