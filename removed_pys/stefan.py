"""
import re
import pandas as pd
import json

def extract_instance_data(file_path):
    # Create an empty list to store the rows of the dataframe
    rows = []

    # Open the .out file
    with open(file_path, 'r') as file:
        # Read the whole file line by line
        instance_data = {}
        matrix_name = None
        features = None
        status = None
        solving_time = None
        random_seed_shift = None  # New column

        # Iterate over each line in the file
        for line in file:
            line = line.strip()

            # Extract matrix name (the string between the last slash and the next dot)
            if line.startswith('SCIP> read'):
                matrix_name_match = re.search(r'\/([^\/]+?)\.[^.]+', line)
                if matrix_name_match:
                    matrix_name = matrix_name_match.group(1)

            # Extract the features dictionary after 'FEATURES:'
            if line.startswith('FEATURES:'):
                try:
                    features = json.loads(line.split('FEATURES:')[1].strip())
                except json.JSONDecodeError:
                    continue  # skip line if it's not a valid JSON format

            # Extract SCIP status
            if line.startswith('SCIP Status'):
                status = line.split(':')[1].strip()

            # Extract solving time
            if line.startswith('Solving Time (sec)'):
                solving_time = line.split(':')[1].strip()

            # Extract Random Seed Shift
            if "randomization/randomseedshift" in line:
                match = re.search(r'=\s*([\d.]+)', line)  # Extracts value after '='
                if match:
                    random_seed_shift = int(match.group(1))

            # If we've found the necessary information for one instance, store it and reset
            if matrix_name and features and status and solving_time and random_seed_shift is not None:
                row = {'Matrix Name': matrix_name}
                row['Random Seed Shift'] = random_seed_shift  # Add random seed shift
                row.update(features)  # Add all features as columns
                row['SCIP Status'] = status
                row['Final solution time (cumulative)'] = solving_time

                # Add the row to the list
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

def map_scip_to_xpress(df, compper_df, operation_dict):
    # Get the list of columns in compper_df
    df_columns = df.columns.tolist()

    for new_col, operation in operation_dict.items():
        # Check if all columns in the operation exist in compper_df
        columns_in_operation = [col for col in df if col in operation]

        for col in columns_in_operation:
            if col not in df_columns:
                raise ValueError(f"Column '{col}' not found in stefan_df")

        # Replace column names in the operation with actual references to the columns in df
        for col in df_columns:
            # Use \b to ensure we're replacing whole words only
            operation = re.sub(rf'\b{col}\b', f"df['{col}']", operation)

        try:
            # Evaluate the expression and create the new column in df
            compper_df[new_col] = eval(operation)
        except Exception as e:
            print(f"Error while evaluating operation for {new_col}: {e}")
            compper_df[new_col] = None  # If there's an error, assign NaN or None

    return compper_df

def create_compatible_dataframe(df):
    name_mapping = {'#integer violations at root': 'nintpseudocost',
                    '#nodes in DAG': 'nexpr', #aber vielleicht sind auch nnonlinearexpr oder nnonconvexexpr interessant
                    'Avg coefficient spread for convexification cuts Mixed': 'sumcoefspreadnonlinrows / nnonlinrows',#aber vielleicht ist auch sumcoefspreadactnonlinrows / nactnonlinrows interessant.
                    'Presolve Global Entities': 'nintegervars',
                    'Presolve Columns': 'ncontinuousvars + nbinaryvars + nintegervars',
                    '#nonlinear violations at root': 'nviolconss', #'nnlviolcands waeren die anzahl der branching candidates fuers spatial branching, also die anzahl von variables in nichtkonvexen termen in verletzen nichtlinear constraints',
                    #'Avg work for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed': 'avgstrongbranchrootiter ist die Anzahl der LP iter, but including infeasible ones',
                    'Matrix Equality Constraints': 'nlinearequconss + nnonlinearequconss',
                    'Matrix NLP Formula': 'nnonlinearconss',
                    #'Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed': 'sumintpseudocost / nintpseudocost kann ich als Alternative anbieten',
                    '% vars in DAG (out of all vars)': 'nnonlinearvars/ (ncontinuousvars + nbinaryvars + nintegervars)', #aber vielleicht ist nnonconvexvars besser',
                    '% vars in DAG integer (out of vars in DAG)': '(nnonlinearbinvars + nnonlinearintvars) / nnonlinearvars', #oder es ist (nnonconvexbinvars + nnonconvexintvars) / nnonconvexvars besser',
                    '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)': 'nquadexpr / (nquadexpr + nsuperquadexpr)',
                    '% vars in DAG unbounded (out of vars in DAG)': 'nnonlinearunboundedvars / nnonlinearvars', #, aber vielleicht ist nnonconvexunboundedvars / nnonconvexvars besser',
                    'Matrix Quadratic Elements': 'nquadcons'
                    }

    compa_df = pd.DataFrame(columns= ['Matrix Name', 'Random Seed Shift']+[name for name in name_mapping.keys()]+['SCIP Status Mixed', 'SCIP Status Int', 'Final solution time (cumulative) Mixed', 'Final solution time (cumulative) Int'], index=df.index)

    for col_name in compa_df.columns:
        if col_name not in name_mapping:
            compa_df[col_name] = df[col_name]
        elif name_mapping[col_name] in df.columns:
            compa_df[col_name] = df[name_mapping[col_name]]

    compa_df = map_scip_to_xpress(df, compa_df, name_mapping)

    return compa_df

complete_df = pd.read_excel("/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/stefan_outs/stefan_merged_complete.xlsx")
marvin_compper = create_compatible_dataframe(complete_df)
marvin_compper.to_excel("/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/stefan_outs/complete_compper.xlsx",
                        index=False)


stefans_feats = marvin_compper[['#integer violations at root',
       '#nodes in DAG',
       'Avg coefficient spread for convexification cuts Mixed',
       'Presolve Global Entities', 'Presolve Columns',
       '#nonlinear violations at root', 'Matrix Equality Constraints',
       'Matrix NLP Formula', '% vars in DAG (out of all vars)',
       '% vars in DAG integer (out of vars in DAG)',
       'Matrix Quadratic Elements',
       '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)']].copy()
# '% vars in DAG unbounded (out of vars in DAG)' contains just one nonzero entry => Out

stefans_feats.to_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/stefan_outs/stefans_feats.xlsx',
                           index=False)


# Feats I needs:
# Forest:
# Avg coefficient spread for convexification cuts Mixed	Check
# Presolve Global Entities	Check
# LinReg:
# Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed
#nodes in DAG
#nonlinear violations at root
"""
