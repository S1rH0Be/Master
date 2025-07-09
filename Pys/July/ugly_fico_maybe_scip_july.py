import pandas as pd

DEBUG = False
'''READ AND RENAME COLUMNS'''
# input: df, list. Renames columns of df to strings in lst
def rename_cols(df, int_cols, dbl_cols):

    named_df = df
    # Create a mapping for renaming patterns
    replacements = {
        "(Fritz Global PR - Public Discrete Nonconvex GLOBALSPATIALBRANCHIFPREFERINT=1)": "Int",
        "(Fritz Global PR - Public Discrete Nonconvex def)": "Mixed",
        "(Fritz Global PR - Public Discrete Nonconvex def vs Fritz Global PR - Public Discrete Nonconvex GLOBALSPATIALBRANCHIFPREFERINT=1)": "",
        "(User-defined attribute)": "",
        "# ":"#",
        "  ":" "
    }
    #replace _ with ' ' in the column names
    named_df.columns = named_df.columns.str.replace('_', ' ')
    #change official name of rules to mixed and int, and also replace the rest of the replacment dict
    def rename_column(col_name):
        for pattern, replacement in replacements.items():
            col_name = col_name.replace(pattern, replacement)
            col_name.strip()
        return col_name
    #now we actually change the column names
    named_df.columns = [rename_column(col).strip() for col in named_df.columns]

    #change all Integer x and Double x column names to what they actually are
    for i in range(len(max(int_cols, dbl_cols))):#take len of longer list
        #try because the shorter list will give a out of bounds error
        try:
            #(?!\d) makes sure that no number follows i+1 directly
            #e.g. Integer 1 doesnt find Integer 10
            named_df.columns =named_df.columns.str.replace(r"Integer "+str(i+1)+r"(?!\d)", int_cols[i], regex=True)
        except:
            if i >=len(int_cols):
                pass
            else:
                print("Somethings wrong at rename Integer")
        try:
            named_df.columns =named_df.columns.str.replace(r"Double "+str(i+1)+r"(?!\d)", dbl_cols[i], regex=True)

            
        except:
            if i >=len(int_cols):
                  pass
            else:
                print("Somethings wrong at rename Double")
    return named_df

def read_and_rename(path_to_file:str, int_cols:list, dbl_cols:list):
    #takes as input name of the data file as a string
    #also fills nan permutation seeds with 0.0
    data_df = pd.read_csv(path_to_file)
    
    renamed_df = rename_cols(data_df, int_cols, dbl_cols)
    #unnamed contains just the indices of rows
    renamed_df.drop('Unnamed: 0', axis=1, inplace=True)
    #fillna for permutation seeds
    perm_cols = ['permutation seed Mixed', 'permutation seed Int']
    
    renamed_df[perm_cols] = renamed_df[perm_cols].fillna(0.0)
    renamed_df.sort_values(by=['Matrix Name', 'permutation seed Mixed'], inplace = True, ascending=False)

    return renamed_df

'''DTYPE CONVERTER AND CHECKER'''

def to_float(df, columns):
    #input: Dataframe, List of column names as strings 
    #output: DataFrame, List of dropped instances because not float
    bad_cols = []
    for column in columns:
        try:
           df[column] = df[column].astype(float)
        except:
            bad_cols.append(column)
    if len(bad_cols) > 0:
        print("Not Float: ", bad_cols)
    return df, bad_cols

# TODO: integrate check_datatype into  datatype_converter
def check_datatype(df):
    #input: DataFrame
    #Output: List of matrix names which do not have the proper dtype

    # all columns which should be int
    # TODO: Add scip only int columns
    int_cols = ['Matrix Equality Constraints', 'Matrix Quadratic Elements', 'Matrix NLP Formula',
                'Presolve Columns Mixed', 'Presolve Columns Int', 'Presolve Global Entities Mixed',
                'Presolve Global Entities Int']
    # add all columns containing # but not 'Cmp', because the quantity of variables etc should be int and Cmp are percentages ergo floats
    other_int_cols = [col for col in df.columns if '#' in col and 'Cmp' not in col]
    int_cols += other_int_cols

    # object columns
    # TODO: Add scip only object columns
    object_cols = ['Matrix Name', 'Status Mixed', 'Status Int']

    # float cols are the remaining ones
    non_floats = int_cols + object_cols
    float_cols = [col for col in df.columns if col not in non_floats]
    # In bad_instances we store the Matrix Name for instances with one entry with wrong datatype
    bad_instances = []
    # First check if everything which should be an integer is an integer
    for int_col in int_cols:
        if int_col in df.columns:
            bad_rows = df[~df[int_col].apply(lambda x: isinstance(x, int))]['Matrix Name']
            bad_instances.extend(bad_rows.tolist())
        else:
            pass
    # Then check if everything which should be a float is a float
    for float_col in float_cols:
        if float_col in df.columns:
            bad_rows = df[~df[float_col].apply(lambda x: isinstance(x, float))]['Matrix Name']
            bad_instances.extend(bad_rows.tolist())
        else:
            pass
    # Finally check if everything which should be an object is an object
    for object_col in object_cols:
        if object_col in df.columns:
            bad_rows = df[~df[object_col].apply(lambda x: isinstance(x, str))]['Matrix Name']
            bad_instances.extend(bad_rows.tolist())
        else:
            pass

    if len(bad_instances) > 0:
        print("Wrong Datatype: ", bad_instances)

    return bad_instances

def datatype_converter(df):
    """Takes a df and tries to convert each column to the right datatype"""
    bad_instances = []
    # remove percent sign in order to be able to work with the percentages as floats
    df.replace('%', '', regex=True, inplace=True)
    # columns which had a % sign need to be converted from dtype object to float
    # only columns with cmp in their name need to be converted
    perc_columns = df.filter(like='Cmp')
    # call to_float to convert columns to float columns
    df, bad_cols = to_float(df, perc_columns)
    
    bad_instances += check_datatype(df)

    if len(bad_instances)>0:
        print("Datatype converter: ", bad_instances)
    return df, bad_cols

'''SINGLE COLUMN CHECKER'''

def check_col_consistency(df, requirement_df, SCIP=False):

    """Hier kommen alle requirement checker"""
    '''Am ende werden sie dann aufgerufen'''
    # Function to check and convert column to float
    def convert_column_to_float(data_frame:pd.DataFrame, column_name:str):
        broken_instances = []

        for index, value in data_frame[column_name].items():
            try:
                float(value)  # Try converting to float
            except ValueError:
                broken_instances.append((data_frame.loc[index, 'Matrix Name'], 'not float'))  # Collect corresponding 'Matrix Name'

        if not broken_instances:
            data_frame.loc[:, column_name] = data_frame[column_name].astype(float)  # Convert if all values are valid

        return broken_instances

    # Function to check and convert column to integer
    def convert_column_to_integer(data_frame:pd.DataFrame, column_name:str):
        broken_instances = []

        for index, value in data_frame[column_name].items():
            try:
                int(value)  # Try converting to integer
            except ValueError:
                broken_instances.append((data_frame.loc[index, 'Matrix Name'], 'not int'))  # Collect corresponding 'Matrix Name'

        if not broken_instances:
            data_frame.loc[:,column_name] = data_frame[column_name].astype(int)  # Convert if all values are valid

        return broken_instances

    # function to check if all entries are nonneg
    # i check beforehand that only numerical values are in the tested column
    def is_non_neg(data_frame:pd.DataFrame, column_name:str):
        minimum = data_frame[column_name].min()
        broken_instances = []
        # Check if any values are outside the range [0, 1]
        if minimum < 0:
            for index, row in data_frame.iterrows():
                if row[column_name] < 0:
                    if DEBUG:
                        print(f"Instance '{data_frame.loc[index, 'Matrix Name']}' in '{column_name}' is negative.")
                    broken_instances.append((row['Matrix Name'], 'Negative'))
            return broken_instances
        return broken_instances

    # function to check if all entries are positiv
    # i check beforehand that only numerical values are in the tested column
    def is_pos(data_frame:pd.DataFrame, column_name:str):
        minimum = data_frame[column_name].min()
        broken_instances = []
        # Check if any values are outside the range [0, 1]
        if minimum <= 0:
            for index, row in data_frame.iterrows():
                if row[column_name] <= 0:
                    if DEBUG:
                        print(f"Instance '{data_frame.loc[index, 'Matrix Name']}' in '{column_name}' is nonpositive.")
                    broken_instances.append((row['Matrix Name'], 'nonpositive'))
            return broken_instances
        return broken_instances

    # check if all entries are strings
    def is_string(data_frame:pd.DataFrame, column_name:str):
        broken_instances = []
        for index, row in data_frame.iterrows():
            if type(row[column_name]) != str:
                if DEBUG:
                    print(f"Instance '{row['Matrix Name']}' is not a string.")
                broken_instances.append((row['Matrix Name'], 'Not string'))
        return broken_instances

    # check if an instance from fico terminates in a valid state
    def valid_final_state_fico(data_frame:pd.DataFrame, column_name:str, is_scip_set:bool=False):
        valid_states = ['Optimal', 'Timeout', 'Fail', 'Infeasible']
        if is_scip_set:
            valid_states.append('gap limit')
            if 'Fail' in valid_states:
                valid_states.remove('Fail')
        broken_instances = []
        for index, row in data_frame.iterrows():
            if row[column_name] not in valid_states:
                if DEBUG:
                    print(f"Instance '{row['Matrix Name']}' has an invalid final state.")
                broken_instances.append((row['Matrix Name'], 'Invalid final state'))
        return broken_instances

    # check if an instance from scip terminates in a valid state
    def valid_final_state_scip(data_frame:pd.DataFrame, column_name:str):
        valid_states = ['optimal', 'timeout','gap limit','infeasible']
        broken_instances = []
        for index, row in data_frame.iterrows():
            if row[column_name] not in valid_states:
                if DEBUG:
                    print(f"Instance '{row['Matrix Name']}' has an invalid final state.")
                broken_instances.append((row['Matrix Name'], 'Invalid final state'))
        return broken_instances

    # check if all values are in the intervall [0,1]
    def in_zero_one(data_frame:pd.DataFrame, column_name:str):
        minimum = data_frame[column_name].min()
        maximum = data_frame[column_name].max()
        broken_instances = []
        # Check if any values are outside the range [0, 1]
        if minimum < 0 or maximum > 1:
            for index, row in data_frame.iterrows():
                if (row[column_name] < 0) | (row[column_name] > 1):
                    if DEBUG:
                        print(f"Instance '{data_frame.loc[index, 'Matrix Name']}' in '{column_name}' is not in [0,1].")
                    broken_instances.append((row['Matrix Name'], 'Not in [0,1]'))
            return broken_instances
        return broken_instances

    # check if a valid permutation from Fico Xpress was chosen
    def valid_permutation_seed_fico(data_frame:pd.DataFrame, column_name:str):
        valid_perms = [0.0, 202404273.0, 202404274.0]
        broken_instances = []
        for index, row in data_frame.iterrows():
            if row[column_name] not in valid_perms:
                if DEBUG:
                    print(f"Instance '{row['Matrix Name']}' has an invalid permutation seed.")
                broken_instances.append((row['Matrix Name'], 'Invalid permutation seed'))
        return broken_instances

    # check if a valid permutation from Scip was chosen
    def valid_permutation_seed_scip(data_frame:pd.DataFrame, column_name:str):
        valid_perms = [0, 1, 2, 3, 4, 5]
        broken_instances = []
        for index,row in data_frame.iterrows():
            if row[column_name] not in valid_perms:
                if DEBUG:
                    print(f"Instance '{row['Matrix Name']}' has an invalid permutation seed.")
                broken_instances.append((row['Matrix Name'], 'Invalid permutation seed'))
        return broken_instances

    # will think about it later
    def perms_have_same_entries(data_frame:pd.DataFrame, column_name:str):
        # first sort instances by name so that all entries which should be equal are together
        sorted_df = data_frame.sort_values(by='Matrix Name')
        broken_instances = []

        for index in range(0,len(data_frame), 3):
            # check if three consecutive instances have the same entries
            if sorted_df[column_name].iloc[index]==sorted_df[column_name].iloc[index+1]==sorted_df[column_name].iloc[index+2]:
                continue
            else:
                if DEBUG:
                    print('Permutation not consistent', column_name, sorted_df['Matrix Name'].iloc[index])
                broken_instances.append((sorted_df['Matrix Name'].iloc[index], f'Permutation not consistent: {column_name}'))
        return broken_instances

    # some float columns are nonnegative floats but have -1 if this action did not happen
    def nonneg_or_minus_one(data_frame:pd.DataFrame, column_name:str):
        broken_instances = []
        for index in data_frame[column_name].index:
            if (data_frame[column_name].loc[index]<0) & (data_frame[column_name].loc[index]!=-1):
                if DEBUG:
                    print(f"Instance '{data_frame['Matrix Name'].loc[index]}' not a float or -1.")
                broken_instances.append((data_frame['Matrix Name'].loc[index], 'Not float nor -1'))
        return broken_instances

    # each instance name should appear exactly three times; once for each permutaion
    def each_name_thrice(data_frame:pd.DataFrame, column_name:str):
        broken_instances = []

        matrix_names = data_frame[column_name].unique()
        for matrix_name in matrix_names:
            if len(data_frame[data_frame[column_name]==matrix_name])!=3:
                broken_instances.append((matrix_name, 'Not 3 times'))
        return broken_instances

    # if #nodes in DAG is 0 we don't have nonlinearities => delete instances
    def linear_problems(data_frame:pd.DataFrame, column_name:str):
        broken_instances = []

        for index, row in data_frame.iterrows():
            if row[column_name] == 0.0:
                broken_instances.append(row['Matrix Name'])
        if DEBUG:
            print(f'Number of Linear Problems: {len(set(broken_instances))}')
        return broken_instances



    req_dict = {'Strings': is_string, 'Integers': convert_column_to_integer, 'nonneg': is_non_neg, 
                'Each_permutation_same_entries': perms_have_same_entries, '[Optimal,Timeout,Fail,Infeasible]': valid_final_state_fico,
                '[optimal,gap limit,timeout,infeasible]': valid_final_state_scip,
                'Floats': convert_column_to_float,
                'is_pos': is_pos, '[0,1]': in_zero_one, '[0,202404273,202404274]': valid_permutation_seed_fico,
                '[0,1,2]':valid_permutation_seed_scip,
                'nonneg_or_minus_one': nonneg_or_minus_one, 'Each_name_3_Times': each_name_thrice,
                'linear_problems': linear_problems}

    #make requirement_df usable
    def clean_requirements(data_frame:pd.DataFrame, req_df:pd.DataFrame, scip=False):
        if not scip:
            req_df = req_df.drop(req_df.columns[-1], axis=1)
        req_df.columns = data_frame.columns
        return req_df

    requirement_df = clean_requirements(df, requirement_df, SCIP)
    
    broken_instances_with_reason = []
    for col in df.columns:
        helper = requirement_df[col]
        requirements = [word.strip() for string in helper for word in string.split(', ')]
        for req in requirements:
            broken_instances_with_reason += req_dict[req](df, col)

    # TODO: Add it to requirement_df
    # broken_instances_with_reason += linear_problems(df)

    return df, broken_instances_with_reason






















