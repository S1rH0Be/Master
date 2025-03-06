import pandas as pd
import numpy as np

'''READ AND RENAME COLUMNS'''
# input: df, list. Renames columns of df to strings in lst
def rename_cols(df, int_cols, dbl_cols):
    
    '''Rename Columns'''
    named_df = df
    # Create a mapping for renaming patterns
    replacements = {
        "(Fritz Global PR - Public Discrete Nonconvex GLOBALSPATIALBRANCHIFPREFERINT=1)": "Int",
        "(Fritz Global PR - Public Discrete Nonconvex def)": "Mixed",
        "(Fritz Global PR - Public Discrete Nonconvex def vs Fritz Global PR - Public Discrete Nonconvex GLOBALSPATIALBRANCHIFPREFERINT=1)": "Mixed vs Int",
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

def read_and_rename(file):
    #takes as input name of the data file as a string
    #also fills nan permutation seeds with 0.0
    data_df = pd.read_csv(file)

    int_cols = ['#spatial branching entities fixed (at the root)', '#non-spatial branch entities fixed (at the root)',
    '#nodes in DAG', '#integer violations at root','#nonlinear violations at root']
    dbl_cols = [
    '% vars in DAG (out of all vars)',
    '% vars in DAG unbounded (out of vars in DAG)',
    '% vars in DAG integer (out of vars in DAG)',
    '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)',
    'Avg ticks for propagation + cutting / entity / rootLPticks',
    'Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones)',
    'Avg ticks for solving strong branching LPs for integer branchings (not including infeasible ones)',
    'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones)',
    'Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones)',
    'Avg coefficient spread for convexification cuts']
    #6-9 are set to -1 if they did not happen
    
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
    dropped_instances = []
    
    for column in columns:
        try:
           df[column] = df[column].astype(float)
        except:
            bad_cols.append(column)
    if len(bad_cols) > 0:
        print("Not Float: ", bad_cols)
    return df, bad_cols

def check_datatype(df, int_cols, float_cols, object_cols):
    #input: DataFrame, Lists of strings containing the column names which should be int, float or object type
    #Output: List of column names which do not have the proper dtype
    '''Now its working for this case, to generalize add function to better deal with bad cases'''
    bad_columns = []
    
    for int_col in int_cols:
        if np.dtype(df[int_col])!='int64':
            bad_columns.append(int_col)
            
    for float_col in float_cols:
        if np.dtype(df[float_col])!='float64':
            bad_columns.append(float_col)
    
    for object_col in object_cols:
        if np.dtype(df[object_col])!='object':
            bad_columns.append(object_col)
    if len(bad_columns) > 0:
        print("Bad Columns: ", bad_columns)
    return bad_columns

def datatype_converter(df):
    '''Takes a df and tries to convert each column to the right datatype'''
    #input: Dataframe
    #Output: DataFrame where each row has the right datatype, names of the dropped instances 
    #dropped_instances = []
    
    #remove percent sign in order to be able to work with the percentages as floats
    df.replace('%', '', regex=True, inplace=True)
    
    #columns which had a % sign need to be converted from dtype object to float
    #only columns with cmp in their name need to be converted
    perc_columns = df.filter(like='Cmp')
    #call to_float to convert columns to float columns
    df, bad_cols = to_float(df, perc_columns)
    
    #gather all columns which should be int
    int_cols = ['Matrix Equality Constraints', 'Matrix Quadratic Elements', 'Matrix NLP Formula', 
                'Presolve Columns Mixed', 'Presolve Columns Int', 'Presolve Global Entities Mixed', 
                'Presolve Global Entities Int']
    #add all columns containing # but not 'Cmp' ,because the quantity of variables etc should be int
    #and Cmp are percentages ergo floats
    other_int_cols = [col for col in df.columns if '#' in col and 'Cmp' not in col]
    
    int_cols+= other_int_cols

    #object columns
    object_cols = ['Matrix Name', 'Status Mixed', 'Status Int']
    
    #float cols are the remaining ones
    non_floats = int_cols+object_cols
    float_cols = [col for col in df.columns if col not in non_floats]
    
    bad_cols+=check_datatype(df, int_cols, float_cols, object_cols)
    if len(bad_cols)>0:
        print("Datatype converter: ", bad_cols)
    return df, bad_cols

'''SINGLE COLUMN CHECKER'''

def check_col_consistency(df, requirement_df):

    '''Hier kommen alle requirement checker'''
    '''Am ende werden sie dann aufgerufen'''
    # Function to check and convert column to float
    def convert_column_to_float(df, column_name):
        try:
            # Attempt to convert the column to float
            df[column_name] = df[column_name].astype(float)
            return[]
        except ValueError:
            print(column_name, 'not Float')
            return [column_name]
            
    # Function to check and convert column to integer
    def convert_column_to_integer(df, column_name):
        try:
            # Attempt to convert the column to float
            df[column_name] = df[column_name].astype(int)
            #print(f"Column '{column_name}' successfully converted to float.")
            return []
        except ValueError:
            print(column_name, 'not Int')
            return [column_name]
    
    #function to check if all entries are nonneg
    #i check beforehand that only numerical values are in the tested column
    def is_non_neg(df, column_name):
        minimum = df[column_name].min()
        if minimum<0:
            print(f"Column '{column_name}' has a negative entry.")
            return [column_name]
        return []
    
    #function to check if all entries are positiv
    #i check beforehand that only numerical values are in the tested column
    def is_pos(df, column_name):
        minimum = df[column_name].min()
        if minimum<=0:
            print(f"Column '{column_name}' is not entirely positive.")
            return [column_name]
        return []
    #check if all entries are strings
    def is_string(df, column_name):
        # Check if all entries in the column 'Column Name' are strings
        if not df[column_name].apply(lambda x: isinstance(x, str)).all():
            print(f"Column '{column_name}' is not string.")
            return [column_name]
        return []
    #check if instance terminates in a valid state
    def valid_final_state(df, column_name):
        valid_states = ['Optimal', 'Timeout','Fail','Infeasible']
        if not df[column_name].isin(valid_states).all():
            print(f"Column '{column_name}' has an invalid state.")
            return [column_name]
        return []
    #check if all values are in the intervall [0,1]
    def in_zero_one(df, column_name):
        minimum = df[column_name].min()
        maximum = df[column_name].max()
        # Check if any values are outside the range [0, 1]
        if minimum < 0 or maximum > 1:
            print(f"Column '{column_name}' is not in [0,1].")
            
            # # Find all entries that are outside the range
            # broken_instances = df[(df[column_name] < 0) | (df[column_name] > 1)]
            # #print(broken_instances)
            # #add these instances to the deleted_df just for error search
            # global names_for_deletion_lst
            # print(type(broken_instances['Matrix Name'].unique()))
            # names_for_deletion_lst+=list(broken_instances['Matrix Name'].unique())
            return [column_name]
        return []
    #check if a valid permutation was chosen
    def valid_permutation_seed(df, column_name):
        valid_perms = [0.0,202404273.0,202404274.0]
        if not df[column_name].isin(valid_perms).all():
            print(f"Column '{column_name}' has an invalid permutation seed.")
            return [column_name]
        return []
    #will think about it later
    def perms_have_same_entries(df, column_name):
        #first sort instances by name so that all entries which should be equal are together
        df = df.sort_values(by='Matrix Name')
        for index in range(0,len(df), 3):
    
            if df[column_name].iloc[index]==df[column_name].iloc[index+1]==df[column_name].iloc[index+2]:
                continue
            else:
                print('Permutation not consistent', df['Matrix Name'].iloc[index])
                return [column_name]
        return []
    #some float columns are nonnegative floats but have -1 if this action did not happen
    def float_or_minus_one(df, column_name):
        for value in df[column_name]:
            if (value<0)&(value!=-1):
                print('Not float nor -1')
                return [column_name]
        return []
    
    req_dict = {'Strings': is_string, 'Integers': convert_column_to_integer, 'nonneg': is_non_neg, 
                'Each_permutation_same_entries': perms_have_same_entries, '[Optimal,Timeout,Fail,Infeasible]': valid_final_state,
                'Floats': convert_column_to_float,
                'is_pos': is_pos, '[0,1]': in_zero_one, '[0,202404273,202404274]': valid_permutation_seed,
                'float_or_minus_one': float_or_minus_one}
    #make requirement_df usable
    def clean_requirements(req_df):
        req_df = req_df.drop(req_df.columns[-1], axis=1)
        req_df.columns = df.columns
        return req_df

    
    requirement_df = clean_requirements(requirement_df)
    
    broken_cols = []
    for col in df.columns:
        helper = requirement_df[col]
        requirements = [word for string in helper for word in string.split(', ')]
        for req in requirements:
            broken_cols += req_dict[req](df, col)

    return df, broken_cols






















