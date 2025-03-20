from visualize_data_and_results_last_bt import *
from das_ist_die_richtige_regression import *
from scip_data_cleaner import *

import ast

def regression():
    imputators = ['median', 'constant', 'median', 'mean']
    scaling = [QuantileTransformer(n_quantiles=100, output_distribution="normal", random_state=42),
               PowerTransformer('yeo-johnson')]
    regression_models = {"LinearRegression": LinearRegression(),
                         "RandomForest": RandomForestRegressor(n_estimators=100, random_state=729154)}

    all_features = ['Matrix Equality Constraints', 'Matrix Quadratic Elements',
                    'Matrix NLP Formula', 'Presolve Columns', 'Presolve Global Entities',
                    '#nodes in DAG', '#integer violations at root',
                    '#nonlinear violations at root', '% vars in DAG (out of all vars)',
                    '% vars in DAG unbounded (out of vars in DAG)',
                    '% vars in DAG integer (out of vars in DAG)',
                    '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)',
                    'Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
                    'Avg ticks for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
                    'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Mixed',
                    'Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
                    '#spatial branching entities fixed (at the root) Mixed',
                    'Avg coefficient spread for convexification cuts Mixed']
    #
    # integer_feature = ['Matrix Equality Constraints', 'Matrix Quadratic Elements',
    #        'Matrix NLP Formula', 'Presolve Columns', 'Presolve Global Entities',
    #        '#nodes in DAG', '#integer violations at root',
    #        '#nonlinear violations at root', '#spatial branching entities fixed (at the root) Mixed']
    #
    # float_feature = ['% vars in DAG (out of all vars)',
    #        '% vars in DAG unbounded (out of vars in DAG)',
    #        '% vars in DAG integer (out of vars in DAG)',
    #        '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)',
    #        'Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
    #        'Avg ticks for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
    #        'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Mixed',
    #        'Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
    #        'Avg coefficient spread for convexification cuts Mixed']

    t3_feats_combined = ['Avg coefficient spread for convexification cuts Mixed',
                         'Presolve Global Entities',
                         '#integer violations at root']

    t3_linear = ['#nodes in DAG',
                 'Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
                 '#integer violations at root']

    t2_linear = ['#nodes in DAG',
                 'Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed']

    t1_linear = ['#nodes in DAG']

    t2_forest = ['Avg coefficient spread for convexification cuts Mixed',
                 'Presolve Global Entities']

    t3_forest = ['Avg coefficient spread for convexification cuts Mixed',
                 'Presolve Global Entities',
                 '#integer violations at root']

    setup_for_now = ['hundred', regression_models,
                     [QuantileTransformer(n_quantiles=100, output_distribution="normal", random_state=42)],
                     ['median'], all_features, 1000]

    preset_everything = ['hundred', regression_models, scaling, imputators, all_features, 1000]

    preset_combined_t3 = ['hundred', regression_models, scaling, imputators, t3_feats_combined, 1000]

    preset_linear_t3 = ['hundred', regression_models, scaling, imputators, t3_linear, 1000]

    preset_linear_t2 = ['hundred', regression_models, scaling, imputators, t2_linear, 1000]

    preset_linear_t1 = ['hundred', regression_models, scaling, imputators, t1_linear, 1000]

    preset_forest_t2 = ['hundred', regression_models, scaling, imputators, t2_forest, 1000]

    all_top_presets = [preset_combined_t3, preset_linear_t3, preset_forest_t2]
    preset_names = ['GlobTop3', 'LinearTop3', 'ForestTop2']

    regress_on_different_sets_based_on_label_magnitude(preset_everything[0], preset_everything[1], preset_everything[2],
                                                       preset_everything[3], 'STEFAN',
                                                       data_set_name='Stefan', outlier_threshold=preset_everything[5],
                                                       directory_for_excels='/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/ready_to_ml/all_with_feature/Testruns/Testrun2',
                                                       log_label=True, to_excel=True, sgm=True
                                                       )

def plot_acc(file_path="/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/ready_to_ml/all_with_feature/Testruns/Testrun2/Logged/Accuracy/logged_STEFAN_both_below_1000_hundred_seeds_2_20_03.xlsx",
        title='Accuracy of Stefan', shift=1.0):
    accuracy = pd.read_excel(file_path)
    plot_sgm_accuracy(accuracy, title, shift)

def print_sgm_number_extrem_cases(file_path="/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/ready_to_ml/all_with_feature/Testruns/Testrun2/Logged/Accuracy/logged_STEFAN_both_below_1000_hundred_seeds_2_20_03.xlsx",
                            shift=1.0):
    number_extreme_cases = pd.read_excel(file_path).loc[:, 'Number Extreme Instances'].replace(np.nan, 0, regex=True)
    number_extreme_cases_tuples = number_extreme_cases.apply(ast.literal_eval)
    accuracy_number_extreme_cases = number_extreme_cases_tuples.apply(lambda x: x[0])
    print(shifted_geometric_mean(accuracy_number_extreme_cases, shift))


def plot_sgm_acc(file_path="/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/ready_to_ml/all_with_feature/Testruns/Testrun2/Logged/SGM/sgm_logged_STEFAN_both_below_1000_hundred_seeds_2_20_03.xlsx",
                      title='Second SGM of Stefan'):

    sgm = pd.read_excel(file_path)
    plot_sgm_relative_to_mixed(sgm, title, shift=1.0)

def clean_scip_data(data_set_file_path="/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/ready_to_ml/all_with_feature/scip_data_reduced_columns_no_nan.xlsx",
                    requirement_file_path="/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/ready_to_ml/all_with_feature/scip_requirements.xlsx",
                    to_excel=True,):
    sauberer_darter, broken = main(data_set_file_path, requirement_file_path, to_excel)
    return sauberer_darter, broken

regression()
plot_acc()
plot_sgm_acc()

