from visualize_data_and_results_last_bt import *
from das_ist_die_richtige_regression import *
from Pys.May.scip_data_cleaner import *
from Pys.May.scip_data import *
from feature_importance import *

import ast
'''DEPTH OF RANFOR ANGEPASST!!!!!!!!!!!!!!!!!!!!!!'''
def regression(feature_space, preset_name='STEFAN', data_set_name='Stefan',
              directory_for_excels='/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/ready_to_ml/all_with_feature/Testruns/Testrun2',
              log_label=True, to_excel=True, sgm=True):
    imputators = ['median', 'mean', 'constant']
    scaling = [QuantileTransformer(n_quantiles=100, output_distribution="normal", random_state=42),
               PowerTransformer('yeo-johnson')]
    regression_models = {"LinearRegression": LinearRegression(),
                         "RandomForest": RandomForestRegressor(n_estimators=100, random_state=729154, max_depth=3)}

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

    preset_everything = ['hundred', regression_models, scaling, imputators, ['all'], 1000]

    preset_combined_t3 = ['hundred', regression_models, scaling, imputators, t3_feats_combined, 1000]

    preset_linear_t3 = ['hundred', regression_models, scaling, imputators, t3_linear, 1000]

    preset_linear_t2 = ['hundred', regression_models, scaling, imputators, t2_linear, 1000]

    preset_linear_t1 = ['hundred', regression_models, scaling, imputators, t1_linear, 1000]

    preset_forest_t2 = ['hundred', regression_models, scaling, imputators, t2_forest, 1000]

    preset_example_order_of_magnitude = ['hundred', {"LinearRegression": LinearRegression()}, [MinMaxScaler()], imputators, ['#MIP nodes Mixed', '% vars in DAG (out of all vars)'], 1000]

    all_top_presets = [preset_combined_t3, preset_linear_t3, preset_forest_t2]
    preset_names = ['GlobTop3', 'LinearTop3', 'ForestTop2']

    scip_forest_t1_logged_unscaled = ['hundred', regression_models, scaling, imputators, ['Avg coefficient spread for convexification cuts Mixed'], 1000]
    scip_combined_t2_unscaled = ['hundred', regression_models, scaling, imputators,
                             ['Presolve Columns Mixed', '#nonlinear violations at root Mixed'], 1000]

    # Function to get list by name
    def get_list_by_name(feature_space, namespace):
        return namespace.get(feature_space, f"Preset {feature_space} not found!")
    preset = get_list_by_name(feature_space, locals())
    regress_on_different_sets_based_on_label_magnitude(preset[0], preset[1], preset[2], preset[3], preset[4],
                                                       preset_name, data_set_name, preset[5], log_label, to_excel, sgm,
                                                       directory_for_excels)

def plot_acc(file_path="/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/ready_to_ml/all_with_feature/Testruns/Testrun2/Logged/Accuracy/logged_STEFAN_both_below_1000_hundred_seeds_2_1_20_03.xlsx",
        title='Accuracy of Stefan', shift=1.0):
    accuracy = pd.read_excel(file_path)
    shift = (max(accuracy['Accuracy']) - min(accuracy['Accuracy'])) * 0.1
    plot_sgm_accuracy(accuracy, title, shift)

def print_sgm_number_extrem_cases(file_path="/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/ready_to_ml/all_with_feature/Testruns/Testrun2/Logged/Accuracy/logged_STEFAN_both_below_1000_hundred_seeds_2_1_20_03.xlsx",
                            shift=1.0):
    number_extreme_cases = pd.read_excel(file_path).loc[:, 'Number Extreme Instances'].replace(np.nan, 0, regex=True)
    number_extreme_cases_tuples = number_extreme_cases.apply(ast.literal_eval)
    accuracy_number_extreme_cases = number_extreme_cases_tuples.apply(lambda x: x[0])
    print(shifted_geometric_mean(accuracy_number_extreme_cases, shift))

def plot_sgm_acc(file_path="/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/ready_to_ml/all_with_feature/Testruns/Testrun2/Logged/SGM/sgm_logged_STEFAN_both_below_1000_hundred_seeds_2_1_20_03.xlsx",
                      title='Second SGM of Stefan'):

    accuracies = pd.read_excel(file_path)

    shift = abs(accuracies.iloc[:,1:].min().min())+(accuracies.iloc[:,1:].max().max() - accuracies.iloc[:,1:].min().min()) * 0.1
    plot_sgm_relative_to_mixed(accuracies, title, shift)

def clean_scip_data(data_set_file_path="/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/ready_to_ml/all_with_feature/scip_data_reduced_columns_no_nan.xlsx",
                    requirement_file_path="/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/ready_to_ml/all_with_feature/scip_requirements.xlsx",
                    to_excel=True):
    sauberer_darter, broken = main(data_set_file_path, requirement_file_path, to_excel)
    return sauberer_darter, broken

def call_scip_data(directory_path = "/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/Outs",
                             fico=False, to_excel=True):
    read_in_and_call_process(directory_path, fico, to_excel)


def plot_logged_feature_importance(model:['Linear','Forest'], title:str, file_name,
                       directory='/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/ready_to_ml/all_with_feature/Testruns/Testrun2/Logged/Importance/'):
    feat_impo = pd.read_excel(directory+model+'/'+file_name)
    plot_sgm_feature_importance(feat_impo, title)


def plot_unscaled_feature_importance(model:['Linear','Forest'], title:str, file_name,
                                directory='/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/ready_to_ml/all_with_feature/Testruns/Testrun2/Unscaled/Importance/'):
    feat_impo = pd.read_excel(directory+model+'/'+file_name)
    plot_sgm_feature_importance(feat_impo, title)

def call_feature_importance_plots(logged=True, unscaled=True):
    if logged:
        plot_logged_feature_importance(model='Linear',file_name='logged_lin_impo_STEFAN_below_1000_hundred_seeds_2_20_03.xlsx',
                                  title='LinearRegression Feature Importance, logged label')
        plot_logged_feature_importance(model='Forest', file_name='logged_forest_impo_STEFAN_below_1000_hundred_seeds_2_20_03.xlsx',
                       title='RandomForestRegression Feature Importance, logged label')
    if unscaled:
        plot_unscaled_feature_importance(model='Linear',
                                  file_name='unscaled_lin_impo_STEFAN_below_1000_hundred_seeds_2_20_03.xlsx',
                                  title='LinearRegression Feature Importance, unscaled label')
        plot_unscaled_feature_importance(model='Forest',
                                  file_name='unscaled_forest_impo_STEFAN_below_1000_hundred_seeds_2_20_03.xlsx',
                                  title='RandomForestRegression Feature Importance, unscaled label')

def call_importance_scores(title:str, unscaled_or_logged:str,
                            directory='/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/Stefan/Stefan_Werte/ready_to_ml/all_with_feature/Testruns/Testrun2/'):
    lin_df = pd.read_excel(directory+unscaled_or_logged+'/Importance/Linear/'+f'{unscaled_or_logged.lower()}_lin_impo_STEFAN_below_1000_hundred_seeds_2_20_03.xlsx')
    for_df = pd.read_excel(directory+unscaled_or_logged+'/Importance/Forest/'+f'{unscaled_or_logged.lower()}_forest_impo_STEFAN_below_1000_hundred_seeds_2_20_03.xlsx')
    importance_ranking = create_importance_score_df(lin_df, for_df)
    importance_ranking.to_excel(f'{directory}{unscaled_or_logged}/Importance/{unscaled_or_logged}Ranking/{unscaled_or_logged}_importance_ranking.xlsx', index=False)

def create_fico_feat_sheet(clean_darter:pd.DataFrame):
    feature_cols = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/921/base_feats_no_cmp_24_01.xlsx').columns
    clean_feats = clean_darter[feature_cols].copy()
    if len(clean_feats) == 918:
        clean_feats.to_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/918/base_feats_no_cmp_918_24_01.xlsx')

# create_fico_feat_sheet(pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/918/clean_data_final_06_03.xlsx'))

regression('preset_everything', data_set_name='Timo', log_label=False)
