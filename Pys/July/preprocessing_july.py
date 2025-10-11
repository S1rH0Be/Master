import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_histo(data, color, number_bins, title):

    plt.hist(data, bins=number_bins, color=color, alpha=1)
    #plt.title(title)

    # Adjust layout
    plt.tight_layout()
    # Show the plots once all are created
    plt.show()
    # Close the plot to free up memory
    plt.close()


def log_data(data_frame:pd.DataFrame):
    data = data_frame.copy()
    columns_to_be_logged = ['Matrix Equality Constraints', # later be relative to all constraints
                            'Matrix Quadratic Elements',
                            'Matrix NLP Formula', # later be relative to all constraints
                            '#MIP nodes Mixed',
                            '#MIP nodes Int',
                            'Presolve Columns',
                            'Presolve Global Entities',
                            '#spatial branching entities fixed (at the root) Mixed',
                            '#spatial branching entities fixed (at the root) Int',
                            '#non-spatial branch entities fixed (at the root)',
                            'NodesInDAG',
                            '#integer violations at root',
                            '#nonlinear violations at root',
                            ]
    fico_feats = ['Matrix Equality Constraints', 'Matrix Quadratic Elements',
                  'Matrix NLP Formula', 'Presolve Columns', 'Presolve Global Entities',
                  'NodesInDAG', '#integer violations at root',
                  '#nonlinear violations at root', '% vars in DAG (out of all vars)',
                  '% vars in DAG unbounded (out of vars in DAG)',
                  '% vars in DAG integer (out of vars in DAG)',
                  '% quadratic nodes in DAG (out of all non-plus/sum/scalar-mult operator nodes in DAG)',
                  'Avg work for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
                  'Avg work for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
                  'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Mixed',
                  'Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
                  '#spatial branching entities fixed (at the root) Mixed',
                  'Avg coefficient spread for convexification cuts Mixed']

    data[columns_to_be_logged] = data[columns_to_be_logged].apply(lambda entry: np.log10(1 + entry))
    data.to_csv('/Users/fritz/Downloads/ZIB/Master/October/Bases/FICO/Scaled/logged_fico_clean_data.csv', index=False)
    features = data[fico_feats]
    features.to_csv('/Users/fritz/Downloads/ZIB/Master/October/Bases/FICO/Scaled/logged_fico_feats.csv', index=False)
    return data, columns_to_be_logged


fico_data_clean = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/October/Bases/FICO/Cleaned/fico_clean_data_753.csv')

fico_logged, cols = log_data(fico_data_clean)

