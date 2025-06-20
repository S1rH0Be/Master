import pandas as pd


def sort_by_forest_importance(data_set, to_csv=False, latex=False):
    forest = data_set.sort_values('Forest', ascending=True)
    lin = data_set.sort_values('Linear', ascending=True)

    if to_csv:
        if latex:
            specials = {
                r'&': r'\textbackslash{}&',
                r'%': r'\textbackslash{}%',
                r'$': r'\$',
                r'#': r'\#',
                r'_': r'\_',
                r'{': r'\{',
                r'}': r'\}',
                r'~': r'\textasciitilde{}',
                r'^': r'\textasciicircum{}',
                r'\\': r'\textbackslash{}',
            }

            forest.replace(specials, regex=True)
            forest.to_csv(
                '/Users/fritz/Downloads/ZIB/Master/Treffen/Präsis/treffen_11-06/FeatImpo/fico_for_importance_pre_latex.csv',
            index=False)
            print(forest.head())
            lin.replace(specials, regex=True)
            lin.to_csv(
                '/Users/fritz/Downloads/ZIB/Master/Treffen/Präsis/treffen_11-06/FeatImpo/fico_lin_importance_pre_latex.csv',
            index=False)

        else:
            forest.to_csv(
                '/Users/fritz/Downloads/ZIB/Master/Treffen/Präsis/treffen_11-06/FeatImpo/fico_for_importance.csv',
                          index=False)
            lin.to_csv(
                '/Users/fritz/Downloads/ZIB/Master/Treffen/Präsis/treffen_11-06/FeatImpo/fico_lin_importance.csv',
                       index=False)


fico_imoportance = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/Präsis/treffen_11-06/FeatImpo/fico_importance.csv')

sort_by_forest_importance(fico_imoportance, to_csv=True, latex=True)