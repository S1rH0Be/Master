import pandas as pd
import matplotlib.pyplot as plt

scip_default_base = pd.read_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/default/complete/scip_default_ready_to_ml.csv')
scip_no_pseudocosts_base = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/no_pseudocosts/complete/scip_no_pseudocosts_ready_to_ml.csv',)
fico_base = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/GitCode/Master/NewEra/BaseCSVs/918/clean_data_final_06_03.xlsx')


# drop _dup1-4
fico_base['Matrix Name'] = fico_base['Matrix Name'].str.replace(r'_dup\d+', '', regex=True)
scip_instances = scip_default_base['Matrix Name'].tolist()
fico_instances = fico_base['Matrix Name'].tolist()





fico_indices = fico_base[fico_base['Matrix Name'].isin(scip_instances)].index.tolist()
fico_schnitt = fico_base.loc[fico_indices, ['Matrix Name',
                                            'Final solution time (cumulative) Mixed',
                                            'Final solution time (cumulative) Int',
                                            'Cmp Final solution time (cumulative)']]


scip_indices = scip_default_base[scip_default_base['Matrix Name'].isin(fico_instances)].index.tolist()
scip_default_schnitt = scip_default_base.loc[scip_indices, ['Matrix Name',
                                                            'Final solution time (cumulative) Mixed',
                                                            'Final solution time (cumulative) Int',
                                                            'Cmp Final solution time (cumulative)']]

scip_no_pseudocosts_schnitt = scip_no_pseudocosts_base.loc[scip_indices, ['Matrix Name',
                                                                          'Final solution time (cumulative) Mixed',
                                                                          'Final solution time (cumulative) Int',
                                                                          'Cmp Final solution time (cumulative)']]


fico_schnitt.to_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/fico_schnitt.csv', index=False)
scip_default_schnitt.to_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/scip_default_schnitt.csv', index=False)
scip_no_pseudocosts_schnitt.to_csv(f'/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/scip_no_pseudocosts_schnitt.csv', index=False)


schnitte = [(scip_default_schnitt, 'SCIP Default Share Mixed vs PrefInt'),
            (scip_no_pseudocosts_schnitt, 'SCIP No Pseudocosts Share Mixed vs PrefInt'),
            (fico_schnitt, 'FICO Xpress Share Mixed vs PrefInt')]
for schnitt in schnitte:
    mixed = (schnitt[0]['Cmp Final solution time (cumulative)'] > 0).sum().sum()
    pref_int = (schnitt[0]['Cmp Final solution time (cumulative)'] < 0).sum().sum()

    values = [mixed, pref_int]
    non_zeros = len(schnitt[0])-schnitt[0]['Cmp Final solution time (cumulative)'].eq(0).sum()

    values = [(value / non_zeros) * 100 for value in values]
    bar_colors = (['turquoise', 'magenta'])

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.bar(['Mixed', 'Prefer Int'], values, color=bar_colors)
    #plt.title(schnitt[1])
    plt.ylim(20, 80)  # Set y-axis limits for visibility
    plt.xticks(rotation=45, fontsize=6)
    # Create custom legend entries with value annotations
    # legend_labels = [f"{label}: {value}" for label, value in zip(labels, values)]
    # plt.legend(bars, legend_labels, title="Values")
    # Display the plot
    plt.show()
    plt.close()

print(fico_base['Final solution time (cumulative) Int'].sum()/fico_base['Final solution time (cumulative) Mixed'].sum())
print(fico_schnitt['Final solution time (cumulative) Int'].sum()/fico_schnitt['Final solution time (cumulative) Mixed'].sum())