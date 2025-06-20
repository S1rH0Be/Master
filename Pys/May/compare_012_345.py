import pandas as pd

origi_scip = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasVeinteDos/CSVs/scip_bases/cleaned_scip/scip_default_clean_data.csv')
new_perm_seeds = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasVeinteDos/CSVs/scip_bases/cleaned_scip/scip_345_clean_data.csv')

def get_difference(df1, df2):
    df1['source'] = '012'
    df2['source'] = '345'

    # Select only 'Matrix Name' and source column
    df1_names = df1[['Matrix Name', 'source']]
    df2_names = df2[['Matrix Name', 'source']]

    # Merge on 'Matrix Name' with outer join
    merged = pd.merge(df1_names, df2_names, on='Matrix Name', how='outer', suffixes=('_df1', '_df2'))

    # Keep only rows that appear in exactly one of the two (i.e. one source is NaN)
    only_one = merged[merged['source_df1'].isna() ^ merged['source_df2'].isna()].copy()

    # Add a column to indicate which DataFrame it came from
    only_one.loc[:,'origin'] = only_one['source_df1'].fillna(only_one['source_df2'])

    # Final result: Matrix Name + which df it belongs to
    result = only_one[['Matrix Name', 'origin']]
    result.to_csv(
        '/Users/fritz/Downloads/ZIB/Master/Treffen/TreffenMasVeinteDos/CSVs/scip_bases/cleaned_scip/difference_012_345.csv')
    return result

def intersection_of_both(df1, df2):
    # Define mapping
    seed_map = {0: 3, 1: 4, 2: 5}

    # Create a new column with the mapped value in df1
    df1 = df1[df1['Random Seed Shift'].isin(seed_map.keys())].copy()
    df1['target_seed'] = df1['Random Seed Shift'].map(seed_map)

    # Filter df2 to only keep relevant seed values
    df2 = df2[df2['Random Seed Shift'].isin(seed_map.values())].copy()

    # Merge on 'Matrix Name' and seed mapping
    merged = pd.merge(
        df1,
        df2,
        left_on=['Matrix Name', 'target_seed'],
        right_on=['Matrix Name', 'Random Seed Shift'],
        suffixes=(' 012', ' 345')
    )
    # Step 5: Order columns so matching names appear side by side
    base_cols = [col for col in df1.columns if col not in ['mapped_seed']]
    base_cols = [col for col in base_cols if col in df2.columns]

    # Interleave _df1 and _df2 versions
    ordered_cols = []
    for col in base_cols[1:]:
        ordered_cols.extend([f"{col} 012", f"{col} 345"])
    final_cols = ['Matrix Name'] + ordered_cols
    merged = merged[final_cols]
    merged.to_csv('/Users/fritz/Downloads/ZIB/Master/Treffen/CSVs/scip_bases/cleaned_scip/012_345_intersection.csv')

intersection_of_both(origi_scip, new_perm_seeds)
