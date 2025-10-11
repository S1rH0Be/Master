import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from visualize_erfolg import compare_time
from feature_distribution import scale_by_hand
from feature_importance import feature_importance_linreg, feature_importance_forrest

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold

for_top_feat = []
lin_top_feat = []

def plot_lin_coefficients(model, feat_train, printer=True):
    if printer:
        coefs = pd.DataFrame(
            model.coef_, columns=["Coefficients"], index=feat_train.columns
        )
        coefs.plot(kind="barh", figsize=(9, 7))
        #plt.title("Ridge model")
        #plt.axvline(x=0, color=".5")
        plt.subplots_adjust(left=0.3)
       # plt.show()

def cross_validate_feature(model, feature, label, printer=True):
    if printer:
        cv_model = cross_validate(
            model,
            feature,
            label,
            cv=RepeatedKFold(n_splits=5, n_repeats=5),
            return_estimator=True,
            n_jobs=2,
        )
        coefs = pd.DataFrame(
            [cvmodel.coef_ for cvmodel in cv_model["estimator"]],
            columns=feature.columns,
        )
        plt.figure(figsize=(9, 7))
        sns.boxplot(data=coefs, orient="h", color="cyan", saturation=0.5)
        #plt.axvline(x=0, color=".5")
        plt.xlabel("Coefficient importance")
        #plt.title("Coefficient importance and its variability")
        plt.subplots_adjust(left=0.3)
        #plt.show()

def residual(prediction, actual, pred_v_act=True, resi=True):
    residuals = actual - prediction
    mask = np.abs(residuals) <= 1000
    # Apply mask to filter data
    pred = prediction[mask]
    test = actual[mask]
    residuals = residuals[mask]
    # Plotting the residuals
    if pred_v_act:
        plt.figure(figsize=(8, 6))
        plt.scatter(pred, test, color='blue', edgecolor='k', alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--')  # Reference line at y=0
        #plt.title("Pred vs Actual")
        plt.xlabel("Pred values")
        plt.ylabel("Actual Values")
        #plt.show()
    if resi:
        plt.figure(figsize=(8, 6))
        plt.scatter(pred, residuals, color='blue', edgecolor='k', alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--')  # Reference line at y=0
        #plt.title("Residual Plot")
        plt.xlabel("Predicted values")
        plt.ylabel("Residuals")
        #plt.show()
# linreg
def lin_reg(feat_train, feat_test, label_train, label_test, scaling, imputation, printer=False):
    if printer:
        print('LinReg', scaling, imputation)


    rel_indices = label_test.index[label_test != 1.0]
    # Create the model
    model = LinearRegression()

    # Train the model
    fitted = model.fit(feat_train, label_train)
    mvp_train, mvp_test = get_important_feature_lin(fitted, feat_train, feat_test, label_train, label_test, ('Top', 8))
    mvp_lin_train.append(mvp_train)
    mvp_lin_test.append(mvp_test)
    # Make predictions on the test data
    y_pred = model.predict(feat_test)

    #residual(y_pred, label_test, pred_v_act=True, resi=True)

    rel_pred = [y_pred[i] for i in range(len(y_pred)) if label_test.iloc[i]!=1.0]

    return y_pred, rel_pred, rel_indices#, importance_df
# forrest_reg
def forrest_reg(feat_train, feat_test, label_train, label_test, scaling, imputation, size=100, printer=False):
    if printer:
        print('FORREST')
    rel_indices = label_test.index[label_test != 1.0]
    # Create the Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=size, random_state=42)
    # Train the model
    model.fit(feat_train, label_train)
    # print(f"model score on training data: {model.score(feat_train, label_train)}")
    # print(f"model score on testing data: {model.score(feat_test, label_test)}")
    importance_df = feature_importance_forrest(feat_train.columns, model, imputation, scaling).sort_values(
        by="Importance",
        key=lambda x: x.abs(),
        ascending=False  # Set to True for ascending order
    )

    for_top_feat.append(importance_df['Feature'].iloc[0:8].values)
    # Make predictions on the test data
    y_pred = model.predict(feat_test)

    rel_pred = [y_pred[i] for i in range(len(y_pred)) if label_test.iloc[i]!=1.0]

    return y_pred, rel_pred, rel_indices#, importance_df

def create_time_df(df, indices):
    time_df =  df[df.columns[df.columns.str.contains('time')]]#.drop('Final solution time (cumulative)', axis=1)
    time_df = time_df.loc[indices,:]
    time_df['Virtual Best'] = time_df.apply(lambda row: min(row['Final solution time (cumulative) Mixed'], row['Final solution time (cumulative) Int']),
    axis=1)
    return time_df

def imputations(df, median=True, mean=True):
    imputated = []
    imputation_names = []
    if median:
        median_df = df.copy()
        for col in df.columns:
            median_df[col] = median_df[col].fillna(median_df[col].median())
        imputated.append(median_df)
        imputation_names.append('Median Imputation')

    if mean:
        mean_df = df.copy()
        for col in df.columns:
            mean_df[col] = mean_df[col].fillna(mean_df[col].mean())
        imputated.append(mean_df)
        imputation_names.append('Mean Imputation')

    return imputated, imputation_names

def acc_on_cmp(pred, test, regressor, extreme_threshold=1.5, printer=True):
    test_indices = test.index
    pred_sign = [np.sign(i) for i in pred]
    test_sign = [np.sign(i) if i!=1 else 0 for i in test]
    all_choices = [p-t if t!= 0 else 0 for p, t in zip(pred_sign, test_sign)]

    relevant_choices = [p-t for p, t in zip(pred_sign, test_sign) if t!=0 ]
    total_non_trivial_accuracy = np.round((relevant_choices.count(0) / len(relevant_choices)) * 100, 2)


    """"Extreme cases kommen hier"""
    #extreme case: Cmp factor >=1.5(or extreme_threshold)
    extreme_cmp_indices = []
    extreme_cmp_choices = []
    for inst in range(len(test)):
        if abs(test.iloc[inst])>=extreme_threshold:
            extreme_cmp_choices.append(all_choices[inst])
            extreme_cmp_indices.append(test_indices[inst])

    extreme_cmp_accuracy = np.round((extreme_cmp_choices.count(0) / len(extreme_cmp_choices)) * 100, 2)
    if printer:
        print('Total non-trivial Accuracy '+regressor+': ', total_non_trivial_accuracy,'Extreme: ', extreme_cmp_accuracy)

    """Acc per intervall"""
    intervall_acc = [total_non_trivial_accuracy]
    number_of_inst_in_intervall =[str(len(all_choices))]

    for intervall in np.arange(1, 1.5, 0.5):
        in_intervall_indices = []
        choices_on_intervall = []
        for inst in range(len(test)):
            if intervall<= abs(test.iloc[inst])<intervall+0.5:
                choices_on_intervall.append(all_choices[inst])
                in_intervall_indices.append(test_indices[inst])

        intervall_accuracy = np.round((choices_on_intervall.count(0) / len(choices_on_intervall)) * 100, 2)
        intervall_acc.append(intervall_accuracy)
        number_of_inst_in_intervall.append(len(choices_on_intervall))

    intervall_acc.append(extreme_cmp_accuracy)
    number_of_inst_in_intervall.append(len(extreme_cmp_choices))
    number_of_inst_in_intervall = [str(val) for val in number_of_inst_in_intervall]
    # Define colors based on value thresholds
    colors = ['green' if value > 80 else 'red' if value <= 60 else 'magenta' for value in intervall_acc]
    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(number_of_inst_in_intervall, intervall_acc, color=colors)
    plt.suptitle(regressor)
    subtitle = "Intervals: [0,inf),[1,1.5),[1.5,inf)"
    #plt.title(subtitle, fontsize=10)
    plt.ylim(0, 100)  # Set y-axis limits for visibility
    plt.xlabel("Number of Instances in Intervall")
    plt.ylabel("Accuracy in %")
    # plt.show()
    plt.close()
    return total_non_trivial_accuracy, extreme_cmp_accuracy

# def predictions_diff_imputation(features_train, features_test, label_train, label_test, regressor, printer=False):
#     """ADD RELevant PRED TO RETURN SO I CAN USE IT FOR REGRESSION FUNCTION IN ORDER TO VISUALIZE JUST THE RELEVANT INSTANCES"""
#     #imputated, imputation_names = imputations(features)
#     #Iterate over both lists using zip
#     #its just median right now
#     #for impu, name in zip(imputated, imputation_names):
#     # if printer:
#     #     print(f"Running models with {name}")
#     if regressor == 'LinReg':
#         return lin_reg(features_train, features_test, label_train, label_test)
#
#     else:
#         return forrest_reg(features_train, features_test, label_train, label_test)

def pred_time_visualization(df, prediction, test_values):
    predicted_time = []
    for i in range(len(prediction)):
        if np.sign(prediction[i]) >=0:#cmp time>=0 -> TimeMixed<=TimeInt
            predicted_time.append(np.round(df['Final solution time (cumulative) Mixed'].loc[test_values.index[i]],2))
        else:
            predicted_time.append(np.round(df['Final solution time (cumulative) Int'].loc[test_values.index[i]],2))
    #create df with int, mixed and pred to visualize them(vbs gets added in the visualize function
    cmp_df = pd.DataFrame(index=test_values.index, columns=['Final solution time (cumulative) Int', 'Final solution time (cumulative) Mixed', 'Final solution time (cumulative) Predicted'])
    cmp_df.loc[:, ['Final solution time (cumulative) Int','Final solution time (cumulative) Mixed']] = df.loc[test_values.index, ['Final solution time (cumulative) Int','Final solution time (cumulative) Mixed']]
    cmp_df.loc[:, 'Final solution time (cumulative) Predicted'] = predicted_time

    #pred_test_tuple = [(prediction[i], test_values.iloc[i]) for i in range(len(test_values))]
    #compare_time(cmp_df, reference='Mixed', plot=True, title_add_on=' on Testset', y_pred_test=pred_test_tuple)

def visualize_it(df, pred, test):
    pred_test_tuple = [(p,t) for p,t in zip(pred, test)]
    compare_time(df, 'Mixed', pred_test_tuple, title_add_on='on Testset')

def add_pred_time_col(df, pred):
    for index, predi in zip(df.index, pred):
        if predi>=0:
            df.loc[index,'Predicted Time'] = df['Final solution time (cumulative) Mixed'].loc[index]
        else:
            df.loc[index,'Predicted Time'] = df['Final solution time (cumulative) Int'].loc[index]
    return df

def add_abs_time_diff_col(df):
    helper_df =df.copy()
    helper_df['Abs time diff'] = 0.0
    for index,row in helper_df.iterrows():
        if row['Cmp Final solution time (cumulative)']!=0:
            helper_df.loc[index, 'Abs time diff'] = abs(row['Final solution time (cumulative) Mixed']-row['Final solution time (cumulative) Int'])
    return helper_df

def lin_pred(feature, label, regressor, only_relevant=False, accuracy=True, compare=False):
    pred, test, test_indices, rel_preds, rel_ind = predictions_diff_imputation(feature, label,'LinReg')

    lin_tdf = create_time_df(clean_df, test_indices)

    if only_relevant:
        lin_tdf = lin_tdf.loc[rel_ind, :]
        lin_tdf['Cmp Final solution time (cumulative) Predicted'] = rel_preds
        lin_tdf = add_pred_time_col(lin_tdf, rel_preds)
        lin_tdf = add_abs_time_diff_col(lin_tdf)
        # lin_tdf.to_excel('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/Master_Excel/only_rel_lin_tdf.xlsx', index=False)

        if accuracy:
            acc_on_cmp(rel_preds, test.loc[rel_ind], regressor, extreme_threshold=1.5, printer=True)
    else:
        lin_tdf['Cmp Final solution time (cumulative) Predicted'] = pred
        lin_tdf = add_pred_time_col(lin_tdf, pred)
        lin_tdf = add_abs_time_diff_col(lin_tdf)
        # lin_tdf.to_excel('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/Master_Excel/lin_tdf.xlsx', index=False)

        if accuracy:
            acc_on_cmp(pred, test, regressor, extreme_threshold=1.5, printer=True)

    if compare:
        compare_time(lin_tdf[['Final solution time (cumulative) Mixed','Final solution time (cumulative) Int',
                         'Predicted Time','Virtual Best']], 'Mixed')

def forrest_pred(feature, label, regressor, only_relevant=False, accuracy=True, compare=False):
    pred, test, test_indices, rel_preds, rel_ind = predictions_diff_imputation(feature, label, regressor)

    forrest_tdf = create_time_df(clean_df, test_indices)

    if only_relevant:
        forrest_tdf = forrest_tdf.loc[rel_ind,:]
        forrest_tdf['Cmp Final solution time (cumulative) Predicted'] = rel_preds
        forrest_tdf = add_pred_time_col(forrest_tdf, rel_preds)
        forrest_tdf = add_abs_time_diff_col(forrest_tdf)
        # forrest_tdf.to_excel('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/Master_Excel/only_rel_forrest_tdf.xlsx', index=False)
        if accuracy:
            acc_on_cmp(rel_preds, test.loc[rel_ind], regressor, extreme_threshold=1.5, printer=True)
    else:
        forrest_tdf['Cmp Final solution time (cumulative) Predicted'] = pred
        forrest_tdf = add_pred_time_col(forrest_tdf, pred)
        forrest_tdf = add_abs_time_diff_col(forrest_tdf)
        # forrest_tdf.to_excel('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/Master_Excel/forrest_tdf.xlsx', index=False)

        if accuracy:
            acc_on_cmp(pred, test, regressor, extreme_threshold=1.5, printer=True)

    if compare:
        compare_time(forrest_tdf[['Final solution time (cumulative) Mixed','Final solution time (cumulative) Int',
                         'Predicted Time','Virtual Best']].loc[rel_ind,:], 'Mixed')

def pred_cmp_to_pred_time(df, col_names):
        for index,row in df.iterrows():
            for col in col_names:
                if row[col] >=0:
                    df.loc[index,col] = df['Final solution time (cumulative) Mixed'].loc[index]
                else:
                    df.loc[index, col] = df['Final solution time (cumulative) Int'].loc[index]
        return df

def get_important_feature_lin(fitted_model, train_feat, test_feat, label_train, label_test, criterion):
    #criterion is a tuple either('Top', Integer) or ('Threshold', float)
    criterions = ['All', 'Top', 'Threshold']
    if criterion[0] not in criterions:
        print("Not a valid criterion needs to be in ['Top', 'Threshold']")
        return None

    train_score = fitted_model.coef_

    test_score = fitted_model.coef_

    train_tuples = [(feature,importance) for feature,importance in zip(train_feat.columns, train_score)]
    test_tuples = [(feature,importance) for feature,importance in zip(test_feat.columns, test_score)]
    train_tuples.sort(key=lambda tup: tup[1])
    test_tuples.sort(key=lambda tup: tup[1])

    if criterion[0] == 'Top':
        train_tuples = train_tuples[0:criterion[1]] #criterion[1] is an int who gives the number of features to be kept
        test_tuples = test_tuples[0:criterion[1]]

    elif criterion[0] == 'Threshold':
        train_tuples = [t for t in train_tuples if t[1] >= criterion[1]]#criterion[1] is a float which defines a threshold if a feature is important enough to be kept
        test_tuples = [t for t in test_tuples if t[1] >= criterion[1]]

    return train_tuples, test_tuples

def impute_scale_predict(test_label, train_label, relevant_labels, feat_set, median=True, mean=True, byhand=True, power=True, no_scaling=True, accuracy=True, plot_accuracy=True, create_time_frame=True, cmp_time=True):

    train_imputations, imputations_names_train = imputations(X_train, median, mean)
    test_imputations, imputations_names_test = imputations(X_test, median, mean)
    #median imputation
    if median:
        train_imputed = train_imputations[0]
        test_imputed = test_imputations[0]
        if byhand:
            X_train_by_hand_median = scale_by_hand(train_imputed)
            X_test_by_hand_median = scale_by_hand(test_imputed)
            lin_predicted_median, lin_relevant_pred_median, lin_relevant_indices_median = lin_reg(X_train_by_hand_median, X_test_by_hand_median, train_label, test_label, 'Median', 'Hand', printer=True)
            forrest_predicted_median, forrest_relevant_pred_median, forrest_relevant_indices_median = forrest_reg(X_train_by_hand_median, X_test_by_hand_median, train_label, test_label, 'Median', 'Hand', printer=True)
        if power:
            transformer = PowerTransformer(method='yeo-johnson')
            X_train_normal_median = pd.DataFrame(
                transformer.fit_transform(train_imputed),
                columns=train_imputed.columns,  # Preserve original column names
                index=train_imputed.index  # Preserve original index
            )
            X_test_normal_median = pd.DataFrame(
                transformer.fit_transform(test_imputed),
                columns=test_imputed.columns,  # Preserve original column names
                index=test_imputed.index  # Preserve original index
            )
            lin_predicted_median_power, lin_relevant_pred_median_normal, lin_relevant_indices_median_normal = lin_reg(X_train_normal_median, X_test_normal_median, train_label, test_label, 'Median', 'PowerTransform', printer=True)
            forrest_predicted_median_power, forrest_relevant_pred_median_normal, forrest_relevant_indices_median_normal = forrest_reg(X_train_normal_median, X_test_normal_median, train_label, test_label, 'Mean', 'PowerTransform', printer=True)
        if no_scaling:
            forrest_median, forrest_relevant_median, forrest_relevant_indices = forrest_reg(train_imputations[0], test_imputations[0], train_label, test_label, 'Median', 'None', printer=False)
    #mean imputation
    if mean:
        train_imputed = train_imputations[-1]
        test_imputed = test_imputations[-1]
        if byhand:
            X_train_by_hand_mean = scale_by_hand(train_imputed)
            X_test_by_hand_mean = scale_by_hand(test_imputations[1])
            lin_predicted_mean, lin_relevant_pred_mean, lin_relevant_indices_mean = lin_reg(X_train_by_hand_mean, X_test_by_hand_mean, train_label, test_label, 'Mean', 'Hand', printer=True)
            forrest_predicted_mean, forrest_relevant_pred_mean, forrest_relevant_indices_mean = forrest_reg(X_train_by_hand_mean, X_test_by_hand_mean, train_label, test_label, 'Mean', 'Hand', printer=True)

        if power:
            transformer = PowerTransformer(method='yeo-johnson')
            X_train_normal_mean = pd.DataFrame(
                transformer.fit_transform(train_imputed),
                columns=train_imputed.columns,  # Preserve original column names
                index=train_imputed.index      # Preserve original index
            )
            X_test_normal_mean = pd.DataFrame(
                transformer.fit_transform(test_imputed),
                columns=test_imputed.columns,  # Preserve original column names
                index=test_imputed.index      # Preserve original index
            )
            lin_predicted_mean_power, lin_relevant_pred_mean_normal, lin_relevant_indices_mean_normal = lin_reg(X_train_normal_mean, X_test_normal_mean, train_label, test_label, 'Mean', 'PowerTransform', printer=True)
            forrest_predicted_mean_power, forrest_relevant_pred_mean_normal, forrest_relevant_indices_mean_normal = forrest_reg(X_train_normal_mean, X_test_normal_mean, train_label, test_label, 'Mean', 'PowerTransform', printer=True)

        if no_scaling:
            forrest_mean, forrest_relevant_mean, forrest_relevant_indice = forrest_reg(train_imputed, test_imputed, train_label, test_label, 'None', 'Mean', printer=False)
    #accs
    if accuracy:
        #lin
        lhmed= acc_on_cmp(lin_relevant_pred_median, y_test.loc[lin_relevant_indices_median], 'LinReg Median; scaled by hand',extreme_threshold=1.5, printer=True)
        lhmean=acc_on_cmp(lin_relevant_pred_mean, y_test.loc[lin_relevant_indices_mean], 'LinReg Mean; scaled by hand',extreme_threshold=1.5, printer=True)
        lpowermed=acc_on_cmp(lin_relevant_pred_median_normal, y_test.loc[lin_relevant_indices_median_normal],'LinReg Median; Power transform', extreme_threshold=1.5, printer=True)
        lpowermean=acc_on_cmp(lin_relevant_pred_mean_normal, y_test.loc[lin_relevant_indices_mean_normal],'LinReg Mean; Power transform', extreme_threshold=1.5, printer=True)
        #forrest
        fhmedian=acc_on_cmp(forrest_relevant_pred_median, y_test.loc[forrest_relevant_indices_median],'Forrest Median; scaled by hand', extreme_threshold=1.5, printer=True)
        fhmean=acc_on_cmp(forrest_relevant_pred_mean, y_test.loc[forrest_relevant_indices_mean], 'Forrest Mean; scaled by hand', extreme_threshold=1.5, printer=True)
        fpowermed=acc_on_cmp(forrest_relevant_pred_median_normal, y_test.loc[forrest_relevant_indices_median_normal],'Forrest Median; Power transform', extreme_threshold=1.5, printer=True)
        fpowermean=acc_on_cmp(forrest_relevant_pred_mean_normal, y_test.loc[forrest_relevant_indices_mean_normal],'Forrest Mean; Power transform', extreme_threshold=1.5, printer=True)
        fnomed=acc_on_cmp(forrest_relevant_median, y_test.loc[forrest_relevant_indices], 'Forrest Median; NoScaling',extreme_threshold=1.5, printer=True)
        fnomean=acc_on_cmp(forrest_relevant_mean, y_test.loc[forrest_relevant_indice], 'Forrest Mean; NoScaling',extreme_threshold=1.5, printer=True)

        # all_acc_tuples = [lhmed, lhmean, lpowermed, lpowermean, fhmedian, fhmean, fpowermed, fpowermean, fnomed, fnomean]
        lin_acc_tuples = [lhmed, lhmean, lpowermed, lpowermean]
        lin_names = ['lhmed', 'lhmean', 'lpowermed', 'lpowermean']
        forrest_acc_tuples = [fhmedian, fhmean, fpowermed, fpowermean, fnomed, fnomean]
        forrest_names = ['fhmedian', 'fhmean', 'fpowermed', 'fpowermean', 'fnomed', 'fnomean']

        if plot_accuracy:
            #Prepare x and y values for plotting
            x_forrest = []
            y_forrest = []
            colors = []

            for i, (a, b) in enumerate(forrest_acc_tuples):
                # Append names for x-axis
                x_forrest.extend([f"{forrest_names[i]}", f"{forrest_names[i]}_extr"])  # Unique labels for each bar
                y_forrest.extend([a, b])  # Add the values
                # Assign magenta for even tuples and green for odd tuples
                tuple_color = "magenta" if i % 2 == 0 else "orange"
                colors.extend([tuple_color, tuple_color])
            # Plot the values
            plt.figure(figsize=(8, 5))
            plt.bar(x_forrest, y_forrest, color=colors)
            plt.xlabel("Model")
            plt.ylabel("Accuracy")
            #plt.title("ForReg "+feat_set+" Accuracy")
            plt.ylim(60, 100)
            plt.xticks(rotation=70)
            plt.tight_layout()  # Adjust layout to avoid label overlap
            plt.show()
            plt.close()

            #Prepare x and y values for plotting
            x_lin = []
            y_lin = []
            colors = []

            for i, (a, b) in enumerate(lin_acc_tuples):
                # Append names for x-axis
                x_lin.extend([f"{lin_names[i]}", f"{lin_names[i]}_extr"])  # Unique labels for each bar
                y_lin.extend([a, b])  # Add the values
                # Assign magenta for even tuples and green for odd tuples
                tuple_color = "magenta" if i % 2 == 0 else "orange"
                colors.extend([tuple_color, tuple_color])
            # Plot the values
            plt.figure(figsize=(8, 5))
            plt.bar(x_lin, y_lin, color=colors)
            plt.xlabel("Model")
            plt.ylabel("Accuracy")
            #plt.title("LinReg "+feat_set+" Accuracy")
            plt.ylim(50, 100)
            plt.xticks(rotation=70)
            plt.tight_layout()  # Adjust layout to avoid label overlap
            plt.show()
            plt.close()
    #all_preds = [lin_predicted_median, lin_predicted_mean, lin_predicted_mean_power, forrest_predicted_mean, forrest_mean, lin_predicted_median_power,forrest_predicted_median,forrest_predicted_median_power, forrest_predicted_mean_power,forrest_median]

    relevant_preds_lin = [lin_relevant_pred_median, lin_relevant_pred_mean, lin_relevant_pred_mean_normal, lin_relevant_pred_mean_normal]
    relevant_preds_forrest = [forrest_relevant_pred_mean, forrest_relevant_mean, lin_relevant_pred_median_normal,forrest_relevant_pred_median,forrest_relevant_pred_median_normal, forrest_relevant_pred_mean_normal,forrest_relevant_median]

    pred_col_names_lin = ['LinPredTime Median byHand','LinPredTime Mean byHand','LinPredTime Median Power','LinPredTime Mean Power']

    pred_col_names_forrest = ['ForrestPredTime Median byHand','ForrestPredTime Mean byHand','ForrestPredTime Median Power',
                              'ForrestPredTime Mean Power', 'ForrestPredTime Median NoScale', 'ForrestPredTime Mean NoScale']

    if create_time_frame:
        tdf = create_time_df(clean_df, relevant_labels).drop('Cmp Final solution time (cumulative)', axis=1)
        lin_tdf = tdf

        for name, values in zip(pred_col_names_lin, relevant_preds_lin):
            lin_tdf.loc[:,name] = values

        lin_tdf = pred_cmp_to_pred_time(lin_tdf, pred_col_names_lin)

        if cmp_time:
            compare_time(lin_tdf, 'Mixed', plot=True, plot_all=True, title_add_on='')

        forrest_tdf = tdf
        for name, values in zip(pred_col_names_forrest, relevant_preds_forrest):
            forrest_tdf.loc[:, name] = values
        forrest_tdf = pred_cmp_to_pred_time(forrest_tdf, pred_col_names_lin)

        if cmp_time:
            compare_time(forrest_tdf, 'Mixed', plot=True, plot_all=True, title_add_on='')



clean_df = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/Master_Excel/pointfive_is_zero_df.xlsx')
#features unscaled
X = pd.read_excel('/Users/fritz/Downloads/ZIB/Master/CSV/SecondIteration/Jetzt Ernst/CSVs/Master_Excel/features_pointfive_is_zero_df.xlsx')

#eventuell drop these columns but right now makes it worse
# X = X.drop(['Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
#                           'Avg relative bound change for solving strong branching LPs for integer branchings (not including infeasible ones) Int'],
#                           axis=1)

#label y
y = clean_df['Cmp Final solution time (cumulative)']

mvp_lin_train = []
mvp_lin_test = []
mvp_forrest_train = []
mvp_forrest_test = []


top_features_lin_reg = ['#MIP nodes Int', '#MIP nodes Mixed',
       '#spatial branching entities fixed (at the root) Mixed',
       'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Int',
       '#spatial branching entities fixed (at the root) Int',
       'Matrix NLP Formula',
       'Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
       'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Mixed',
       'Matrix Quadratic Elements', 'Presolve Columns Int',
       '% vars in DAG integer (out of vars in DAG) Mixed',
       '% vars in DAG integer (out of vars in DAG) Int']
lin_top_ten = ['#MIP nodes Int', '#MIP nodes Mixed',
       '#spatial branching entities fixed (at the root) Mixed',
       'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Int',
       '#spatial branching entities fixed (at the root) Int',
       '% vars in DAG integer (out of vars in DAG) Mixed',
       'Presolve Columns Int',
       'Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
       'Matrix NLP Formula', '% vars in DAG integer (out of vars in DAG) Int']
forrest_top_nine = ['#MIP nodes Mixed', '#MIP nodes Int',
       'Avg coefficient spread for convexification cuts Mixed',
       'Avg coefficient spread for convexification cuts Int',
       'Avg ticks for solving strong branching LPs for integer branchings (not including infeasible ones) Int',
       'Avg ticks for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
       '% vars in DAG (out of all vars) Int',
       'Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
       '% vars in DAG (out of all vars) Mixed']
#each feature has to be at least in 2 top 8
top_eleven_for_lin_for = ['#MIP nodes Mixed', '#MIP nodes Int',
       'Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
       '#spatial branching entities fixed (at the root) Int',
       'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Int',
       '#spatial branching entities fixed (at the root) Mixed',
       'Avg ticks for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
       'Avg ticks for solving strong branching LPs for integer branchings (not including infeasible ones) Int',
       'Avg coefficient spread for convexification cuts Mixed',
       'Avg coefficient spread for convexification cuts Int',
       'Matrix NLP Formula']
#each feature has to be at least in 3 top 8
top_6_for_lin_for = ['#MIP nodes Mixed', '#MIP nodes Int',
       'Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
       '#spatial branching entities fixed (at the root) Int',
       'Avg relative bound change for solving strong branching LPs for spatial branchings (not including infeasible ones) Int',
       '#spatial branching entities fixed (at the root) Mixed']
#each feature has bo at least in 5 top8, across all scaling/impu/regressor pairs
global_mvps = ['#MIP nodes Int', '#MIP nodes Mixed',
       'Avg ticks for solving strong branching LPs for integer branchings (not including infeasible ones) Int',
       'Avg coefficient spread for convexification cuts Int',
       'Avg coefficient spread for convexification cuts Mixed']#,
       # 'Avg ticks for solving strong branching LPs for integer branchings (not including infeasible ones) Mixed',
       # 'Avg ticks for solving strong branching LPs for spatial branching (not including infeasible ones) Mixed',
       # '% vars in DAG (out of all vars) Int',
       # '% vars in DAG (out of all vars) Mixed']

diff_feature_x = [X]#, X[lin_top_ten], X[forrest_top_nine], X[global_mvps]]
feature_set_names = ['AllFeat']#, 'LinTop10', 'ForTop9', 'MVP']



for test_set in range(len(diff_feature_x)):
    feature_space = diff_feature_x[test_set]
    feat_space_name = feature_set_names[test_set]

    X_train, X_test, y_train, y_test = train_test_split(feature_space, y, test_size=0.2, random_state=75475)
    relevant_label_indices = [index for index in y_test.index if y_test.loc[index] != 1]
    impute_scale_predict(y_test, y_train, relevant_label_indices, feat_space_name, accuracy=True, plot_accuracy=True, cmp_time=False)





# top_features_for_reg = [item for sublist in for_top_feat for item in sublist]
# wurm = [item for sublist in lin_top_feat for item in sublist]
# counts = pd.Series(top_features_for_reg+wurm).value_counts()
# print(counts.iloc[0:6].index)


# train_set_df = create_time_df(clean_df, y_train.index)
# train_set_df = train_set_df[['Final solution time (cumulative) Mixed', 'Final solution time (cumulative) Int', 'Virtual Best']]
# test_set_df = tdf[['Final solution time (cumulative) Mixed', 'Final solution time (cumulative) Int', 'Virtual Best']]

"""Multiple Imputation
Multiple imputation is considered a good approach for data sets with a large amount of missing data. 
Instead of substituting a single value for each missing data point, the missing values are exchanged 
for values that encompass the natural variability and uncertainty of the right values. 
Using the imputed data, the process is repeated to make multiple imputed data sets. 
Each set is then analyzed using the standard analytical procedures and the results are combined to 
produce an overall result.

The various imputations incorporate natural variability into the missing values, which creates a 
valid statistical inference. Multiple imputations can produce statistically valid results even when
there is a small sample size or a large amount of missing data."""




"""Code for pairplot features"""
# train_dataset = X_train.copy()
# train_dataset.insert(0, "Cmp Final solution time (cumulative)", y_train)
# _ = sns.pairplot(
#     train_dataset[
#         ["Cmp Final solution time (cumulative)", 'Matrix Equality Constraints', 'Matrix Quadratic Elements',
#        'Matrix NLP Formula', '#MIP nodes Mixed', '#MIP nodes Int',]
#     ],
#     kind="reg",
#     diag_kind="kde",
#     plot_kws={"scatter_kws": {"alpha": 0.1}},
# )
# plt.show()