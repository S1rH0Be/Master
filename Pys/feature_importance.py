from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def importance_bar(df, title):
    features = df['Feature']
    for col in df.columns[1:]:
        values = df[col]

        # Create the bar plot
        plt.figure(figsize=(8, 6))
        #color bars according to the branching rule they belong to
        colors = []
        for feature in features:
            if 'Mixed' in feature:
                colors.append('magenta')
            elif 'Int' in feature:
                colors.append('green')
            else:
                colors.append('orange')

        plt.bar(features, values, color=colors, alpha=0.7, edgecolor='black')
        #remove x ticks
        # Remove x-axis ticks
        plt.xticks([])
        # Add labels and title
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        plt.title(title, fontsize=12)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', markersize=10, label='Mixed'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Int'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Both')]

        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        # Show the plot
        plt.show()
        plt.close()

def feature_importance_linreg(features, model, scaling, imputation):
    importance_df = pd.DataFrame({'Feature': features, 'Importance': model.coef_})
    #importance_bar(importance_df, "Feature Importance: LinReg imputed by "+imputation+" and scaled by "+scaling)
    return importance_df

def feature_importance_forrest(features, model, scaling, imputation):
    importance_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
    #importance_bar(importance_df, "Feature Importance: ForrestReg imputed by " + imputation + " and scaled by " + scaling)
    return importance_df

def bar(values, bar_names, title):
    plt.bar(bar_names, values, align='center')
    plt.title(title)
    plt.xticks(range(len(bar_names)), bar_names)
    plt.show()
    plt.close()


linimp = pd.read_excel(f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/linear_importances_yeo_20_01.xlsx').drop('Feature', axis=1)
forimp = pd.read_excel(f'/Users/fritz/Downloads/ZIB/Master/GitCode/Master/CSVs/forest_importances_20_01.xlsx').drop('Feature', axis=1)

# Calculate row-wise variance
lin_row_variance = linimp.var(axis=1)
lin_noscaling_variance = linimp.iloc[:,0::3].var(axis=1)
lin_byhand_variance = linimp.iloc[:,1::3].var(axis=1)
lin_yeojohn_variance = linimp.iloc[:,2::3].var(axis=1)

bar(lin_row_variance.values, [str(i) for i in range(len(lin_row_variance.values))], 'Linear Variance Total')
bar(lin_noscaling_variance.values, [str(i) for i in range(len(lin_row_variance.values))], 'Linear Variance NoScaling')
bar(lin_byhand_variance.values, [str(i) for i in range(len(lin_row_variance.values))], 'Linear Variance byHand')
bar(lin_yeojohn_variance.values, [str(i) for i in range(len(lin_row_variance.values))], 'Linear Variance Yeo-Johnson')




# Calculate row-wise minimum
lin_row_min = linimp.min(axis=1)
# Calculate row-wise maximum
lin_row_max = linimp.max(axis=1)
# Calculate row-wise mean
lin_row_mean = linimp.mean(axis=1)



# Calculate row-wise variance
for_row_variance = forimp.var(axis=1)
# Calculate row-wise minimum
for_row_min = forimp.min(axis=1)
# Calculate row-wise maximum
for_row_max = forimp.max(axis=1)
# Calculate row-wise mean
for_row_mean = forimp.mean(axis=1)


