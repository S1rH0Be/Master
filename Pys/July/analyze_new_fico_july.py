import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import gmean, wilcoxon
from sklearn.model_selection import train_test_split
import sklearn as sk

def shifted_geometric_mean(values, shift):
    values = np.array(values)

    if values.dtype == 'object':
        # Attempt to convert to float
        values = values.astype(float)

    # Shift the values by the constant
    # Check if shift is large enough
    if shift <= -values.min():
        print(f"Shift {shift} too small. Minimum value is {values.min()}, so shift must be > {-values.min()}")
        raise ValueError(f"Shift too small. Minimum value is {values.min()}, so shift must be > {-values.min()}")

    shifted_values = values + shift

    shifted_values_log = np.log(shifted_values)  # Step 1: Log of each element in shifted_values

    log_mean = np.mean(shifted_values_log)  # Step 2: Compute the mean of the log values
    geo_mean = np.exp(log_mean) - shift
    # geo_mean = np.round(geo_mean, 6)
    return geo_mean

def get_avas_of_dict(mean_dict, kind_of_mean):
    int_gmean = np.round(gmean(mean_dict["int"]), 2)
    vbs_mean = np.round(gmean(mean_dict["vbs"]), 2)

    print(f"{kind_of_mean}")
    print(int_gmean, vbs_mean)
    # print("Min/Max")
    # print(min(dict["int"]), max(dict["int"]))
    # print(min(dict["vbs"]), max(dict["vbs"]))

fico_5 = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/October/Bases/FICO/Cleaned/9_5_ready_to_ml.csv')
fico_6 = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/October/Bases/FICO/Cleaned/9_6_ready_to_ml.csv')

raw_5 = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/October/Bases/FICO/Raw/9.5_new_fritz_anon.csv')
raw_6 = pd.read_csv('/Users/fritz/Downloads/ZIB/Master/October/Bases/FICO/Raw/9.6_new_fritz_anon.csv')

mixed_time = fico_5['Final solution time (cumulative)'].sum()
int_time = fico_5['Final solution time (cumulative) Int'].sum()
hundred_seeds = [2207168494, 288314836, 1280346069, 1968903417, 1417846724, 2942245439, 2177268096, 571870743,
                     1396620602, 3691808733, 4033267948, 3898118442, 24464804, 882010483, 2324915710, 316013333,
                     3516440788, 535561664, 1398432260, 572356937, 398674085, 4189070509, 429011752, 2112194978,
                     3234121722, 2237947797, 738323230, 3626048517, 733189883, 4126737387, 2399898734, 1856620775,
                     829894663, 3495225726, 1844165574, 1282240360, 2872252636, 1134263538, 1174739769, 2128738069,
                     1900004914, 3146722243, 3308693507, 4218641677, 563163990, 568995048, 263097927, 1693665289,
                     1341861657, 1387819803, 157390416, 2921975935, 1640670982, 4226248960, 698121968, 1750369715,
                     3843330071, 2093310729, 1822225600, 958203997, 2478344316, 3925818254, 2912980295, 1684864875,
                     362704412, 859117595, 2625349598, 3108382227, 1891799436, 1512739996, 1533327828, 1210988828,
                     3504138071, 1665201999, 1023133507, 4024648401, 1024137296, 3118826909, 4052173232, 3143265894,
                     1584118652, 1023587314, 666405231, 2782652704, 744281271, 3094311947, 3882962880, 325283101,
                     923999093, 4013370079, 2033245880, 289901203, 3049281880, 1507732364, 698625891, 1203175353,
                     1784663289, 2270465462, 537517556, 2411126429]


means = {"int":[], "mixed":[], "vbs":[]}
gmeans = {"int":[], "mixed":[], "vbs":[]}
sgmeans = {"int":[], "mixed":[], "vbs":[]}
def create_dicts(dataset, set_name):
    print(set_name)
    count = 0
    for seed in hundred_seeds:
        X_train, X_test, y_train, y_test = train_test_split(dataset, dataset['Cmp Final solution time (cumulative)'], test_size=0.2,
                                                            random_state=seed)
        mixed_mean = X_test['Final solution time (cumulative)'].mean()
        int_mean = X_test['Final solution time (cumulative) Int'].mean()
        vbs_mean = X_test['Virtual Best'].mean()

        mixed_sum = dataset['Final solution time (cumulative)'].sum()
        mixed_train = X_train['Final solution time (cumulative)'].sum()
        mixed_test = X_test['Final solution time (cumulative)'].sum()
        int_sum = dataset['Final solution time (cumulative) Int'].sum()
        int_train = X_train['Final solution time (cumulative) Int'].sum()
        int_test = X_test['Final solution time (cumulative) Int'].sum()
        count += mixed_test - int_test
        mean_mean = (mixed_mean+int_mean+vbs_mean)/3

        mixed_gmean = gmean(X_test['Final solution time (cumulative)'])
        int_gmean = gmean(X_test['Final solution time (cumulative) Int'])
        vbs_gmean = gmean(X_test['Virtual Best'])

        mixed_sgm = shifted_geometric_mean(X_test['Final solution time (cumulative)'], mean_mean)
        int_sgm = shifted_geometric_mean(X_test['Final solution time (cumulative) Int'], mean_mean)
        vbs_sgm = shifted_geometric_mean(X_test['Virtual Best'], mean_mean)

        gmeans["int"].append(np.round(int_gmean/mixed_gmean, 2))
        gmeans["mixed"].append(np.round(mixed_gmean/mixed_gmean, 2))
        gmeans["vbs"].append(np.round(vbs_gmean/mixed_gmean, 2))

        means["int"].append(np.round(int_mean / mixed_mean, 2))
        means["mixed"].append(np.round(mixed_mean / mixed_mean, 2))
        means["vbs"].append(np.round(vbs_mean / mixed_mean, 2))

        sgmeans["int"].append(np.round(int_sgm / mixed_sgm, 2))
        sgmeans["mixed"].append(np.round(mixed_sgm / mixed_sgm, 2))
        sgmeans["vbs"].append(np.round(vbs_sgm / mixed_sgm, 2))
    get_avas_of_dict(means, "Mean")
    get_avas_of_dict(gmeans, "GeoMean")
    get_avas_of_dict(sgmeans, "SGM")

def raw_stats(dataset, version):
    if version == "5":
        mixed_sum = dataset['Final solution time (cumulative) (Fritz Global PR - Public Discrete Nonconvex def)'].sum()
        int_sum = dataset['Final solution time (cumulative) (Fritz Global PR - Public Discrete Nonconvex GLOBALSPATIALBRANCHIFPREFERINT=1)'].sum()
    else:
        mixed_sum = dataset['Final solution time (cumulative) (Fritz 9.6 Global Pull Request default)'].sum()
        int_sum = dataset['Final solution time (cumulative) (Fritz 9.6 Global Pull Request GLOBALSPATIALBRANCHIFPREFERINT=1)'].sum()
    print(mixed_sum, int_sum)

# raw_stats(raw_5, "5")
# raw_stats(raw_6, "6")
#
# create_dicts(fico_5, "5")
# create_dicts(fico_6, "6")

def plot(series, color):
    plt.figure(figsize=(8, 5))
    plt.hist(series.dropna(), bins=30, color=color, alpha=0.7, density=False)
    #plt.title('Histogram of values')
    plt.xlabel('value')
    plt.ylabel('count')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()
    plt.close()

def create_time_df(dataset, version):
    new_df = dataset.copy()
    time_cols = ["Final solution time (cumulative)",
                 "Final solution time (cumulative) Int",
                 "Virtual Best",
                 "Cmp Final solution time (cumulative)"]

    time_df = new_df[time_cols]
    time_df.to_csv(f"/Users/fritz/Downloads/ZIB/Master/October/Bases/FICO/Cleaned/time_{version}.csv", index=False)
    print(time_df["Virtual Best"].min())



sumtime_5 = pd.read_csv("/Users/fritz/Downloads/ZIB/Master/October/Runs/Final/FICO5/NoOutlier/Logged/ScaledLabel/SGM/sumtime.csv")
pos = 0
neg = 0
for index, row in sumtime_5.iterrows():
    if row["Mixed-Int"]>=0:
        pos += row["Mixed-Int"]
    elif row["Mixed-Int"]<0:
        neg += row["Mixed-Int"]
    else:
        pass

fico_5_int = 0
fico_5_mixed = 0
fico_5_vbs = 0
fico_6_int = 0
fico_6_mixed = 0
fico_6_vbs = 0

for i in fico_5['Final solution time (cumulative) Int']:
    fico_5_int += i
for i in fico_5['Final solution time (cumulative)']:
    fico_5_mixed += i
for i in fico_5['Virtual Best']:
    fico_5_vbs += i

for i in fico_6['Final solution time (cumulative) Int']:
    fico_6_int += i
for i in fico_6['Final solution time (cumulative)']:
    fico_6_mixed += i
for i in fico_6['Virtual Best']:
    fico_6_vbs += i


import pandas as pd
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error


def label_scaling(label):
    y_pos = label[label >= 0]
    y_neg = label[label < 0]
    y_pos_log = np.log(y_pos + 1)
    y_neg_log = np.log(abs(y_neg) + 1) * -1
    y_log = pd.concat([y_pos_log, y_neg_log]).sort_index()
    return y_log


prediction = pd.read_csv("/Users/fritz/Downloads/ZIB/Master/October/Runs/Final/FightOverfitting/FICO6/BestCombi/NEWPRED/NoOutlier/Logged/ScaledLabel/Prediction/fico_prediction_df.csv")
data = pd.read_csv("/Users/fritz/Downloads/ZIB/Master/October/Bases/FICO/Cleaned/9_6_ready_to_ml.csv")
hundred_seeds = [2207168494, 288314836, 1280346069, 1968903417, 1417846724, 2942245439, 2177268096, 571870743,
                     1396620602, 3691808733, 4033267948, 3898118442, 24464804, 882010483, 2324915710, 316013333,
                     3516440788, 535561664, 1398432260, 572356937, 398674085, 4189070509, 429011752, 2112194978,
                     3234121722, 2237947797, 738323230, 3626048517, 733189883, 4126737387, 2399898734, 1856620775,
                     829894663, 3495225726, 1844165574, 1282240360, 2872252636, 1134263538, 1174739769, 2128738069,
                     1900004914, 3146722243, 3308693507, 4218641677, 563163990, 568995048, 263097927, 1693665289,
                     1341861657, 1387819803, 157390416, 2921975935, 1640670982, 4226248960, 698121968, 1750369715,
                     3843330071, 2093310729, 1822225600, 958203997, 2478344316, 3925818254, 2912980295, 1684864875,
                     362704412, 859117595, 2625349598, 3108382227, 1891799436, 1512739996, 1533327828, 1210988828,
                     3504138071, 1665201999, 1023133507, 4024648401, 1024137296, 3118826909, 4052173232, 3143265894,
                     1584118652, 1023587314, 666405231, 2782652704, 744281271, 3094311947, 3882962880, 325283101,
                     923999093, 4013370079, 2033245880, 289901203, 3049281880, 1507732364, 698625891, 1203175353,
                     1784663289, 2270465462, 537517556, 2411126429]
wilcoxon_lin = []
wilcoxon_for = []
for i in range(len(hundred_seeds)):
    pred = prediction.copy()
    pred_lin = pred.iloc[:,i+1]
    pred_lin = pred_lin[pred_lin!=0]
    pred_for = pred.iloc[:,i+101]
    pred_for=pred_for[pred_for!=0]

    test_lin = label_scaling(data['Cmp Final solution time (cumulative)'].loc[pred_lin.index])
    test_for = label_scaling(data['Cmp Final solution time (cumulative)'].loc[pred_for.index])
    # if stats.wilcoxon(test_lin, pred_lin)[1] < 0.05:
    #     print("Lin", i + 1, hundred_seeds[i])
    #     print(stats.wilcoxon(test_lin, pred_lin))
    # wilcoxon_lin.append(stats.wilcoxon(test_lin, pred_lin)[0])
    # if stats.wilcoxon(test_for, pred_for)[1] < 0.05:
    #     print("for", i + 101, hundred_seeds[i])
    #     print(stats.wilcoxon(test_for, pred_for))
    wilcoxon_for.append(stats.wilcoxon(test_for, pred_for)[0])
    # print(sk.metrics.r2_score(test_lin, pred_lin))
    # print(sk.metrics.r2_score(test_for, pred_for))
    print("------------------------------------------------------------------------------------------")
    print(mean_squared_error(test_for, pred_for))
    print(root_mean_squared_error(test_for, pred_for))
    print(r2_score(test_for, pred_for))
    print(mean_absolute_percentage_error(test_for, pred_for))


