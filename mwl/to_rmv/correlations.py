import numpy as np
import seaborn as sn
from load_data import Ikky_data
from to_rmv.features import reverse_features
#import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt


normalized_labels = True

name_classes = ["Low cognitive load", "High cognitive load"]

db = Ikky_data(reset=False)
table = db.get_tc_table()


# and "ADF" not in col]# and "blinks_duration" not in col] #+ ["mean_theoretical_NASA_tlx"]  #+["flight_hours]
features_cols = [
    col for col in table.columns if "feature" in col and "NASA_tlx" not in col]

valid_indexes = ((~table[features_cols].isna()).product(axis=1)).astype(bool)


X = table[valid_indexes][features_cols]

X_HBagging = X.copy()

X_HBagging.iloc[:, np.array([list(X.columns).index("feature_tc_fixed_windows_"+feature)
                             for feature in reverse_features if "feature_tc_fixed_windows_"+feature in list(X.columns)])] *= -1

# X_HBagging

X_3 = X_HBagging.iloc[:18, :]
X_4 = X_HBagging.iloc[18:33, :]
X_5 = X_HBagging.iloc[33:45, :]
X_6 = X_HBagging.iloc[45:64, :]
X_7 = X_HBagging.iloc[64:80, :]
X_8 = X_HBagging.iloc[80:98, :]
X_9 = X_HBagging.iloc[98:, :]

plt.rcParams["figure.figsize"] = (60, 30)

#df_pil2 = pd.DataFrame(data_pil2, columns = ['oral', 'boitier', 'mental', 'physical', 'temporal', 'effort', 'performance', 'frustration', 'mean_declared', 'theorical_mental', 'theorical_physical', 'theorical_temporal', 'theorical_effort', 'theorical_mean'])

# Refaire avec les p-values plutôt qu'avec les correlations
# une heatmap moyenne, une standard deviation


def corrMatrix(df, idx="3"):
    plt.figure()
    sn.heatmap(df.corr(), annot=True)
    plt.title("pilot "+idx+" Pearson correlation coefficient")
    plt.show()

    pval = df.corr().copy()

    for i in range(df.shape[1]):
        for j in range(df.shape[1]):
            try:
                y = df.columns[i]
                x = df.columns[j]
                df_ols = sm.ols(
                    formula='Q("{}") ~ Q("{}")'.format(y, x), data=df).fit()
                pval.iloc[i, j] = df_ols.pvalues[1]
            except ValueError:
                pval.iloc[i, j] = None

    plt.figure()
    plt.title("pilot "+idx+" p-value of correlation matrix")
    sn.heatmap(pval, center=0, cmap="Blues", annot=True)
    plt.show()


corrMatrix(X_HBagging, "all")
corrMatrix(X_3, "3")
corrMatrix(X_4, "4")
corrMatrix(X_5, "5")
corrMatrix(X_6, "6")
corrMatrix(X_7, "7")
corrMatrix(X_8, "8")
corrMatrix(X_9, "9")
