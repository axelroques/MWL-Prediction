

import numpy as np
from pathlib import Path
import matplotlib

from features import reverse_features, dic_variables

from paths import figpath
from load_data import Ikky_data

from ML_tools.classification.ikky_classification import run_ikky_classification


list_colors = ["red", "magenta", "blue", "darkblue", "orange", "grey", "brown",
               "darkred", "darkgreen", "black", "darkorchid", "lightsalmon", "skyblue"]

font = {'family': 'serif',
        'weight': 'normal'}
matplotlib.rc('font', **font)


cross_validations_splits = 5
train_prop = 0.8


normalized_labels = True

name_classes = ["Low cognitive load", "High cognitive load"]

db = Ikky_data(reset=False)
print('db =', db)
table = db.get_tc_table()
print('table =', table)


# and "ADF" not in col]# and "blinks_duration" not in col] #+ ["mean_theoretical_NASA_tlx"]  #+["flight_hours]
features_cols = [
    col for col in table.columns if "feature" in col and "NASA_tlx" not in col]

valid_indexes = ((~table[features_cols].isna()).product(axis=1)).astype(bool)


X = table[valid_indexes][features_cols]

# print(table[valid_indexes]["pilot"].unique())

if not normalized_labels:
    y = table[valid_indexes]["oral_tc"].values

    y_raw = y.copy()

    y[y < 35] = 0
    y[y >= 35] = 1

    y = y.astype(int)

else:
    y = table[valid_indexes]["binary_normalized_oral_tc"].values

    y_raw = table[valid_indexes]["normalized_oral_tc"].values


actual_ground_truth = "oral_declaration"

dic_variables = {"features": features_cols}
# dic_variables = {key:["feature_tc_fixed_windows_"+feature for feature in list_features] for key, list_features in dic_variables.items()}


rename_dic = {}


""""""""""""""""""""""""""""""""""""""""""""""""
"""HBagging"""
""""""""""""""""""""""""""""""""""""""""""""""""
print("------")
print("HBAGGING")

X_HBagging = X.copy()

X_HBagging.iloc[:, np.array([list(X.columns).index("feature_tc_fixed_windows_"+feature)
                             for feature in reverse_features if "feature_tc_fixed_windows_"+feature in list(X.columns)])] *= -1

# X_HBagging["flight_hours"] *= -1

params_grid = {"nb_classifiers": np.arange(2, 11), "tolerance": [0.1]}

auc_mean, auc_std, mean_individual_auc, median_individual_auc = run_ikky_classification(X_HBagging, y, dic_variables=dic_variables, algo="hbagging", params_grid=params_grid, train_prop=0.8,
                                                                                        name_classes=name_classes, num_iterations=300, n_cross_val_splits=5,
                                                                                        variables_names=features_cols, rename_dic=rename_dic, stratify_=True, plot=True,
                                                                                        fig_path=Path(figpath, "model"), title="", reversed_variables_names=["features_tc_fixed_windows_"+feature for feature in reverse_features]+["flight_hours"],
                                                                                        remove_helico=False, remove_commands=False,
                                                                                        remove_cardio=False, remove_respi=False, remove_blinks=False,
                                                                                        remove_oculo=False, remove_aoi=False,
                                                                                        ground_truth="tc", add_noise=False, cv_scheme="iB",
                                                                                        actual_ground_truth=actual_ground_truth)
#
#wit(median_individual_auc, actual_ground_truth, "whole dataset", auc_mean, auc_std)

print("------")
print("---------")
print("-------")
