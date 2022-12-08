







import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
from scipy.signal import convolve2d
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.metrics import roc_curve, auc
from collections import OrderedDict
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, roc_curve
from to_rmv.features import reverse_features

from paths import savepath, figpath
from load_data import Ikky_data
from load_data import cols_NASA_tlx, cols_theoretical_NASA_tlx, cols_time_NASA_tlx
#from ML_tools.classification.hbagging_run import hbagging_classification
#from ML_tools.classification.classic_methods import logistic_lasso_classification, random_forest_classification
from scipy.stats import pearsonr
import itertools

list_colors=["red","magenta","blue","darkblue","orange","grey","brown","darkred","darkgreen", "black", "darkorchid", "lightsalmon", "skyblue"]

font = {'family':'serif',
        'weight' : 'normal'}
matplotlib.rc('font', **font)


cross_validations_splits = 5
train_prop = 0.8

np.random.seed(25)


db = Ikky_data(reset=False)
table = db.get_theo_NASA_tlx_table()


NASA_tlx = "mean_NASA_tlx"
theo_NASA_tlx = "mean_theoretical_NASA_tlx"
tc = "oral_tc"

cols = [NASA_tlx, theo_NASA_tlx, tc]

all_cols = ["oral_tc", 'mental_demand', 'physical_demand', 'temporal_demand', 'effort',
 'performance', 'frustration', 'theoretical_mental_demand',
 'theoretical_physical_demand', 'theoretical_temporal_demand',
 'theoretical_effort', 'mean_NASA_tlx', 'mean_theoretical_NASA_tlx', "flight_hours"]


all_valid = ((~table[cols].isna()).product(axis=1)).astype(bool)

values = table.loc[all_valid, cols]

print(values)

bins = np.arange(len(values))






fig, ax = plt.subplots(1, figsize=(15,10))

ax.plot(bins, values["mean_theoretical_NASA_tlx"], color="mediumseagreen", alpha=0.8)
ax.plot(bins, values["mean_NASA_tlx"], color="dodgerblue", alpha=0.8)
ax.plot(bins, values["oral_tc"], color="indianred", alpha=0.8)

ax.scatter(bins, values["mean_theoretical_NASA_tlx"], color="mediumseagreen", label="Mean theoretical NASA tlx")
ax.scatter(bins, values["mean_NASA_tlx"], color="dodgerblue", label="Mean NASA tlx")
ax.scatter(bins, values["oral_tc"], color="indianred", label="Oral auto-evaluation")

ax.legend(fontsize=15)
ax.set_xticks([])
ax.tick_params("y", labelsize=15)

fig.tight_layout()

fig.savefig(Path(figpath, "comparison_scores.pdf"))


for score1, score2 in list(itertools.combinations(all_cols, 2)):
    
    all_valid_score_1_score2 = ((~table[[score1, score2]].isna()).product(axis=1)).astype(bool)

    correlation, pvalue = pearsonr(table.loc[all_valid_score_1_score2, score1].values, table.loc[all_valid_score_1_score2, score2])
    
    if pvalue < 0.01 and "mean" not in score1 and "mean" not in score2:
        print("Correlation between "+score1+" and "+score2+": "+str(correlation)+" (pvalue="+str(pvalue)+")")


    fig, ax = plt.subplots(1, figsize=(10,10))
    ax.scatter(table.loc[all_valid, score1].values, table.loc[all_valid, score2].values)
    ax.set_xlabel(score1, fontsize=25)
    ax.set_ylabel(score2, fontsize=25)
    fig.savefig(Path(figpath, "scores_comparison", "Scatter_comparison_"+score1+"_"+score2+".png"))
    
    plt.close(fig)


for pilot in table["pilot"].unique():
    
    where_pilot = (table["pilot"]==pilot)
    
    values = table.loc[all_valid & where_pilot, cols]
    
    bins = np.arange(len(values))


    fig, ax = plt.subplots(1, figsize=(15,10))

    ax.plot(bins, values["mean_theoretical_NASA_tlx"], color="mediumseagreen", alpha=0.8)
    ax.plot(bins, values["mean_NASA_tlx"], color="dodgerblue", alpha=0.8)
    ax.plot(bins, values["oral_tc"], color="indianred", alpha=0.8)

    ax.scatter(bins, values["mean_theoretical_NASA_tlx"], color="mediumseagreen", label="Mean theoretical NASA tlx")
    ax.scatter(bins, values["mean_NASA_tlx"], color="dodgerblue", label="Mean NASA tlx")
    ax.scatter(bins, values["oral_tc"], color="indianred", label="Oral auto-evaluation")

    ax.legend(fontsize=15)
    ax.set_xticks([])
    ax.tick_params("y", labelsize=15)

    fig.tight_layout()

    fig.savefig(Path(figpath, "scores_comparison", "comparison_scores_pilot_"+pilot+".png"))
