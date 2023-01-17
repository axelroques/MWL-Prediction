
from .hweak import HWeakClassifier

from sklearn.metrics import auc, roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

plt.rcParams["font.family"] = "serif"


class HBagging():

    def __init__(self):

        self.nb_classifiers = None
        self.tolerance = None
        self.features_used = []

    def fit(self, X_train, y_train, nb_classifiers, tolerance):

        self.nb_classifiers = nb_classifiers
        self.tolerance = tolerance

        self.values = []
        self.thresholds = []
        self.sides = []

        # Train a weak classifier on each feature
        for dim in list(range(X_train.shape[1])):

            weak_classifier = HWeakClassifier()
            weak_classifier.fit(X_train[:, dim], y_train, tolerance=tolerance)

            self.values.append(weak_classifier.value)
            self.thresholds.append(weak_classifier.threshold)
            self.sides.append(weak_classifier.side)

        self.values = np.array(self.values)

        # Features selection
        nb_valid = np.sum((self.values > 0))
        nb_to_select = min(self.nb_classifiers, nb_valid)
        self.features_used = np.argsort(self.values)[::-1][:nb_to_select]

    def predict_proba(self, X_test):

        y_predict = np.zeros(len(X_test))

        for dim in self.features_used:

            X_test_dim = X_test[:, dim]

            inds_right = (X_test_dim >= self.thresholds[dim])

            y_predict[inds_right] += 1

        prediction_class_one = np.array(
            y_predict/self.nb_classifiers).reshape(-1, 1)

        return np.concatenate([1-prediction_class_one, prediction_class_one], axis=1)

    def predict(self, X_test, threshold=0.5):

        proba_predictions = self.predict_proba(X_test)[:, 1]

        predictions = [1 if proba_predictions[i] >=
                       threshold else 0 for i in range(len(proba_predictions))]

        return predictions

    def draw(self, X, y, features_names, reversed_variables_names=None, fig_path=None, title=None, name_classes=None, density=False, rename_dic={}, **kwargs):

        if len(self.features_used) < 4:
            fig, axes = plt.subplots(
                nrows=len(self.features_used), ncols=1, figsize=(24, 16))
        elif len(self.features_used) < 7:
            fig, axes = plt.subplots(
                nrows=len(self.features_used), ncols=1, figsize=(24, 22))
        else:
            fig, axes = plt.subplots(
                nrows=len(self.features_used), ncols=1, figsize=(24, 26))

        for i in range(len(self.features_used)):

            threshold = float(self.thresholds[self.features_used[i]])
            side = str(self.sides[self.features_used[i]])

            if self.nb_classifiers > 1:
                axis = axes[i]
            else:
                axis = axes

            axis.set_title("Weak classifier n° "+str(i+1), fontsize=20)

            X_best_dim = X[:, self.features_used[i]]

            name = features_names[self.features_used[i]]

            if name in reversed_variables_names:
                X_best_dim = X_best_dim*-1
                threshold = threshold * -1

            inds_sorted = np.argsort(X_best_dim)
            class_inds_sorted = y[inds_sorted]
            X_best_dim_sorted = X_best_dim[inds_sorted]

#            distrib = FastKerDist()
#            distrib.fit(X_best_dim_sorted[class_ind_sorted==1])

            step = 0.01 * (np.max(X_best_dim) - np.min(X_best_dim))
            bins = np.arange(np.min(X_best_dim), np.max(X_best_dim)+step, step)

            if name_classes is None:
                name_classes = ["Class 0", "Class 1"]

            axis.hist(X_best_dim_sorted[class_inds_sorted == 1], color="red",
                      bins=bins, density=False, alpha=0.6, label=name_classes[1])
            axis.hist(X_best_dim_sorted[class_inds_sorted == 0], color="mediumseagreen",
                      bins=bins, density=False, alpha=0.6, label=name_classes[0])

            # axis.hist([X_best_dim_sorted[class_inds_sorted==1], X_best_dim_sorted[class_inds_sorted==0]], color=["red", "mediumseagreen"], bins=bins, alpha=0.6, label=[name_classes[1], name_classes[0]])

            # kde = gaussian_kde(X_best_dim_sorted[class_inds_sorted==0], bw_method=0.2)
            # distrib_class_zero = kde(bins)

            # kde = gaussian_kde(X_best_dim_sorted[class_inds_sorted==1], bw_method=0.2)
            # distrib_class_one = kde(bins)

            # axis.plot(bins, distrib_class_one, color="red", label=name_classes[1])
            # axis.fill_between(bins, distrib_class_one, color="red", alpha=0.2)

            # axis.plot(bins, distrib_class_zero, color="mediumseagreen", label=name_classes[0])
            # axis.fill_between(bins, distrib_class_zero, color="mediumseagreen", alpha=0.2)


#            axis.scatter(X_best_dim_sorted[class_inds_sorted==1],[0.025]*np.sum(class_inds_sorted==1),color="red",marker="x", s=100)
#            axis.scatter(X_best_dim_sorted[class_inds_sorted==0],[-0.025]*np.sum(class_inds_sorted==0),color="green",marker=".", s=100)
#

            color = "green" if side == "left" else "red"

            lims_y = axis.get_ylim()
            axis.plot([threshold, threshold], [lims_y[0], lims_y[1]],
                      color=color, linestyle="--", linewidth=4)

            lims_x = axis.get_xlim()
            range_x = lims_x[1] - lims_x[0]

            range_y = lims_y[1] - lims_y[0]

#            axis.plot([threshold, threshold], [-0.05, 0.05], color=color, linestyle="--")
            axis.text(s="Threshold : "+"%.2f" % threshold, x=threshold-0.05 *
                      range_x, y=lims_y[1]+0.2*range_y, fontsize=30, color=color)

#            axis.set_ylim([-0.15,0.15])
#
#            axis.set_xlabel(features_names[self.features_used[i]], fontsize=20)
#            axis.get_yaxis().set_visible(False)

#            minx = np.min(X_best_dim)
#            maxx = np.max(X_best_dim)
#            scale = np.max(X_best_dim)-np.min(X_best_dim)
#            axis.plot([minx-0.1*scale, maxx+0.1*scale], [0,0], color="black", linestyle="--")

            axis.set_ylim(lims_y[0], lims_y[1]+0.6*range_y)
#

            new_name = rename_dic[name] if name in rename_dic else name
            # +", threshold: "+"%3.f"%threshold, fontsize=30)
            axis.set_title(new_name, fontsize=40)
            axis.tick_params(labelsize=20)

        axes[0].legend(fontsize=40)

        fig.tight_layout()

        if fig_path is not None:

            if title is None:
                savepath = Path(fig_path, "hbagging")
            else:
                savepath = Path(fig_path, title)

            fig.savefig(savepath)

            # plt.close(fig)


if __name__ == "__main__":

    from sklearn.metrics import roc_auc_score

    X_train = np.random.random((200, 10))
    y_train = (X_train[:, 2] > 0.7) | (X_train[:, 5] > 0.35).astype(int)

    # Les labels y_train valent 1 lorsque la valeur de la 2eme colonne de X_train est > 0.7 ou lorsque
    # la valeur de la 5eme colonne est > 0.35.

    # L'algo utilise deux classifieurs et choisit d'abord la variable la plus discriminante

    hbagging = HBagging(nb_classifiers=2)
    hbagging.fit(X_train, y_train)

    print("nums of features used", hbagging.features_used)
    print("thresholds found", np.array(hbagging.thresholds)
          [np.array(hbagging.features_used)])

    X_test = np.random.random((50, 10))
    y_test = (X_test[:, 2] > 0.7) | (X_test[:, 5] > 0.35).astype(int)

    predictions = hbagging.predict(X_test)

    auc = roc_auc_score(y_true=y_test, y_score=predictions)

    print("AUC", auc)

    features_names = ["feature_num_"+str(i) for i in range(X_train.shape[1])]

    hbagging.draw(X_train, y_train, title="training",
                  features_names=features_names)
    hbagging.draw(X_test, y_test, title="testing",
                  features_names=features_names)
