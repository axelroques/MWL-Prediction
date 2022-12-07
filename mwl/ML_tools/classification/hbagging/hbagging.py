
from .hweak import HWeakClassifier

from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

plt.rcParams["font.family"] = "serif"


class HBagging():

    def __init__(self):

        self.nb_classifiers = None
        self.tolerance = None
        self.features_used = []

    def fit(self, X_train, y_train, nb_classifiers=3, tolerance=0.1, list_features1=None, list_features2=None):

        self.nb_classifiers = nb_classifiers
        self.tolerance = tolerance

        self.values = []
        self.thresholds = []
        self.sides = []

        for dim in list(range(X_train.shape[1])):

            if dim == 26:
                verbose = True
            else:
                verbose = False

            weak_classifier = HWeakClassifier()
            weak_classifier.fit(X_train[:, dim], y_train, tolerance=tolerance)

            self.values.append(weak_classifier.value)
            self.thresholds.append(weak_classifier.threshold)

            self.sides.append(weak_classifier.side)

        self.values = np.array(self.values)

        if list_features1 is not None and list_features2 is not None:

            nb_valid = np.sum((self.values > 0.6))

            nb_to_select = max(self.nb_classifiers, nb_valid)

            sorted_dims = np.argsort(self.values)[::-1]

            self.features_used = []

            used_features1 = []
            used_features2 = []

            for dim in sorted_dims:

                if dim in list_features1:
                    if len(used_features1) > int(nb_classifiers/2):
                        continue

                    used_features1.append(dim)

                if dim in list_features2:
                    if len(used_features2) > int(nb_classifiers/2):
                        continue

                    used_features2.append(dim)

                self.features_used.append(dim)

            self.features_used = np.array(
                self.features_used[:self.nb_classifiers])

        else:

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

    def draw_old(self, X, y, features_names, path_fig=None, title=None, **kwargs):

        fig, axes = plt.subplots(
            nrows=self.nb_classifiers, ncols=1, figsize=(24, 18))

        for i in range(self.nb_classifiers):

            threshold = self.thresholds[self.features_used[i]]
            side = self.sides[self.features_used[i]]

            if self.nb_classifiers > 1:
                axis = axes[i]
            else:
                axis = axes

            axis.set_title("Weak classifier n° "+str(i+1), fontsize=20)

            X_best_dim = X[:, self.features_used[i]]

            inds_sorted = np.argsort(X_best_dim)
            class_inds_sorted = y[inds_sorted]
            X_best_dim_sorted = X_best_dim[inds_sorted]

            axis.scatter(X_best_dim_sorted[class_inds_sorted == 1], [
                         0.025]*np.sum(class_inds_sorted == 1), color="red", marker="x", s=100)
            axis.scatter(X_best_dim_sorted[class_inds_sorted == 0], [-0.025]*np.sum(
                class_inds_sorted == 0), color="green", marker=".", s=100)

            color = "green" if side == "left" else "red"

            axis.plot([threshold, threshold], [-0.05, 0.05],
                      color=color, linestyle="--")
            axis.text(s="Threshold : "+"%.2f" % threshold,
                      x=threshold-0.05, y=0.1, fontsize=14)

            axis.set_ylim([-0.15, 0.15])

            axis.set_xlabel(features_names[self.features_used[i]], fontsize=20)
            axis.get_yaxis().set_visible(False)

            minx = np.min(X_best_dim)
            maxx = np.max(X_best_dim)
            scale = np.max(X_best_dim)-np.min(X_best_dim)
            axis.plot([minx-0.1*scale, maxx+0.1*scale],
                      [0, 0], color="black", linestyle="--")

            axis.set_title(
                features_names[self.features_used[i]]+", threshold: "+str(threshold), fontsize=20)

        if title is not None:
            fig.suptitle(title)
        fig.subplots_adjust(hspace=0.5)

        if path_fig is not None:
            fig.savefig(path_fig)

        fig.show()


def hbagging_classification(X, y, params_grid, train_prop=0.8,
                            num_iterations=10, cross_validations_splits=10,
                            variable_names=None, stratify_=True, plot=True,
                            nb_classifiers=10, tolerance=0.1, optimize_hyperparameters=False,
                            fig_path=None):

    auc_test = []
    selected_models = []

    strat = y if stratify_ else None

    if variable_names is None:
        variable_names = ["var "+str(i) for i in range(X.shape[1])]

    dic_importances = {var: [] for var in variable_names}

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(1)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Random classifier', alpha=.8)

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    for k in range(num_iterations):

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=1-train_prop,
                                                            stratify=strat,
                                                            shuffle=True)


#        mean_X_train = np.mean(X_train, axis=0)
#        std_X_train = np.std(X_train, axis=0)
#
#        X_train = (X_train - mean_X_train)
#        X_test = (X_test - mean_X_train)
#
#        X_train = X_train / std_X_train
#        X_test = X_test / std_X_train

        if optimize_hyperparameters:

            auc_grid = []

            for nb_classifiers in params_grid["nb_classifiers"]:

                for tolerance in params_grid["tolerance"]:

                    if nb_classifiers > X_train.shape[1]:
                        auc_grid.append((0, nb_classifiers, tolerance))
                        continue

                    model_grid = HBagging()

                    cross_val_scores = []

                    for v in range(cross_validations_splits):

                        X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(X_train, y_train,
                                                                                      test_size=1-train_prop,
                                                                                      stratify=y_train,
                                                                                      shuffle=True)

                        model_grid.fit(
                            X_train_cv, y_train_cv, nb_classifiers=nb_classifiers, tolerance=tolerance)

                        auc_score = roc_auc_score(
                            y_true=y_val_cv, y_score=model_grid.predict_proba(X_val_cv)[:, 1])
                        cross_val_scores.append(
                            (auc_score, nb_classifiers, tolerance))
                    sorted_auc_cross_val = sorted(
                        cross_val_scores, key=lambda tup: tup[0])[::-1]

                    best_nb_classifiers_val = sorted_auc_cross_val[0][1]
                    best_tolerance_val = sorted_auc_cross_val[0][2]
                    best_score_val = sorted_auc_cross_val[0][0]

                    auc_grid.append(
                        (best_score_val, best_nb_classifiers_val, best_tolerance_val))

            sorted_auc_grid = sorted(auc_grid, key=lambda tup: tup[0])[::-1]

            best_nb_classifiers = sorted_auc_grid[0][1]
            best_tolerance = sorted_auc_grid[0][2]

        else:
            best_nb_classifiers = nb_classifiers
            best_tolerance = tolerance

        print("nb_classifiers:", best_nb_classifiers,
              ", tolerance:", best_tolerance)

        model = HBagging()

        model.fit(X_train, y_train, nb_classifiers=best_nb_classifiers,
                  tolerance=best_tolerance)

        selected_models.append(model)

        fpr, tpr, thresholds = roc_curve(
            y_true=y_test, y_score=model.predict_proba(X_test)[:, 1])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        auc_score = roc_auc_score(
            y_true=y_test, y_score=model.predict_proba(X_test)[:, 1])
        auc_test.append(auc_score)

        variables_used = model.features_used

        for i, var in enumerate(variable_names):
            dic_importances[var].append(1 if i in variables_used else 0)

        if plot:
            model.draw(X_train, y_train, title="training_"+str(k),
                       features_names=variable_names, fig_path=fig_path)
            model.draw(X_test, y_test, title="testing_"+str(k),
                       features_names=variable_names, fig_path=fig_path)

    mean_importances = [np.mean(dic_importances[var])
                        for var in variable_names]
    indices = np.argsort(np.abs(mean_importances))[::-1]
    sorted_variables = np.array(variable_names)[indices]

    print("Poids des variables dans le modèle :\n")

    for i, var in enumerate(sorted_variables):
        variables_list_importances = dic_importances[var]
#        if np.abs(np.mean(variables_list_importances)) < 0.05*np.max(np.abs(mean_importances)):
#            continue
        if np.abs(np.mean(variables_list_importances)) == 0:
            continue
        print(var, ", poids moyen :", "%.3f" % np.mean(variables_list_importances),
              ", écart-type :", "%.3f" % np.std(variables_list_importances))

    print("AUC moyenne sur les ensembles de test :", "%.2f" %
          np.mean(auc_test), "écart-type", "%.2f" % np.std(auc_test))

    ax.set_ylabel("True positive rate", fontsize=25)
    ax.set_xlabel("False positive rate", fontsize=25)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='lightgrey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set_aspect("equal")

    ax.legend(loc="lower right", fontsize=20)


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
