
from .hbagging.hbagging import HBagging

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas

np.random.seed(8)


def fit_classification(X_train, y_train, algo, params_grid, n_cross_val_splits):

    if algo == "hbagging":

        auc_grid = []

        for nb_classifiers in params_grid["nb_classifiers"]:

            for tolerance in params_grid["tolerance"]:

                if nb_classifiers > X_train.shape[1]:
                    auc_grid.append((0, nb_classifiers, tolerance))
                    continue

                model_grid = HBagging()

                cross_val_scores = []

                skf = StratifiedKFold(n_splits=n_cross_val_splits)

                for train_index, test_index in skf.split(X_train, y_train):
                    X_train_cv, X_val_cv = X_train[train_index], X_train[test_index]
                    y_train_cv, y_val_cv = y_train[train_index], y_train[test_index]

                    model_grid.fit(
                        X_train_cv, y_train_cv, nb_classifiers=nb_classifiers, tolerance=tolerance)

                    auc_score = roc_auc_score(
                        y_true=y_val_cv, y_score=model_grid.predict_proba(X_val_cv)[:, 1])
                    cross_val_scores.append(auc_score)

                auc_grid.append((np.mean(cross_val_scores),
                                 nb_classifiers, tolerance))

        sorted_auc_grid = sorted(auc_grid, key=lambda tup: tup[0])[::-1]

        best_nb_classifiers = sorted_auc_grid[0][1]
        best_tolerance = sorted_auc_grid[0][2]

        model = HBagging()
        model.fit(X_train, y_train, nb_classifiers=best_nb_classifiers,
                  tolerance=best_tolerance)

        features_used = model.features_used

        importances = [
            1 if i in features_used else 0 for i in range(X_train.shape[1])]

    if algo == "logistic":

        grid = GridSearchCV(LogisticRegression(penalty='l1', solver='liblinear'), params_grid,
                            cv=n_cross_val_splits, scoring="roc_auc")

        grid.fit(X_train, y_train)

        keys_attributes = grid.cv_results_.keys()

        df_results = pandas.DataFrame.from_dict(grid.cv_results_)

        model = grid.best_estimator_
        model.fit(X_train, y_train)

        importances = model.coef_[0]

    if algo == "forest":

        grid = GridSearchCV(RandomForestClassifier(), params_grid,
                            cv=n_cross_val_splits, scoring="roc_auc")

        grid.fit(X_train, y_train)

        keys_attributes = grid.cv_results_.keys()

        df_results = pandas.DataFrame.from_dict(grid.cv_results_)

        model = grid.best_estimator_
        model.fit(X_train, y_train)

        importances = model.feature_importances_

    return model, importances
