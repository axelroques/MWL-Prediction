
from .hbagging.hbagging import HBagging

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import numpy as np

np.random.seed(8)


def fit(
    X_train, y_train,
    n_classifiers=None,
    tolerance=0.1,
    n_cross_val_splits=5
):
    """
    Fit HBagging algorithm.
    """

    n_features = X_train.shape[1]

    # Choose to restrict the number of features to use
    if n_classifiers:
        n_max_features = n_classifiers
    else:
        n_max_features = n_features

    # Test HBagging classification with different numbers of features
    n_classifier_list = np.arange(2, n_max_features+1)
    auc_grid = []
    for n_classifiers in n_classifier_list:

        # No classification if n_classifiers < n_features
        if n_classifiers > n_features:
            auc_grid.append((0, n_classifiers, tolerance))
            continue

        model_grid = HBagging()
        skf = StratifiedKFold(n_splits=n_cross_val_splits)

        cross_val_scores = []
        for train_index, test_index in skf.split(X_train, y_train):
            X_train_cv, X_val_cv = X_train[train_index], X_train[test_index]
            y_train_cv, y_val_cv = y_train[train_index], y_train[test_index]

            model_grid.fit(
                X_train_cv, y_train_cv,
                nb_classifiers=n_classifiers,
                tolerance=tolerance
            )

            auc_score = roc_auc_score(
                y_true=y_val_cv,
                y_score=model_grid.predict_proba(X_val_cv)[:, 1]
            )
            cross_val_scores.append(auc_score)

        auc_grid.append(
            (np.mean(cross_val_scores), n_classifiers, tolerance)
        )

    # Get best parameters
    sorted_auc_grid = sorted(auc_grid, key=lambda tup: tup[0])[::-1]
    best_n_classifiers = sorted_auc_grid[0][1]
    best_tolerance = sorted_auc_grid[0][2]

    # Final fit with the best parameters
    model = HBagging()
    model.fit(
        X_train, y_train,
        nb_classifiers=best_n_classifiers,
        tolerance=best_tolerance
    )

    features_used = model.features_used

    importances = [
        1 if i in features_used else 0 for i in range(n_features)
    ]

    return model, importances
