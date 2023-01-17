
from .cv import train_test_split
from .fit import fit

from sklearn.metrics import roc_auc_score
from collections import Counter
import numpy as np

from warnings import filterwarnings


filterwarnings(
    action='ignore', category=DeprecationWarning,
    message='`np.bool` is a deprecated alias'
)
filterwarnings(
    action='ignore', category=DeprecationWarning,
    message='`np.int` is a deprecated alias'
)


def fit_predict(
    data, X, y,
    features_labels,
    train_prop,
    n_cross_val_splits,
    n_iterations,
    n_classifiers,
    tolerance,
    add_noise=False,
    verbose=True
):

    # Transform X DataFrame into numpy array
    X = np.array(X)

    # Tuple indices for custom cross-validation
    indices_cv = np.array(data.loc[:, ['phase', 'pilot']])

    # Features labels
    features_labels = np.array(features_labels)

    # <-- To check
    # Ajout de bruit
    if add_noise:
        B_1 = np.random.uniform(-4, 10, (X.shape[0], 1))
        B_2 = np.random.uniform(-1, 2, (X.shape[0], 1))
        features_labels = np.append(features_labels, "noise_1")
        features_labels = np.append(features_labels, "noise_2")
        X = np.append(X, B_1, axis=1)
        X = np.append(X, B_2, axis=1)

    # Initialization
    # AUC list
    AUCs = []
    # AUC labels
    AUC_labels = {var: [] for var in features_labels}
    # Feature contributions
    feature_contributions = Counter(
        {feature: [] for feature in features_labels})
    # Array to retrieve the predictions (n_samples x n_iterations)
    all_predictions = np.nan * np.ones((X.shape[0], n_iterations))
    # Array to compute individual AUCs (n_pilots x n_iterations)
    individual_AUCs = np.nan * np.ones(
        (np.unique(indices_cv[:, 1]).shape[0], n_iterations)
    )

    # Main loop
    for k in range(n_iterations):

        # Cross-validation scheme
        X_train, X_test, y_train, y_test, \
            pilots_tested, testing_indices = train_test_split(
                X, y, indices_cv, train_prop, k
            )

        # Fit model
        fitted_model, importances = fit(
            X_train, y_train, n_classifiers=n_classifiers,
            tolerance=tolerance, n_cross_val_splits=n_cross_val_splits
        )

        # Predict MWL with the fitted model
        predictions = fitted_model.predict_proba(X_test)[:, 1]
        # Store prediction for this iteration
        all_predictions[testing_indices, k] = predictions

        # Update features contributions dictionary
        feature_contributions.update(
            dict(zip(features_labels, importances))
        )

        # Compute AUC score
        AUC_score = roc_auc_score(
            y_true=y_test, y_score=predictions)
        AUCs.append(AUC_score)

        # Individual AUCs
        for i in range(pilots_tested.shape[0]):
            mask = indices_cv[testing_indices, :][:, 1] == pilots_tested[i]
            indexes_for_given_pilot = np.argwhere(testing_indices[mask])
            if (np.mean(y_test[indexes_for_given_pilot]) != 1.0) \
                    and (np.mean(y_test[indexes_for_given_pilot]) != 0.0):
                auc_score_tmp = roc_auc_score(
                    y_true=y_test[indexes_for_given_pilot], y_score=predictions[indexes_for_given_pilot]
                )
                individual_AUCs[pilots_tested[i]-3, k] = auc_score_tmp

    # Individual nan array
    individual_AUCs_mean = np.ones(
        (np.unique(indices_cv[:, 1]).shape[0], 2)
    )
    individual_AUCs_median = np.ones(
        (np.unique(indices_cv[:, 1]).shape[0], 3)
    )
    for k in range(individual_AUCs_mean.shape[0]):
        individual_AUCs_mean[k, 0] = np.nanmean(
            individual_AUCs[k, :], axis=0
        )
        individual_AUCs_mean[k, 1] = np.nanstd(
            individual_AUCs[k, :], axis=0
        )
        individual_AUCs_median[k, 0] = np.nanmedian(
            individual_AUCs[k, :], axis=0
        )
        individual_AUCs_median[k, 1] = np.nanpercentile(
            individual_AUCs[k, :], q=25, axis=0
        )
        individual_AUCs_median[k, 2] = np.nanpercentile(
            individual_AUCs[k, :], q=75, axis=0
        )
        if verbose:
            print(f'Individual AUC via nan matrix for pilot {k+2}:')
            print(f'\tMean={individual_AUCs_mean[k, 0]:.3f}', end='')
            print(f'± {individual_AUCs_mean[k, 1]:.3f}')
            print(f'\t25%={individual_AUCs_median[k, 1]:.3f}', end='; ')
            print(f'Median={individual_AUCs_median[k, 0]:.3f}', end='; ')
            print(f'75%={individual_AUCs_median[k, 2]:.3f}')

    if verbose:
        # Print AUCs
        print(f'\nAUC on test sets: Mean={np.mean(AUCs):.3f}', end='; ')
        print(f'std={np.std(AUCs):.3f}')

        # Compute features contributions
        contributions = [
            feature_contributions[feature] for feature in features_labels
        ]

        # Sort features by decreasing contribution
        indices = np.argsort(np.abs(contributions))[::-1]
        sorted_features = np.array(features_labels)[indices]

        # Print features contributions
        print(f'Features contribution to the model:')
        for i, feature in enumerate(sorted_features):
            contribution = feature_contributions[feature]
            if np.sum(contribution) == 0:
                continue
            print(f'\t{feature}: {np.sum(contribution)}')

    return AUCs, individual_AUCs_mean, individual_AUCs_median
