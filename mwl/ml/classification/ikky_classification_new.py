
from .fit_classification_algorithm import fit_classification
from ...CustomCrossValidations import CustomCVs as ccv

from sklearn.metrics import roc_auc_score
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


def run_ikky_classification_new(
    data, X, y,
    features_labels,
    train_prop,
    n_cross_val_splits,
    n_iterations,
    nb_classifiers,
    tolerance,
    add_noise=False,
):

    # Transform X DataFrame into numpy array
    X = np.array(X)

    # Tuple indices for custom cross-validation
    indices_cv = np.array(data.loc[:, ['phase', 'pilot']])

    # Features labels
    features_labels = np.array(features_labels)
    print(features_labels)

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
    feature_contributions = {var: [] for var in features_labels}
    # Array to retrieve the predictions (n_features x n_iterations)
    predictions = np.nan * np.ones((X.shape[0], n_iterations))
    # Array to compute individual AUCs (n_pilots x n_iterations)
    individual_aucs = np.nan * np.ones(
        (np.unique(indices_cv[:, 1]).shape[0], n_iterations)
    )

    # Main loop
    for k in range(n_iterations):

        # Cross-validation scheme
        X_train, X_test, y_train, y_test, pilots_tested = train_test_split(
            X, y, indices_cv, train_prop, k
        )

        predictions_subtest = []
        auc_subtest = []
        for variable_type, list_ in dic_variables.items():

            index_list_ = index_lists[variable_type]

            sub_X_train = X_train[:, index_list_]
            sub_X_test = X_test[:, index_list_]

            fitted_model, importances = fit_classification(
                sub_X_train, y_train, algo, params_grid, n_cross_val_splits=n_cross_val_splits)

            selected_models[variable_type].append(fitted_model)

            prediction_subtest = fitted_model.predict_proba(sub_X_test)[:, 1]
            predictions_subtest.append(prediction_subtest)

            auc_score = roc_auc_score(
                y_true=y_test, y_score=prediction_subtest)
            auc_subtest.append(auc_score)

            auc_variable_types[variable_type].append(auc_score)

            for i, var in enumerate(variables_names[index_list_]):
                dic_importances[var].append(importances[i])

        predictions_subtest = np.array(predictions_subtest).T

        # Ajouts de Dimitri
        nan_array[indexes_for_testing, k] = predictions_subtest.flatten()
        # Fin des ajouts

        final_predictions = np.mean(predictions_subtest, axis=1)

        auc_score = roc_auc_score(y_true=y_test, y_score=final_predictions)
        auc_test.append(auc_score)

        # AUC individuelle
        for i in range(pilots_tested.shape[0]):
            mask = indexes_from_df[indexes_for_testing,
                                   :][:, 1] == pilots_tested[i]
            indexes_for_given_pilot = np.argwhere(indexes_for_testing[mask])
            if np.mean(y_test[indexes_for_given_pilot]) != 1.0 and np.mean(y_test[indexes_for_given_pilot]) != 0.0:
                auc_score_tmp = roc_auc_score(
                    y_true=y_test[indexes_for_given_pilot], y_score=final_predictions[indexes_for_given_pilot])
                nan_array_individual[pilots_tested[i]-3, k] = auc_score_tmp

    # Ajouts de Dimitri
    # Individual nan array
    nan_individual_mean = np.ones(
        (np.unique(indexes_from_df[:, 1]).shape[0], 2)
    )
    nan_individual_median = np.ones(
        (np.unique(indexes_from_df[:, 1]).shape[0], 3)
    )
    for k in range(nan_individual_mean.shape[0]):
        nan_individual_mean[k, 0] = np.nanmean(
            nan_array_individual[k, :], axis=0
        )
        nan_individual_mean[k, 1] = np.nanstd(
            nan_array_individual[k, :], axis=0
        )
        nan_individual_median[k, 0] = np.nanmedian(
            nan_array_individual[k, :], axis=0
        )
        nan_individual_median[k, 1] = np.nanpercentile(
            nan_array_individual[k, :], q=25, axis=0
        )
        nan_individual_median[k, 2] = np.nanpercentile(
            nan_array_individual[k, :], q=75, axis=0
        )
        print("Individual AUC via nan matrix for pilot", k+3, "is:", "%.3f" %
              nan_individual_mean[k, 0], "±", "%.3f" % nan_individual_mean[k, 1])
        print("Median is:", "%.3f" % nan_individual_median[k, 0], "25%:", "%.3f" %
              nan_individual_median[k, 1], "75%:", "%.3f" % nan_individual_median[k, 2])

    """ print auc """

    for variable_type, names_ in dic_variables.items():

        aucs = auc_variable_types[variable_type]
        print("\n", variable_type, len(index_lists[variable_type]), "%.3f" % np.mean(
            aucs), "écart-type", "%.3f" % np.std(aucs))

    print("\nAUC moyenne sur les ensembles de test :", "%.3f" %
          np.mean(auc_test), "écart-type", "%.3f" % np.std(auc_test))

    """ print features importance """

    for variable_type, list_ in dic_variables.items():
        importances = [np.mean(dic_importances[var]) for var in list_]

        print("--")
        print(variable_type)
        print("--")

        indices = np.argsort(np.abs(importances))[::-1]
        sorted_variables = np.array(list_)[indices]

        print("Poids des variables dans le modèle :\n", variable_type)

        for i, var in enumerate(sorted_variables):

            variable_importance = dic_importances[var]
            if np.sum(variable_importance) == 0:
                continue
            print(var, np.sum(variable_importance))

    """ Gobal model for interpretation only """

    for variable_type, list_ in dic_variables.items():

        index_list_ = []
        for var in list_:
            ind = list(variables_names).index(var)
            index_list_.append(ind)

        index_list_ = np.array(index_list_)

        sub_X = X[:, index_list_]

        total_model, variables_used = fit_classification(
            sub_X, y, algo, params_grid, n_cross_val_splits)

    rename_dic = {name: name.replace("feature_tc_fixed_windows_", "").replace(
        "_", " ") for name in variables_names}
    rename_dic = {k: v[0].upper()+v[1:] for k, v in rename_dic.items()}

    total_model.draw(sub_X, y, title="Hbagging_"+title+"_"+variable_type, features_names=variables_names[index_list_], fig_path=fig_path, reversed_variables_names=None,
                     name_classes=name_classes, rename_dic=rename_dic)

    return np.mean(auc_test), np.std(auc_test), nan_individual_mean, nan_individual_median


def train_test_split(X, y, indices_cv, train_prop, seed):
    """
    Return X_train, X_test, y_train and y_test arrays.

    Assures that the X_test and y_test contain different 
    labels and that the test set contains at least 3 pilots.
    """

    # Initialize arrays at 0
    X_train = np.zeros_like(X)
    X_test = np.zeros_like(X)
    y_train = np.zeros_like(y)
    y_test = np.zeros_like(y)

    seed_increment = 0
    while (np.mean(y_test) == 1) \
            or (np.mean(y_test) == 0) \
            or (np.unique(indices_cv[testing_indices, 1]).size < 3):

        # Cross-validation repartition
        training_indices, testing_indices = ccv.TimeCV(
            indices_cv, 1-train_prop, weighting=False,
            seed_k=seed+seed_increment
        )
        X_train = X[training_indices, :]
        X_test = X[testing_indices, :]
        y_train = y[training_indices]
        y_test = y[testing_indices]

        # Increment
        seed_increment += 1

    pilots_tested = np.unique(indices_cv[testing_indices, 1])

    return X_train, X_test, y_train, y_test, pilots_tested
