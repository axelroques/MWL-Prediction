
from .fit_classification_algorithm import fit_classification
from ...CustomCrossValidations import CustomCVs as ccv


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import numpy as np

from warnings import filterwarnings


filterwarnings(action='ignore', category=DeprecationWarning,
               message='`np.bool` is a deprecated alias')
filterwarnings(action='ignore', category=DeprecationWarning,
               message='`np.int` is a deprecated alias')


def run_ikky_classification_new(X, y, algo, variables_names, dic_variables,
                                params_grid, name_classes, train_prop=0.8,
                                num_iterations=10, n_cross_val_splits=5,
                                stratify_=True, fig_path=None,
                                rename_dic={},
                                title="", add_noise=False, cv_scheme="iB",
                                actual_ground_truth="oral_declaration"):

    print(actual_ground_truth)
    X_copy = X.copy()
    # Addition de Dimitri pour la custom cross-validation
    indexes_from_df = np.array(X.index.values.tolist())
    indexes_from_df = ccv.add_indexes(indexes_from_df)

    # Set the features in their lists
    features_helico = ("std_alti", "std_yaw", "time_spent_communication")
    features_commands = ("number_flights_commands")
    features_hr = ("mean_heart_rate", "mean_hrv")
    features_br = ("mean_breath_rate", "std_breath_rate")
    features_blinks = ("blink_rate", "blink_rate (-)", "mean_blinks_duration")
    features_oculo = ("mean_fixations_duration", "mean_saccades_duration",
                      "mean_saccades_amplitude", "percent_time_fixations_AOI", "gaze_ellipse")
    features_aoi = ("Front panel", "Outside view", "WP", "Block, Speed, Horiz, Vario, Rot",
                    "PFD/ND", "CAD, VEMD", "ADF, XPDR, RCU", "GNS", "APMS", "ICP", "ACU", "Over head panel", )

    X = np.array(X)

    # Rajouter du bruit : le créer
    if add_noise:
        B_1 = np.random.uniform(-4, 10, (X.shape[0], 1))
        B_2 = np.random.uniform(-1, 2, (X.shape[0], 1))
    # Fin de rajout du bruit

    # Matrices des indices
    start_indexes = np.empty((np.unique(indexes_from_df[:, 1]).shape[0]))
    end_indexes = np.empty((np.unique(indexes_from_df[:, 1]).shape[0]))
    start_indexes[0] = 0
    for k in range(1, indexes_from_df.shape[0]):
        if indexes_from_df[k, 1] != indexes_from_df[k-1, 1]:
            start_indexes[np.int32(indexes_from_df[k, 1])-3] = k
            end_indexes[np.int32(indexes_from_df[k-1, 1])-3] = k-1
    end_indexes[-1] = indexes_from_df.shape[0]-1
    # Array to retrieve the predictions
    nan_array = np.nan * np.ones(shape=(X.shape[0], num_iterations))

    # Array to compute individual AUCs
    nan_array_individual = np.nan * \
        np.ones(
            shape=(np.unique(indexes_from_df[:, 1]).shape[0], num_iterations))
    # Fin de l'addition

    variables_names = np.array(variables_names)

    # Ajout de bruit
    if add_noise:
        variables_names = np.append(variables_names, "noise_1")
        variables_names = np.append(variables_names, "noise_2")

        dic_variables["features"].append("noise_1")
        dic_variables["features"].append("noise_2")
    # Fin ajout de bruit

    auc_test = []

    selected_models = {variable_type: [] for variable_type in dic_variables}
    auc_variable_types = {variable_type: [] for variable_type in dic_variables}

    strat = y if stratify_ else None

    index_lists = {}

    for variable_type, list_ in dic_variables.items():
        index_list_ = []
        for var in list_:
            ind = list(variables_names).index(var)
            index_list_.append(ind)
        index_lists[variable_type] = np.array(index_list_)

    dic_importances = {var: [] for var in variables_names}

    for k in range(num_iterations):

        # Retrieve the data with the appropriate columns shuffled
        X = X_copy.copy()
        X = np.array(X)

        # Rajouter du bruit : le mettre dans la matrice de données
        if add_noise:
            X = np.append(X, B_1, axis=1)
            X = np.append(X, B_2, axis=1)
        # Fin rajout bruit

        predictions_subtest = []
        auc_subtest = []

        # Tentative de changer de schéma de cross-validation par Dimitri
        if cv_scheme == "iB":
            indexes_for_training, indexes_for_testing = ccv.TimeCV(
                indexes_from_df, 1-train_prop, weighting=False, seed_k=k)
            X_train = np.take(X, indexes_for_training, axis=0)
            X_test = np.take(X, indexes_for_testing, axis=0)
            y_train = np.take(y, indexes_for_training, axis=0)
            y_test = np.take(y, indexes_for_testing, axis=0)

            p = 1
            while np.mean(y_test) == 1 or np.mean(y_test) == 0 or np.unique(indexes_from_df[indexes_for_testing, 1]).size < 3:
                indexes_for_training, indexes_for_testing = ccv.TimeCV(
                    indexes_from_df, 1-train_prop, weighting=False, seed_k=k+np.random.randint(p))
                X_train = np.take(X, indexes_for_training, axis=0)
                X_test = np.take(X, indexes_for_testing, axis=0)
                y_train = np.take(y, indexes_for_training, axis=0)
                y_test = np.take(y, indexes_for_testing, axis=0)
                p = p + 1

        pilots_tested = np.int32(
            np.unique(indexes_from_df[indexes_for_testing, 1]))

        for variable_type, list_ in dic_variables.items():

            index_list_ = index_lists[variable_type]

            sub_X_train = X_train[:, index_list_]
            sub_X_test = X_test[:, index_list_]

            auc_grid = []

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
        (np.unique(indexes_from_df[:, 1]).shape[0], 2))
    nan_individual_median = np.ones(
        (np.unique(indexes_from_df[:, 1]).shape[0], 3))
    for k in range(nan_individual_mean.shape[0]):
        nan_individual_mean[k, 0] = np.nanmean(
            nan_array_individual[k, :], axis=0)
        nan_individual_mean[k, 1] = np.nanstd(
            nan_array_individual[k, :], axis=0)
        nan_individual_median[k, 0] = np.nanmedian(
            nan_array_individual[k, :], axis=0)
        nan_individual_median[k, 1] = np.nanpercentile(
            nan_array_individual[k, :], q=25, axis=0)
        nan_individual_median[k, 2] = np.nanpercentile(
            nan_array_individual[k, :], q=75, axis=0)
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

        if algo == "logistic" or algo == "forest":
            importances = [np.mean(dic_importances[var]) for var in list_]

        elif algo == "hbagging":
            importances = [np.mean(dic_importances[var]) for var in list_]

        print("--")
        print(variable_type)
        print("--")

        indices = np.argsort(np.abs(importances))[::-1]
        sorted_variables = np.array(list_)[indices]

        print("Poids des variables dans le modèle :\n", variable_type)

        for i, var in enumerate(sorted_variables):

            variable_importance = dic_importances[var]
            if algo == "hbagging" and np.sum(variable_importance) == 0:
                continue

            if algo == "hbagging":
                print(var, np.sum(variable_importance))

            else:
                print(var, np.mean(variable_importance),
                      np.std(variable_importance))

    """ Gobal model for interpretation only, only for hbagging """

    if algo == "hbagging":

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
