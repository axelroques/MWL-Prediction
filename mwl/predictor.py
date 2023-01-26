
from .ml.fit_predict import fit_predict
from .data import Data

import pandas as pd
import numpy as np


class Predictor:

    def __init__(
        self,
        dataObject,
        ground_truth='oral_declaration',
        train_prop=0.8,
        n_cross_val_splits=5,
        n_iterations=300,
        n_classifiers=19,
        tolerance=0.1,
        heuristics=None,
        add_noise=False,
    ) -> None:

        # Store input parameters
        self._ground_truth = ground_truth
        self._train_prop = train_prop
        self._n_cross_val_splits = n_cross_val_splits
        self._n_iterations = n_iterations
        self._n_classifiers = n_classifiers
        self._tolerance = tolerance
        self._add_noise = add_noise

        # Option to add heuristics to the model
        # Features in the heurstics list correspond to features
        # whose increase lead to a decrease in MWL
        self._heuristics = heuristics

        # Prepare input data
        self._prepare(dataObject)

    def fit_predict(self):
        """
        Predict mental workload.
        """

        AUCs, individual_AUCs_mean, individual_AUCs_median, feature_contributions = fit_predict(
            self._data, self._X, self._y,
            features_labels=self._features_col,
            train_prop=self._train_prop,
            n_cross_val_splits=self._n_cross_val_splits,
            n_iterations=self._n_iterations,
            n_classifiers=self._n_classifiers,
            tolerance=self._tolerance,
            add_noise=self._add_noise
        )

        return AUCs, individual_AUCs_mean, individual_AUCs_median, feature_contributions

    def compareFeaturesSet(self, remove_feature_groups=[]):
        """
        Generates a data table that synthetizes the prediction results
        of multiple models trained with the different input feature groups.

        remove_feature_groups should be a list of lists with different
        filenames, e.g.:
        remove_feature_groups = [
            ['am'], ['am', 'aoi']
        ]
        With the previous example, the algorithm will evaluate 4 models.
        The models will be trained respectively with:
            1) All features excepted for those in 'am'
            2) Only the features in 'am'
            3) All features excepted for those in 'am' and 'aoi'
            4) Only the features in 'am' and 'aoi'
        """

        if remove_feature_groups:

            def __updateData(
                main, median, first_quartile, third_quartile, feature_contribution,
                group, AUCs, individual_AUCs_median, contributions
            ):
                """
                Simple function to update the data dictionary.
                """

                main['Features removed'].append(', '.join(group))
                main['Mean AUC'].append(np.mean(AUCs))
                main['Std AUC'].append(np.std(AUCs))

                for feature, contribution in contributions.items():
                    feature_contribution[feature].append(contribution)

                for i, (key_m, key_fq, key_tq) in enumerate(zip(
                    median, first_quartile, third_quartile
                )):
                    median[key_m].append(individual_AUCs_median[i, 0])
                    first_quartile[key_fq].append(individual_AUCs_median[i, 1])
                    third_quartile[key_tq].append(individual_AUCs_median[i, 2])

                return

            # Initialize output data
            main = {
                'Features removed': [],
                'Mean AUC': [],
                'Std AUC': []
            }
            median = {
                f'Median AUC pilot {p}': []
                for p in [2, 3, 4, 5, 6, 7, 8, 9]
                if p not in self._dataObject.exclude_pilots
            }
            first_quartile = {
                f'First quartile AUC pilot {p}': []
                for p in [2, 3, 4, 5, 6, 7, 8, 9]
                if p not in self._dataObject.exclude_pilots
            }
            third_quartile = {
                f'Third quartile AUC pilot {p}': []
                for p in [2, 3, 4, 5, 6, 7, 8, 9]
                if p not in self._dataObject.exclude_pilots
            }
            feature_contribution = {
                feature: [] for feature in self._features_col
            }

            # Iterate over groups of features to remove
            for group in remove_feature_groups:
                print(f'Removing features in {group}...')

                # Remove features in group
                self._dataObject.selectFeatures(
                    exclude_files=group,
                    exclude_pilots=self._dataObject.exclude_pilots
                )

                self._prepare(self._dataObject)

                # Predict MWL with subset of features
                AUCs, _, individual_AUCs_median, contributions = fit_predict(
                    self._features, self._X, self._y,
                    features_labels=self._features_col,
                    train_prop=self._train_prop,
                    n_cross_val_splits=self._n_cross_val_splits,
                    n_iterations=self._n_iterations,
                    n_classifiers=self._n_classifiers,
                    tolerance=self._tolerance,
                    add_noise=self._add_noise,
                    verbose=False
                )

                # Store results
                __updateData(
                    main, median, first_quartile, third_quartile, feature_contribution,
                    group, AUCs, individual_AUCs_median, contributions
                )

                # Remove features outside of group
                inverse_group = [
                    f for f in list(self._dataObject._feature_dictionary.keys())
                    if f not in group
                ]
                self._dataObject.selectFeatures(
                    exclude_files=inverse_group,
                    exclude_pilots=self._dataObject.exclude_pilots
                )

                self._prepare(self._dataObject)

                # Predict MWL with subset of features
                AUCs, _, individual_AUCs_median, contributions = fit_predict(
                    self._features, self._X, self._y,
                    features_labels=self._features_col,
                    train_prop=self._train_prop,
                    n_cross_val_splits=self._n_cross_val_splits,
                    n_iterations=self._n_iterations,
                    n_classifiers=self._n_classifiers,
                    tolerance=self._tolerance,
                    add_noise=self._add_noise,
                    verbose=False
                )

                # Store results
                __updateData(
                    main, median, first_quartile, third_quartile, feature_contribution,
                    inverse_group, AUCs, individual_AUCs_median, contributions
                )

            # Combine all data sources together
            features = main.copy()
            main.update(**median, **first_quartile, **third_quartile)

            # Create another dictionary with the feature contribution
            features.update(**feature_contribution)

        else:
            raise RuntimeError('remove_feature_groups is empty.')

        return pd.DataFrame(data=main), pd.DataFrame(data=features)

    def _prepare(self, dataObject):
        """
        Prepare data. Mainly assures a correct formatting
        to later feed into the model.
        Only accepts a Data object as input.
        """

        # Type checking
        if not isinstance(dataObject, Data):
            raise RuntimeError('Input data should be a Data object.')

        # Store raw data
        self._dataObject = dataObject

        # Get features
        self._features = dataObject.getFeatures()

        # Select features based on the ground truth
        self._data, self._X, self._y = self._dataLoader()

        # Eventual slight feature preprocessing
        self._X = self._featurePreprocessing(self._X)

        return

    def _dataLoader(self):
        """
        Return adequate X and y arrays based on ground truth
        requirements.

        Theoretical ground truth values: [
            'mental_demand', 'physical_demand', 'temporal_demand',
            'effort', 'performance', 'frustration', 'mean_NASA_tlx',
            'theoretical_mental_demand', 'theoretical_physical_demand',
            'theoretical_temporal_demand', 'theoretical_effort',
            'mean_theoretical_NASA_tlx'
        ]
        """

        # Get indices of features for the specific evaluation type
        NASA_TLX_related_ground_truths = [
            'mean_NASA_tlx', 'effort', 'performance', 'frustration',
            'mental_demand', 'physical_demand', 'temporal_demand'
        ]
        theoretical_NASA_TLX_related_ground_truths = [
            'mean_theoretical_NASA_tlx',
            'theoretical_effort', 'theoretical_mental_demand',
            'theoretical_physical_demand', 'theoretical_temporal_demand'
        ]

        # - NASA-TLX related ground truths - #
        #*- Ugly fix because of redondancy in column names -*#
        if self._ground_truth in NASA_TLX_related_ground_truths:
            self._features_col = [
                col for col in self._features.columns
                if ('NASA-TLX' in col) and ('theoretical' not in col)
            ]

        # - Theoretucal NASA-TLX related ground truths - #
        elif self._ground_truth in theoretical_NASA_TLX_related_ground_truths:
            self._features_col = [
                col for col in self._features.columns if 'theoretical' in col
            ]

        # - Self evaluation - #
        elif self._ground_truth == 'oral_declaration':
            self._features_col = [
                col for col in self._features.columns if self._ground_truth in col
            ]
        else:
            raise RuntimeError('Unrecognized ground truth.')

        # Get indices of non-NaN values
        self._valid_indices = (
            ~self._features.loc[:, self._features_col].isna()
        ).product(axis=1).astype(bool)

        # Sub data
        data = self._features.copy()
        data = self._features[self._valid_indices]

        # X variable
        # for col in self._features.columns:
        #     print(col)
        X = self._features.copy()
        X = self._features[self._valid_indices][self._features_col]

        # If heuristics are given, change the sign of theses features
        if self._heuristics:
            X.iloc[:, [
                X.columns.get_loc(f'feature_{self._ground_truth}_{feature}')
                for feature in self._heuristics
                if f'feature_{self._ground_truth}_{feature}' in X
            ]] *= -1

        if self._ground_truth == 'oral_declaration':
            y = self._features[self._valid_indices]['binary_normalized_oral_tc'].to_numpy(
            )

        else:
            y = self._features[self._valid_indices][self._ground_truth].to_numpy()

            # Binarize the data
            y[y < 50] = 0
            y[y >= 50] = 1

        return data, X, y

    @ staticmethod
    def _featurePreprocessing(X):
        """
        Eventual preprocessing.

        For now, this function is completely useless.
        """
        return X
