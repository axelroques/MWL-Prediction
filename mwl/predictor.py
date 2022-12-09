
from .ml.classification.ikky_classification_new import run_ikky_classification_new
from .paths import figpath
from .data import Data

from pathlib import Path


class Predictor:

    def __init__(
        self,
        ground_truth='oral_declaration',
        train_prop=0.8,
        n_cross_val_splits=5,
        n_iterations=300,
        nb_classifiers=19,
        tolerance=0.1,
        add_noise=False,
    ) -> None:

        # Store input parameters
        self._ground_truth = ground_truth
        self._train_prop = train_prop
        self._n_cross_val_splits = n_cross_val_splits
        self._n_iterations = n_iterations
        self._nb_classifiers = nb_classifiers
        self._tolerance = tolerance
        self._add_noise = add_noise

    def fit(self, data):
        """
        Fit model with the data.
        Only accepts a Data object as input.
        """

        # Type checking
        if not isinstance(data, Data):
            raise RuntimeError('Input data should be a Data object.')

        # Store raw data
        self._data = data.getFeatures()

        # Select features based on the ground truth
        self._X, self._y = self._dataLoader()

        # Eventual slight feature preprocessing
        self._X = self._featurePreprocessing(self._X)

        return

    def predict(self):
        """
        Predict mental workload.
        """

        auc_mean, auc_std, mean_individual_auc, median_individual_auc = run_ikky_classification_new(
            self._data, self._X, self._y,
            features_labels=self._features_col,
            train_prop=self._train_prop,
            n_cross_val_splits=self._n_cross_val_splits,
            n_iterations=self._n_iterations,
            nb_classifiers=self._nb_classifiers,
            tolerance=self._tolerance,
            add_noise=self._add_noise,
        )

        return

    def _dataLoader(self):
        """
        Return adequate X and y arrays based on ground truth
        requirements.
        """

        # Get indices of features for the specific evaluation type
        self._features_col = [
            col for col in self._data.columns if self._ground_truth in col
        ]

        # Get indices of non-NaN values
        self._valid_indices = (
            ~self._data.loc[:, self._features_col].isna()
        ).product(axis=1).astype(bool)

        # X variable
        X = self._data.copy()
        X = self._data[self._valid_indices][self._features_col]

        if self._ground_truth == 'oral_declaration':
            y = self._data[self._valid_indices]['binary_normalized_oral_tc'].to_numpy()

        elif self._ground_truth in [
            'NASA-TLX', 'mental_demand', 'physical_demand', 'temporal_demand',
            'effort', 'performance', 'frustration', 'mean_NASA_tlx',
            'theoretical_mental_demand', 'theoretical_physical_demand',
            'theoretical_temporal_demand', 'theoretical_effort',
            'mean_theoretical_NASA_tlx'
        ]:
            y = self._data[self._valid_indices][self._ground_truth].to_numpy()

            # Binarize the data
            y[y < 50] = 0
            y[y >= 50] = 1

        else:
            raise RuntimeError('Unrecognized evaluation type.')

        return X, y

    @ staticmethod
    def _featurePreprocessing(X):
        """
        Eventual preprocessing.
        In Alice's code, the sign for the values of some features
        is reversed (thanks to the "reverse_features" list).
        I don't know why.
        Alternatively, one may want to get rid of the sign altogether
        and take the absolute value of some features.

        For now, this function is completely useless.
        """
        return X
