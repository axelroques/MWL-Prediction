
from .predictor import Predictor
from .ml.fit import fit


import pandas as pd
import numpy as np


class ContinuousPredictor(Predictor):

    def __init__(
        self,
        dataObject,
        contDataObject,
        train_prop=0.8,
        n_cross_val_splits=5,
        n_iterations=300,
        n_classifiers=19,
        tolerance=0.1,
        heuristics=None
    ) -> None:

        super().__init__(
            dataObject,
            ground_truth='oral_declaration',
            train_prop=train_prop,
            n_cross_val_splits=n_cross_val_splits,
            n_iterations=n_iterations,
            n_classifiers=n_classifiers,
            tolerance=tolerance,
            heuristics=heuristics,
            add_noise=False,
        )

        # Prepare input data
        self._prepareContData(contDataObject)

    def fit(self):
        """
        Fit model.
        """

        # Transform X DataFrame into numpy array
        X_train = np.array(self._X)
        y_train = self._y

        print('X_train =', X_train.shape)
        print('y_train =', y_train.shape)

        # Fit model
        self._model, self._importances = fit(
            X_train, y_train, n_classifiers=self._n_classifiers,
            tolerance=self._tolerance, n_cross_val_splits=self._n_cross_val_splits
        )

        return

    def predict(self):
        """
        Predict continuous mental load for a given ContinuousData object. 
        """

        print('X_test =', self._X_test.shape)
        self._predictions = self._model.predict_proba(self._X_test)[:, 1]
        print('predictions =', self._predictions.shape)

        return

    def getContinuousMWL(self):
        """
        Return a dataframe with the continuous MWL prediction given
        the input features.
        """
        return pd.DataFrame(data={
            't': self._all_cont_features['t'],
            'MWL prediction': self._predictions
        })

    def _prepareContData(self, contDataObject):
        """
        Prepare input ContinuousData object.
        """

        # Store raw data
        self._dataObject = contDataObject

        # Get features
        self._all_cont_features = contDataObject.getFeatures()

        # Get X_test
        X_test = self._all_cont_features.copy()
        self._X_test = X_test.iloc[:, 1:].to_numpy()

        return
