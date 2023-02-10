
from .predictor import Predictor
from .paths import figpath
from .ml.fit import fit

from datetime import datetime
from pathlib import Path
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

    def fit(self, plot=False):
        """
        Fit model.
        """

        # Transform X DataFrame into numpy array
        self._X_train = np.array(self._X)
        self._y_train = self._y

        print('X_train =', self._X_train.shape)
        print('y_train =', self._y_train.shape)

        # Fit model
        self._model, self._importances = fit(
            self._X_train, self._y_train, n_classifiers=self._n_classifiers,
            tolerance=self._tolerance, n_cross_val_splits=self._n_cross_val_splits
        )

        # Eventually plot weak classifiers decision boundary
        if plot:
            self._plotWeakClassifiers()

        return

    def predict(self):
        """
        Predict continuous mental load for a given ContinuousData object. 
        """

        print('X_test =', self._X_test.shape)
        self._predictions, self._classifiers_contribution = self._model.predict_proba(
            self._X_test)
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

    def plotClassifierContribution(self):
        """
        Plot of the input classifier's contribution to the MWL.
        Basically superimposes the predic continuous MWL time series
        with the self._classifiers_contribution array.
        """

        import matplotlib.pyplot as plt
        from operator import itemgetter
        from itertools import groupby

        def __merge(contribution):
            """
            Merge successive values into a list of intervals.
            """

            low_MWL = np.where(contribution == 0)[0]
            high_MWL = np.where(contribution == 1)[0]

            low_MWL_intervals = list()
            for _, g in groupby(enumerate(low_MWL), lambda ix: ix[0] - ix[1]):

                interval_local = list(map(itemgetter(1), g))
                if (interval_local[-1] - interval_local[0]) >= 0:
                    ends_local = [interval_local[0], interval_local[-1]]
                    low_MWL_intervals.append(ends_local)

            high_MWL_intervals = list()
            for _, g in groupby(enumerate(high_MWL), lambda ix: ix[0] - ix[1]):

                interval_local = list(map(itemgetter(1), g))
                if (interval_local[-1] - interval_local[0]) >= 0:
                    ends_local = [interval_local[0], interval_local[-1]]
                    high_MWL_intervals.append(ends_local)

            return low_MWL_intervals, high_MWL_intervals

        def __legend_without_duplicate_labels(ax):
            """
            Remove duplicates in legend.
            """

            handles, labels = ax.get_legend_handles_labels()
            unique = [
                (h, l) for i, (h, l) in enumerate(
                    zip(handles, labels)) if l not in labels[:i]
            ]
            ax.legend(
                *zip(*unique),
                loc='center left',
                bbox_to_anchor=(1, 0.5),
                fontsize=15
            )

            return

        # Create continuous MWL DataFrame
        df_MWL = self.getContinuousMWL()

        features_labels = self._X.columns
        for i_feature in range(self._X_test.shape[1]):

            f, ax = plt.subplots(figsize=(12, 4))

            # Get contribution
            contribution = self._classifiers_contribution[:, i_feature]
            label = features_labels[i_feature]

            # Get contribution as intervals
            low_MWL_intervals, high_MWL_intervals = __merge(
                contribution
            )

            # Plot MWL
            ax.plot(
                df_MWL['t'], df_MWL['MWL prediction'],
                c='k', alpha=1, label='MWL prediction'
            )

            # Plot classifier prediction
            for interval in low_MWL_intervals:

                t_start = df_MWL['t'].iloc[interval[0]] - 40
                t_end = df_MWL['t'].iloc[interval[1]] + 10
                ax.axvspan(
                    t_start, t_end,
                    fc='mediumseagreen',
                    ec=None, alpha=0.4,
                    label='Low MWL'
                )

            for interval in high_MWL_intervals:

                t_start = df_MWL['t'].iloc[interval[0]] - 40
                t_end = df_MWL['t'].iloc[interval[1]] + 10
                ax.axvspan(
                    t_start, t_end,
                    fc='crimson',
                    ec=None, alpha=0.4,
                    label='High MWL'
                )

            # Plot params
            ax.set_xlim((0, df_MWL['t'].iloc[-1]))
            ax.set_ylim((-0.05, 1.05))
            ax.set_title(f'{label}', fontsize=20)
            ax.tick_params(labelsize=20)
            __legend_without_duplicate_labels(ax)

            # Save figure
            now = datetime.now()
            date = now.strftime("%Y_%m_%d-%H_%M_%S")
            filename = Path(figpath, f'{date}-{label}')
            f.savefig(filename)

            # Close figure
            plt.close()

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

    def _plotWeakClassifiers(self):
        """
        Plot the decision rules learned by HBagging on the 
        complete dataset, after the fit step.
        """

        import matplotlib.pyplot as plt

        features_labels = self._X.columns
        for i_feature in self._model.features_used:

            f, ax = plt.subplots(figsize=(12, 4))

            # Get weak classifier parameters
            threshold = float(self._model.thresholds[i_feature])

            # Get feature info
            feature = self._X_train[:, i_feature]
            label = features_labels[i_feature]
            if label in self._heuristics:
                feature *= -1
                threshold *= -1

            # Histogram parameters
            step = 0.01 * (np.max(feature) - np.min(feature))
            bins = np.arange(np.min(feature), np.max(feature)+step, step)

            # Plot histogram
            ax.hist(
                feature[self._y_train == 1],
                bins=bins, density=False,
                color='crimson', alpha=0.6,
                label='High MWL',
            )
            ax.hist(
                feature[self._y_train == 0],
                bins=bins, density=False,
                color='mediumseagreen', alpha=0.6,
                label='Low MWL',
            )

            # Plot threshold
            lims_x = ax.get_xlim()
            lims_y = ax.get_ylim()
            range_x = lims_x[1] - lims_x[0]
            range_y = lims_y[1] - lims_y[0]
            ax.plot(
                [threshold, threshold],
                [lims_y[0], lims_y[1]],
                linestyle="--", linewidth=3,
                color='k', alpha=0.8
            )
            ax.text(
                s=f'Threshold: {threshold:.2f}',
                x=threshold-0.05*range_x,
                y=lims_y[1]+0.2*range_y,
                fontsize=20, color='k', alpha=0.8
            )

            # Plot params
            ax.set_ylim(lims_y[0], lims_y[1]+0.6*range_y)
            ax.set_title(f'{label}', fontsize=20)
            ax.tick_params(labelsize=20)
            ax.legend(fontsize=15)

            # Save figure
            now = datetime.now()
            date = now.strftime("%Y_%m_%d-%H_%M_%S")
            filename = Path(figpath, f'{date}-{label}')
            f.savefig(filename)

            # Close figure
            plt.close()

        return
