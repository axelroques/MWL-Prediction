
from .ml.classification.ikky_classification import run_ikky_classification
from .load_data import Ikky_data
from .paths import figpath

from pathlib import Path
import numpy as np


# What is that?
reverse_features = [
    "blink_rate (-)", "mean_blinks_duration", "mean_saccades_duration", "mean_saccades_amplitude",
    "gaze_ellipse", "mean_hrv", "std_breath_rate"
]


class Predictor:

    def __init__(
        self,
        train_prop=0.8,
        num_iterations=300,
        n_cross_val_splits=5,
        stratify_=True,
        tolerance=0.1,
        ground_truth="tc",
        add_noise=False,
        cv_scheme="iB",
        actual_ground_truth="oral_declaration",
        exclude_pilots=[2],
        signal_selection={}
    ) -> None:

        normalized_labels = True

        db = Ikky_data(reset=False, exclude_pilots=exclude_pilots)
        table = db.get_tc_table()

        # and "ADF" not in col]# and "blinks_duration" not in col] #+ ["mean_theoretical_NASA_tlx"]  #+["flight_hours]
        features_cols = [
            col for col in table.columns if "feature" in col and "NASA_tlx" not in col
        ]

        valid_indexes = ((~table[features_cols].isna()
                          ).product(axis=1)).astype(bool)

        X = table[valid_indexes][features_cols]

        if not normalized_labels:
            y = table[valid_indexes]["oral_tc"].values
            y[y < 35] = 0
            y[y >= 35] = 1

            y = y.astype(int)

        else:
            y = table[valid_indexes]["binary_normalized_oral_tc"].values

        X_HBagging = X.copy()
        X_HBagging.iloc[:, np.array([list(X.columns).index("feature_tc_fixed_windows_"+feature)
                                     for feature in reverse_features if "feature_tc_fixed_windows_"+feature in list(X.columns)])] *= -1

        # print('X_HBagging =\n', X_HBagging)

        # Store parameters
        self.X_HBagging = X_HBagging
        self.y = y
        self.train_prop = train_prop
        self.num_iterations = num_iterations
        self.n_cross_val_splits = n_cross_val_splits
        self.stratify_ = stratify_
        self.params_grid = {
            "nb_classifiers": np.arange(2, 11),
            "tolerance": [tolerance]
        }
        self.ground_truth = ground_truth
        self.actual_ground_truth = "oral_declaration"
        self.dic_variables = {"features": features_cols}
        self.rename_dic = {}
        self.name_classes = ["Low cognitive load", "High cognitive load"]
        self.features_cols = features_cols
        self.add_noise = add_noise
        self.cv_scheme = cv_scheme

        ##### Test ######
        self._table = table
        self._test = db.dic_tables

    def predict(self):

        auc_mean, auc_std, mean_individual_auc, median_individual_auc = run_ikky_classification(
            self.X_HBagging, self.y, dic_variables=self.dic_variables, algo="hbagging", params_grid=self.params_grid, train_prop=self.train_prop,
            name_classes=self.name_classes, num_iterations=self.num_iterations, n_cross_val_splits=self.n_cross_val_splits,
            variables_names=self.features_cols, rename_dic=self.rename_dic, stratify_=self.stratify_, plot=True,
            fig_path=Path(figpath, "model"), title="", reversed_variables_names=["features_tc_fixed_windows_"+feature for feature in reverse_features]+["flight_hours"],
            remove_helico=False, remove_commands=False,
            remove_cardio=False, remove_respi=False, remove_blinks=False,
            remove_oculo=False, remove_aoi=False,
            ground_truth=self.ground_truth, add_noise=self.add_noise, cv_scheme=self.cv_scheme,
            actual_ground_truth=self.actual_ground_truth)

        return
