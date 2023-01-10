
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from .paths import savepath, datapath, evalpath
from .processing import processAOI, AOI_groups
from .processing import processASL, APmodes
from .processing import processEyeMovements
from .processing import processECG
from .processing import processRC

import logging
logging.basicConfig(
    filename=Path(savepath, 'log.log'),
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO
)


class Data:

    def __init__(
        self,
        compute_features=False
    ) -> None:

        # Must coincide with what is done during the feature computation step
        self._feature_dictionary = {
            'am': [
                'std_helico_altitude', 'std_helico_yaw',
                'mean_helico_pitch', 'std_helico_pitch',
                'mean_helico_roll', 'std_helico_roll'
            ],
            'aoi': ['gaze_ellipse_area'] + [
                f'proportion_time_spent_{group}'
                for _, group in AOI_groups.items()
            ],
            'asl': [
                f'time_spent_{top_key}_{item}'
                for top_key, mode in APmodes.items()
                for _, item in mode.items()
            ],
            'br': [
                'mean_breathing_rate', 'std_breathing_rate'
            ],
            'ecg': [
                'mean_heart_rate', 'std_heart_rate',
                'mean_ibi', 'std_ibi'
            ],
            'fc': [
                'mean_cmd_coll', 'std_cmd_coll', 'mean_cmd_yaw', 'std_cmd_yaw',
                'mean_cmd_pitch', 'std_cmd_pitch', 'mean_cmd_roll', 'std_cmd_roll',
                'mean_force_coll', 'std_force_coll', 'mean_force_yaw', 'std_force_yaw',
                'mean_force_pitch', 'std_force_pitch', 'mean_force_roll', 'std_force_roll'
            ],
            'rc': [
                'proportion_time_spent_coms'
            ],
            'sed': [
                'mean_fixation_duration', 'mean_saccade_duration', 'mean_saccade_amplitude'
            ]
        }

        # All features together in a list
        self.features_list = [
            col for filename in self._feature_dictionary
            for col in self._feature_dictionary[filename]
        ]

        # Features to be normalized
        self.features_to_normalize = ['aoi', 'br', 'ecg', 'sed']

        # Compute all features and save the data
        if compute_features:
            print('Computing features...', end=' ')
            self._all_features = self._computeFeatures()
            self._saveFeatures()

        # Otherwise just load the features if they were already computed
        else:
            print('Loading features...', end=' ')
            with open(Path(savepath, 'features.pkl'), "rb") as file:
                self._all_features = pickle.load(file)

        # By default, without any selection, the feature set is complete
        self._features = self._all_features.copy()
        self.exclude_files = []
        self.exclude_pilots = []

        print('Done!')

    def selectFeatures(self, exclude_files=[], exclude_pilots=[]):
        """
        Return a subset of self._all_features based on the data in 
        exclude_files and exclude_pilots.

        Features cannot simply be excluded; instead, their content 
        is permuted randomly.
        """

        # Store selection parameters
        self.exclude_files = exclude_files
        self.exclude_pilots = exclude_pilots

        # Get columns to remove from self._all_features
        features_to_exclude = [
            col for filename in exclude_files
            for col in self._feature_dictionary[filename]
        ]
        # Must add the correct evaluation type
        features_to_exclude_complete = [
            f'feature_{eval_type}_{feature}'
            for eval_type in ['NASA-TLX', 'theoretical_NASA-TLX', 'oral_declaration']
            for feature in features_to_exclude
        ]

        # Random permutations for all columns in features_to_exclude_complete
        np.random.seed(19)
        permutations = self._all_features.copy()
        permutations = permutations.loc[
            ~permutations['pilot'].isin(exclude_pilots),
            permutations.columns.isin(features_to_exclude_complete)
        ]
        permutations = permutations.reindex(
            np.random.permutation(permutations.index)
        ).reset_index(drop=True)

        # Select remaining pilots
        features_subset = self._all_features.copy()
        features_subset = features_subset.loc[
            ~features_subset['pilot'].isin(exclude_pilots),
            ~features_subset.columns.isin(features_to_exclude_complete)
        ].reset_index(drop=True)

        # Combine the two DataFrames
        self._features = pd.concat(
            [features_subset, permutations], axis=1
        )

        print(
            'The following features were removed:',
            ', '.join(f for f in features_to_exclude_complete)
        )

        return

    def getFeatures(self):
        """
        Simply return features.
        """
        return self._features

    def _computeFeatures(self):
        """
        Compute all features.
        """

        # Feature processing launcher
        process = {
            'am': self._amFeatures,
            'aoi': self._aoiFeatures,
            'asl': self._aslFeatures,
            'br': self._brFeatures,
            'ecg': self._ecgFeatures,
            'fc': self._fcFeatures,
            'rc': self._rcFeatures,
            'sed': self._sedFeatures,
        }

        # DataFrame with info on where features should be computed
        df_eval = pd.read_csv(evalpath)

        # Initialize features DataFrame
        df_features = df_eval.copy()

        # Initialize data structure that will hold the features
        data = {
            feature: np.full(len(df_eval), np.nan) for feature in self.features_list
        }

        # Iterate over the different evaluation types
        for evaluation_type in ['NASA-TLX', 'theoretical_NASA-TLX', 'oral_declaration']:
            logging.info(f'evaluation type = {evaluation_type}')

            for pilot in [2, 3, 4, 5, 6, 7, 8, 9]:
                logging.info(f'\tpilot = {pilot}')

                # Sub-DataFrame with the mwl evaluation of the pilot
                sub_eval = df_eval.loc[df_eval['pilot'] == pilot]

                # Iterate over the evaluation windows
                windows_indices, windows = self._getWindows(
                    sub_eval, evaluation_type)
                for i_window, window in zip(windows_indices, windows):
                    logging.info(
                        f'\t\twindow = {window}, i_window = {i_window}')

                    for filename in self._feature_dictionary:
                        logging.info(f'\t\t\tfilename = {filename}')

                        # Load raw data
                        df_scenario_1 = None
                        if filename in self.features_to_normalize:
                            df_scenario_1 = self._load(1, pilot, filename)
                        df_scenario_2 = self._load(2, pilot, filename)

                        # Cut DataFrame around the evaluation window
                        cut = self._cut(df_scenario_2, window)

                        # Compute features on the resulting DataFrame
                        features = process[filename](
                            df_scenario_2=cut,
                            df_scenario_1=df_scenario_1,
                            window=window
                        )

                        # Update data dictionary
                        data = self._addFeatures(data, features, i_window)

            # Update features DataFrame
            df_features = self._updateFeaturesDataFrame(
                df_features, data, evaluation_type
            )

        return df_features

    def _saveFeatures(self):
        """
        Save the computed features with pickle.
        """

        with open(Path(savepath, 'features.pkl'), "wb") as file:
            pickle.dump(
                self._all_features, file,
                protocol=pickle.HIGHEST_PROTOCOL
            )

        return

    @staticmethod
    def _load(scenario, pilot, filename):
        """
        Load filename.txt file.

        Scenario = {1, 2}
        Pilot = {2, 3, 4, 5, 6, 7, 8, 9}
        """

        input2file = {
            '12': '2017-05-10_12;47;27_eyeState2',
            '13': '2017-06-14_12;58;22_eyeState2',
            '14': '2017-06-28_12;55;04_eyeState2',
            '15': '2017-09-27_13;18;28_eyeState2',
            '16': '2017-10-11_12;23;22_eyeState2',
            '17': '2017-11-15_13;43;33_eyeState2',
            '18': '2018-01-23_14;30;48',  # Potential issue here
            '19': '2018-01-30_13;52;12_eyeState2',
            '22': '2017-05-11_11;41;18_eyeState2',
            '23': '2017-06-15_12;12;58_eyeState2',
            '24': '2017-06-29_12;07;03_eyeState2',
            '25': '2017-09-28_11;43;46_eyeState2',
            '26': '2017-10-12_12;03;15_eyeState2',
            '27': '2017-11-16_12;47;54_eyeState2',
            '28': '2018-01-24_13;04;14_eyeState2',
            '29': '2018-01-31_12;58;16_eyeState2'
        }

        path = Path(
            datapath, f"{input2file[f'{scenario}{pilot}']}/{filename}.txt"
        )

        return pd.read_csv(path, sep=';', header=0)

    @staticmethod
    def _cut(df, window):
        """
        Return a subsection of a DataFrame constrained to the window. 
        """

        return df.loc[
            (df['reltime'] >= window[0]) & (df['reltime'] < window[1])
        ]

    @staticmethod
    def _getWindows(df_eval, evaluation_type):
        """
        Get all windows based on the evaluation type:
            - 'oral_declaration': (df.iloc[i]['time_tc']-40, df.iloc[i]['time_tc']+10)
            - 'NASA-TLX': (df.iloc[i]['time_start_NASA_tlx'], 
            df.iloc[i]['time_start_NASA_tlx']+df.iloc[i]['time_NASA_tlx_window'])
            - 'theoretical_NASA-TLX': (df.iloc[i]['time_start_tc_david'], 
            df.iloc[i]['time_end_tc_david'])
        """

        if evaluation_type == 'oral_declaration':
            # Get non-NaN values and their indices
            non_nans = df_eval[~df_eval['time_tc'].isnull()]
            windows_indices = list(non_nans.index.values)
            # Retrieve windows
            windows = [
                (
                    non_nans.iloc[i]['time_tc']-40,
                    non_nans.iloc[i]['time_tc']+10
                )
                for i in range(len(non_nans))
            ]

        elif evaluation_type == 'NASA-TLX':
            # Get non-NaN values and their indices
            non_nans = df_eval[~df_eval['time_start_NASA_tlx'].isnull()]
            windows_indices = list(non_nans.index.values)
            # Retrieve windows
            windows = [
                (
                    non_nans.iloc[i]['time_start_NASA_tlx'],
                    non_nans.iloc[i]['time_start_NASA_tlx'] +
                    non_nans.iloc[i]['time_NASA_tlx_window']
                )
                for i in range(len(non_nans))
            ]

        elif evaluation_type == 'theoretical_NASA-TLX':
            # Get non-NaN values and their indices
            non_nans = df_eval[~df_eval['time_start_tc_david'].isnull()]
            windows_indices = list(non_nans.index.values)
            # Retrieve windows
            windows = [
                (
                    non_nans.iloc[i]['time_start_tc_david'],
                    non_nans.iloc[i]['time_end_tc_david']
                )
                for i in range(len(non_nans))
            ]

        else:
            raise RuntimeError('Unknown evaluation type.')

        return windows_indices, windows

    @staticmethod
    def _amFeatures(**kwargs):
        """
        Compute features for the aircraft motion.
        """

        df = kwargs['df_scenario_2']

        return {
            'std_helico_altitude': np.std(df['baro_alti']),
            'std_helico_yaw': np.std(df['yaw']),
            'mean_helico_pitch': np.mean(df['pitch']),
            'std_helico_pitch': np.std(df['pitch']),
            'mean_helico_roll': np.mean(df['roll']),
            'std_helico_roll': np.std(df['roll'])
        }

    @staticmethod
    def _aoiFeatures(**kwargs):
        """
        Compute features for the AOI data.

        The gaze_ellipse_area is normalized by its value 
        over the whole first scenario.
        """

        df = kwargs['df_scenario_2']
        df_norm = kwargs['df_scenario_1']

        gaze_ellipse_area, time_spent = processAOI(df)
        gaze_ellipse_area_norm, _ = processAOI(df_norm)

        # time_spent will now contain all necessary features
        time_spent.update({
            'gaze_ellipse_area': gaze_ellipse_area - gaze_ellipse_area_norm
        })

        return time_spent

    @staticmethod
    def _aslFeatures(**kwargs):
        """
        Compute features for the automatic pilot.
        """

        df = kwargs['df_scenario_2']

        time_spent = processASL(df)

        return time_spent

    @staticmethod
    def _brFeatures(**kwargs):
        """
        Compute features for the heart rate.
        """

        df = kwargs['df_scenario_2']
        df_norm = kwargs['df_scenario_1']

        br = np.array(df['breath_rate'])
        br_norm = np.array(df_norm['breath_rate'])

        return {
            'mean_breathing_rate': np.mean(br) - np.mean(br_norm),
            'std_breathing_rate': np.std(br) - np.std(br_norm)
        }

    @staticmethod
    def _ecgFeatures(**kwargs):
        """
        Compute features for the heart rate.
        """

        df = kwargs['df_scenario_2']
        df_norm = kwargs['df_scenario_1']

        hr, ibi = processECG(df)
        hr_norm, ibi_norm = processECG(df_norm)

        return {
            'mean_heart_rate': np.mean(hr) - np.mean(hr_norm),
            'std_heart_rate': np.std(hr) - np.std(hr_norm),
            'mean_ibi': np.mean(ibi) - np.mean(ibi_norm),
            'std_ibi': np.std(ibi) - np.std(ibi_norm)
        }

    @staticmethod
    def _fcFeatures(**kwargs):
        """
        Compute features for the flight commands.
        """

        df = kwargs['df_scenario_2']

        # Commands
        cmd_coll = np.array(df['cmd_coll'])
        cmd_yaw = np.array(df['cmd_yaw'])
        cmd_pitch = np.array(df['cmd_pitch'])
        cmd_roll = np.array(df['cmd_roll'])

        # Force
        force_coll = np.array(df['force_coll'])
        force_yaw = np.array(df['force_lyaw'])  # Left pedal
        force_pitch = np.array(df['force_pitch'])
        force_roll = np.array(df['force_roll'])

        return {
            'mean_cmd_coll': cmd_coll.mean(),
            'std_cmd_coll': cmd_coll.std(),
            'mean_cmd_yaw': cmd_yaw.mean(),
            'std_cmd_yaw': cmd_yaw.std(),
            'mean_cmd_pitch': cmd_pitch.mean(),
            'std_cmd_pitch': cmd_pitch.std(),
            'mean_cmd_roll': cmd_roll.mean(),
            'std_cmd_roll': cmd_roll.std(),
            'mean_force_coll': force_coll.mean(),
            'std_force_coll': force_coll.std(),
            'mean_force_yaw': force_yaw.mean(),
            'std_force_yaw': force_yaw.std(),
            'mean_force_pitch': force_pitch.mean(),
            'std_force_pitch': force_pitch.std(),
            'mean_force_roll': force_roll.mean(),
            'std_force_roll': force_roll.std()
        }

    @staticmethod
    def _rcFeatures(**kwargs):
        """
        Compute features for the radio communications.
        """

        df = kwargs['df_scenario_2']
        window = kwargs['window']

        time_spent_coms, total_duration = processRC(df, window)

        return {
            'proportion_time_spent_coms': 100*time_spent_coms/total_duration
        }

    @staticmethod
    def _sedFeatures(**kwargs):
        """
        Compute features for the eye movements.

        All features are normalized by their respective values 
        over the whole first scenario.
        """

        df = kwargs['df_scenario_2']
        df_norm = kwargs['df_scenario_1']

        mean_fix_dur, mean_sacc_dur, mean_sacc_amp = processEyeMovements(df)
        mean_fix_dur_norm, mean_sacc_dur_norm, mean_sacc_amp_norm = processEyeMovements(
            df_norm)

        return {
            'mean_fixation_duration': mean_fix_dur - mean_fix_dur_norm,
            'mean_saccade_duration': mean_sacc_dur - mean_sacc_dur_norm,
            'mean_saccade_amplitude': mean_sacc_amp - mean_sacc_amp_norm
        }

    @staticmethod
    def _addFeatures(data, features, i_window):
        """
        Add features to the data dictionary.
        """

        for feature, value in features.items():
            # print(f'\t\t\tfeature={feature} - value={value}')

            try:
                data[feature][i_window] = value

            except KeyError:
                raise RuntimeError(
                    'Mismatch between _feature_dictionary and features computation.'
                )

        return data

    @staticmethod
    def _updateFeaturesDataFrame(df_features, data, evaluation_type):
        """
        Update features dataframe with the new features data.
        """

        # Transform data dictionary into a DataFrame
        df_data = pd.DataFrame(data=data)

        # Concatenate new data to the current features DataFrame
        df_features = pd.concat([df_features, df_data], axis=1)

        # Rename columns with evaluation type
        columns = list(data.keys())
        df_features.rename(
            columns={col: f'feature_{evaluation_type}_{col}' for col in columns},
            inplace=True
        )

        return df_features
