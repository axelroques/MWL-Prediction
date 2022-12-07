
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from .paths import savepath, datapath, evalpath
from.processing import processEyeMovements, processAOI, AOI_groups


class Data:

    def __init__(
        self,
        compute_features=False
    ) -> None:

        # Must coincide with what is done during the feature computation step
        self._feature_dictionary = {
            'am': [
                'std_helico_altitude', 'std_helico_yaw',
                'std_helico_pitch', 'std_helico_roll'
            ],
            'fc': [
                'std_cmd_coll', 'std_cmd_yaw', 'std_cmd_pitch', 'std_cmd_roll',
                'std_force_coll', 'std_force_yaw', 'std_force_pitch', 'std_force_roll'
            ],
            'rc': [
                'proportion_time_spent_coms'
            ],
            'sed': [
                'mean_fixation_duration', 'mean_saccade_duration', 'mean_saccade_amplitude'
            ],
            'aoi': ['gaze_ellipse_area'] + [
                f'proportion_time_spent_{group}' for _, group in AOI_groups.items()
            ],
            'ecg': []
        }

        # All features together in a list
        self.features_list = [
            col for filename in self._feature_dictionary
            for col in self._feature_dictionary[filename]
        ]
        print('feature list', self.features_list)

        # Features to be normalized
        self.features_to_normalize = ['sed', 'aoi', 'ecg']

        # Compute all features and save the data
        if compute_features:
            self._all_features = self._computeFeatures()
            self._saveFeatures()

        # Otherwise just load the features if they were already computed
        else:
            print('Loading features, should return error message if no pickle found.')
            with open(Path(savepath, 'features.pkl'), "rb") as file:
                self._all_features = pickle.load(file)

        # By default, without any selection, the feature set is complete
        self.features = self._all_features.copy()

    def selectFeatures(self, exclude_files=[], exclude_pilots=[2]):
        """
        Return a subset of self._all_features based on the data in 
        exclude_files and exclude_pilots.
        """

        self.exclude_files = exclude_files
        self.exclude_pilots = exclude_pilots

        self.features = self._all_features.loc[
            ~self._all_features['pilot'].isin(self.exclude_pilots),
            ~self._all_features.columns.isin(self.features_list)
        ]

        return

    def _computeFeatures(self, evaluation_type='oral_evaluation'):
        """
        Compute all features.
        """

        # Feature processing launcher
        process = {
            'am': self._amFeatures,
            'fc': self._fcFeatures,
            'rc': self._rcFeatures,
            'sed': self._sedFeatures,
            'aoi': self._aoiFeatures,
            'ecg': self._ecgFeatures,
        }

        # TO DO: Normalization features

        # DataFrame with info on where features should be computed
        df_eval = pd.read_csv(evalpath)

        # Initialize data structure that will hold the features
        data = {
            feature: [] for feature in self.features_list
        }

        for pilot in [2, 3, 4, 5, 6, 7, 8, 9]:

            # Sub-DataFrame with the mwl evaluation of the pilot
            sub_eval = df_eval.loc[df_eval['pilot'] == pilot]

            # Iterate over the evaluation windows
            for i in range(len(sub_eval)):
                for filename in self._feature_dictionary:

                    # Load raw data
                    df_scenario_1 = None
                    if filename in self.features_to_normalize:
                        df_scenario_1 = self._load(1, pilot, filename)
                    df_scenario_2 = self._load(2, pilot, filename)

                    # Get window information
                    window = self._getWindow(sub_eval, evaluation_type, i)
                    print('window =', window)

                    # Cut DataFrame around the evaluation window
                    cut = self._cut(df_scenario_2, window)

                    # Compute features on the resulting DataFrame
                    features = process[filename](
                        df_scenario_2=cut, df_scenario_1=df_scenario_1
                    )
                    print('features =', features)

                    # Update data dictionary
                    data = self._addFeatures(data, features)

        return pd.DataFrame(data=data)

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
            '18': '2018-01-23_14;30;48_eyeState2',
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
    def _getWindow(df_eval, evaluation_type, i):
        """
        Get window based on the evaluation type:
            - 'oral_evaluation': (df.iloc[i]['time_tc']-40, df.iloc[i]['time_tc']+10)
            - 'NASA-TLX': (df.iloc[i]['time_start_NASA_tlx'], 
            df.iloc[i]['time_start_NASA_tlx']+df.iloc[i]['time_NASA_tlx_window'])
            - 'theoretical_NASA-TLX': (df.iloc[i]['time_start_tc_david'], 
            df.iloc[i]['time_end_tc_david'])
        """

        if evaluation_type == 'oral_evaluation':
            window = (
                df_eval.iloc[i]['time_tc']-40,
                df_eval.iloc[i]['time_tc']+10
            )
        elif evaluation_type == 'NASA-TLX':
            window = (
                df_eval.iloc[i]['time_start_NASA_tlx'],
                df_eval.iloc[i]['time_start_NASA_tlx'] +
                df_eval.iloc[i]['time_NASA_tlx_window']
            )
        elif evaluation_type == 'theoretical_NASA-TLX':
            window = (
                df_eval.iloc[i]['time_start_tc_david'],
                df_eval.iloc[i]['time_end_tc_david']
            )
        else:
            raise RuntimeError('Unknown evaluation type.')

        return window

    @staticmethod
    def _amFeatures(**kwargs):
        """
        Compute features for the aircraft motion.
        """

        df = kwargs['df_scenario_2']

        return {
            'std_helico_altitude': np.std(df['baro_alti']),
            'std_helico_yaw': np.std(df['yaw']),
            'std_helico_pitch': np.std(df['pitch']),
            'std_helico_roll': np.std(df['roll'])
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
            'std_cmd_coll': cmd_coll.std(),
            'std_cmd_yaw': cmd_yaw.std(),
            'std_cmd_pitch': cmd_pitch.std(),
            'std_cmd_roll': cmd_roll.std(),
            'std_force_coll': force_coll.std(),
            'std_force_yaw': force_yaw.std(),
            'std_force_pitch': force_pitch.std(),
            'std_force_roll': force_roll.std()
        }

    @staticmethod
    def _rcFeatures(**kwargs):
        """
        Compute features for the radio communications.
        """

        df = kwargs['df_scenario_2']

        diff = df.loc[:, ['reltime', 'p2t_pilot']].diff()
        time_spent_coms = diff.loc[
            diff['p2t_pilot'] == -1, 'reltime'
        ].sum()
        total_duration = df['reltime'][-1] - df['reltime'][0]

        return {
            'proportion_time_spent_coms': time_spent_coms/total_duration*100
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

        return time_spent.update({
            'gaze_ellipse_area': gaze_ellipse_area - gaze_ellipse_area_norm
        })

    @staticmethod
    def _ecgFeatures(**kwargs):
        """
        Compute features for the heart rate.
        """

        df = kwargs['df_scenario_2']

        return

    @staticmethod
    def _addFeatures(data, features):
        """
        Add features to the dictionary d.
        """

        print('data =', data)
        for feature, value in features.items():
            print(f'\t feature={feature} - value={value}')
            try:
                data[feature].append(value)
            except KeyError:
                raise RuntimeError(
                    'Mismatch between _feature_dictionary and features computation.'
                )

        return data
