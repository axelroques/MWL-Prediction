
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from .paths import savepath, datapath
from .processing import processAOI, AOI_groups
from .processing import processASL, APmodes
from .processing import processEyeMovements
from .processing import processECG
from .processing import processAM
from .processing import processFC
from .processing import processRC


class ContinuousData:

    def __init__(
        self,
        compute_features=False,
        pilot=None,
        dt=None
    ) -> None:

        self._pilot = pilot
        self._dt = dt

        # Must coincide with what is done during the feature computation step
        self._feature_dictionary = {
            'am': [
                'std_helico_altitude', 'mean_cross_helico_altitude',
                'std_helico_yaw', 'mean_cross_helico_yaw',
                'mean_helico_pitch', 'std_helico_pitch', 'mean_cross_helico_pitch',
                'mean_helico_roll', 'std_helico_roll', 'mean_cross_helico_roll'
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
                'mean_cmd_coll', 'std_cmd_coll', 'mean_cross_cmd_coll',
                'mean_cmd_yaw', 'std_cmd_yaw', 'mean_cross_cmd_yaw',
                'mean_cmd_pitch', 'std_cmd_pitch', 'mean_cross_cmd_pitch',
                'mean_cmd_roll', 'std_cmd_roll', 'mean_cross_cmd_roll',
                'mean_force_coll', 'std_force_coll', 'mean_cross_force_coll',
                'mean_force_yaw', 'std_force_yaw', 'mean_cross_force_yaw',
                'mean_force_pitch', 'std_force_pitch', 'mean_cross_force_pitch',
                'mean_force_roll', 'std_force_roll', 'mean_cross_force_roll'
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
            with open(Path(
                    savepath,
                    f'continuous_features_pilot_{self._pilot}_dt_{self._dt}.pkl'), "rb") as file:
                self._all_features = pickle.load(file)

        # By default, without any selection, the feature set is complete
        self._features = self._all_features.copy()
        self.exclude_files = []
        self.exclude_pilots = []

        print('Done!')

    def getFeatures(self):
        """
        Simply return features.
        """
        return self._features

    def _computeFeatures(self):
        """
        Compute a DataFrame of features for a given pilot.
        This DataFrame will be used to compute the continuous MWL,
        where continuous means that the features are computed every
        dt seconds.
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

        # Get windows
        t_evaluations, windows = self._getWindows()

        # Initialize data structure that will hold the features
        data = {
            feature: np.zeros(len(windows)) for feature in self.features_list
        }

        # Iterate over the feature type
        for filename in self._feature_dictionary:

            # If the filename should be normalized, load the first scenario
            if filename in self.features_to_normalize:
                df_scenario_1 = self._load(1, self._pilot, filename)
                normalization = process[filename](
                    df=df_scenario_1,
                    window=window,
                    norm=True
                )

            # Load scenario 2
            df_scenario_2 = self._load(2, self._pilot, filename)

            # Iterate over the windows
            for i_window, window in enumerate(windows):

                # Cut DataFrame around the evaluation window
                cut = self._cut(df_scenario_2, window)

                # Compute features on the resulting DataFrame
                features = process[filename](
                    df=cut,
                    window=window,
                    norm=False
                )

                # Eventually normalize features
                if filename in self.features_to_normalize:
                    features = self._normalizeFeatures(features, normalization)

                # Update data dictionary
                data = self._addFeatures(data, features, i_window)

        # Create a features DataFrame

        df_features = pd.DataFrame(data={'t': t_evaluations, **data})

        return df_features

    def _getWindows(self):
        """
        Return all windows (-40s -> +10s) around each evaluation point.
        Evaluations start from the 1st min of the 2nd scenario and then
        are sampled one every self._dt seconds.
        """

        # Load a file just to get the total experiment duration
        df = self._load(2, self._pilot, 'am')
        duration = int(df['reltime'].iloc[-1])

        # Windows
        t_evaluations = [*range(60, duration-10, self._dt)]
        windows = [
            (center-40, center+10)
            for center in t_evaluations
        ]

        return t_evaluations, windows

    def _saveFeatures(self):
        """
        Save the computed features with pickle.
        """

        with open(Path(
                savepath,
                f'continuous_features_pilot_{self._pilot}_dt_{self._dt}.pkl'), "wb") as file:
            pickle.dump(
                self._all_features,
                file,
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
    def _normalizeFeatures(features, norm):
        """
        Normalize features.
        """

        for feature, value in norm.items():
            features[feature] -= value

        return features

    @staticmethod
    def _amFeatures(**kwargs):
        """
        Compute features for the aircraft motion.
        """

        df = kwargs['df']

        std_altitude, mean_cross_altitude, \
            std_helico_yaw, mean_cross_yaw, \
            mean_pitch, std_pitch, mean_cross_pitch, \
            mean_roll, std_roll, mean_cross_roll = processAM(df)

        return {
            'std_helico_altitude': std_altitude,
            'mean_cross_helico_altitude': mean_cross_altitude,
            'std_helico_yaw': std_helico_yaw,
            'mean_cross_helico_yaw': mean_cross_yaw,
            'mean_helico_pitch': mean_pitch,
            'std_helico_pitch': std_pitch,
            'mean_cross_helico_pitch': mean_cross_pitch,
            'mean_helico_roll': mean_roll,
            'std_helico_roll': std_roll,
            'mean_cross_helico_roll': mean_cross_roll
        }

    @staticmethod
    def _aoiFeatures(**kwargs):
        """
        Compute features for the AOI data.

        The gaze_ellipse_area is normalized by its value 
        over the whole first scenario.
        """

        df = kwargs['df']

        if kwargs['norm']:

            gaze_ellipse_area, _ = processAOI(df)

            return {'gaze_ellipse_area': gaze_ellipse_area}

        gaze_ellipse_area, time_spent = processAOI(df)

        # time_spent will now contain all necessary features
        time_spent.update({
            'gaze_ellipse_area': gaze_ellipse_area
        })

        return time_spent

    @staticmethod
    def _aslFeatures(**kwargs):
        """
        Compute features for the automatic pilot.
        """

        df = kwargs['df']

        time_spent = processASL(df)

        return time_spent

    @staticmethod
    def _brFeatures(**kwargs):
        """
        Compute features for the heart rate.
        """

        df = kwargs['df']

        br = np.array(df['breath_rate'])

        return {
            'mean_breathing_rate': np.mean(br),
            'std_breathing_rate': np.std(br)
        }

    @staticmethod
    def _ecgFeatures(**kwargs):
        """
        Compute features for the heart rate.
        """

        df = kwargs['df']

        hr, ibi = processECG(df)

        return {
            'mean_heart_rate': np.mean(hr),
            'std_heart_rate': np.std(hr),
            'mean_ibi': np.mean(ibi),
            'std_ibi': np.std(ibi)
        }

    @staticmethod
    def _fcFeatures(**kwargs):
        """
        Compute features for the flight commands.
        """

        df = kwargs['df']

        mean_cmd_coll, std_cmd_coll, mean_cross_cmd_coll, \
            mean_cmd_yaw, std_cmd_yaw, mean_cross_cmd_yaw, \
            mean_cmd_pitch, std_cmd_pitch, mean_cross_cmd_pitch, \
            mean_cmd_roll, std_cmd_roll, mean_cross_cmd_roll, \
            mean_force_coll, std_force_coll, mean_cross_force_coll, \
            mean_force_yaw, std_force_yaw, mean_cross_force_yaw, \
            mean_force_pitch, std_force_pitch, mean_cross_force_pitch, \
            mean_force_roll, std_force_roll, mean_cross_force_roll = processFC(
                df
            )

        return {
            'mean_cmd_coll': mean_cmd_coll,
            'std_cmd_coll': std_cmd_coll,
            'mean_cross_cmd_coll': mean_cross_cmd_coll,
            'mean_cmd_yaw': mean_cmd_yaw,
            'std_cmd_yaw': std_cmd_yaw,
            'mean_cross_cmd_yaw': mean_cross_cmd_yaw,
            'mean_cmd_pitch': mean_cmd_pitch,
            'std_cmd_pitch': std_cmd_pitch,
            'mean_cross_cmd_pitch': mean_cross_cmd_pitch,
            'mean_cmd_roll': mean_cmd_roll,
            'std_cmd_roll': std_cmd_roll,
            'mean_cross_cmd_roll': mean_cross_cmd_roll,
            'mean_force_coll': mean_force_coll,
            'std_force_coll': std_force_coll,
            'mean_cross_force_coll': mean_cross_force_coll,
            'mean_force_yaw': mean_force_yaw,
            'std_force_yaw': std_force_yaw,
            'mean_cross_force_yaw': mean_cross_force_yaw,
            'mean_force_pitch': mean_force_pitch,
            'std_force_pitch': std_force_pitch,
            'mean_cross_force_pitch': mean_cross_force_pitch,
            'mean_force_roll': mean_force_roll,
            'std_force_roll': std_force_roll,
            'mean_cross_force_roll': mean_cross_force_roll
        }

    @staticmethod
    def _rcFeatures(**kwargs):
        """
        Compute features for the radio communications.
        """

        df = kwargs['df']
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

        df = kwargs['df']

        mean_fix_dur, mean_sacc_dur, mean_sacc_amp = processEyeMovements(df)

        return {
            'mean_fixation_duration': mean_fix_dur,
            'mean_saccade_duration': mean_sacc_dur,
            'mean_saccade_amplitude': mean_sacc_amp
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
