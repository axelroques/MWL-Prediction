
from .ivt import IVT

from scipy.signal import find_peaks
from scipy.stats import f
import numpy as np


AOI_grouping = {
    0: "0",  # Unknown
    1: "1",  # OV
    2: "12",  # Front Panel
    3: "2",  # WP
    4: "3",  # Clock
    5: "3",  # Speed
    6: "3",  # Horiz
    7: "3",  # Vario
    8: "3",  # Rotor
    9: "4",  # PFD
    10: "4",  # ND
    11: "4",  # PFD
    12: "4",  # ND
    13: "5",  # CAD
    14: "0",  # ???
    15: "5",  # VEMD
    16: "5",  # VEMD
    17: "6",  # GNS 1
    18: "6",  # GNS 2
    19: "7",  # APMS
    20: "8",  # ICP1
    21: "8",  # ICP2
    22: "9",  # ACU1
    23: "9",  # ACU2
    24: "11",  # ADF
    25: "11",  # XPDR
    26: "11",  # RCU
    27: "10"  # OverHeadPanel
}

AOI_groups = {
    '0': "Unknown",
    '1': "Outside View",
    '2': "Warning Panel",
    '3': "Analog instruments",  # Clock, speed, horiz, varion, rotor
    '4': "PFDs & NDs",
    '5': "CPDS",  # Center Panel Display System
    '6': "GNS",
    '7': "APMS",
    '8': "ICPs",
    '9': "ACUs",
    '10': "Over Head Panel",
    '11': "ADF, XPDR & RCU",
    '12': "Front Panel"
}

APmodes = {
    'ap_horz_mode': {
        0: 'NONE',
        1: 'SAS',
        2: 'HDG',
        3: 'NAV_NMS',
        4: 'NAV_VOR',
        5: 'APP_LOC',
        6: 'APP_VORA',
        7: 'GO_AROUND',
        8: 'OTHER'
    },
    'ap_vert_mode': {
        0: 'NONE',
        1: 'SAS',
        2: 'IAS',
        3: 'ALT',
        4: 'ALTA',
        5: 'VS',
        6: 'GS',
        7: 'GO_AROUND',
        8: 'OTHER'
    }
}


def processAM(df_am):
    """
    Aircraft motion processing pipeline.

    Returns the mean and standard deviation of aircraft
    orientation. Also computes the number of times the 
    values cross the mean (indicator of rate of change).
    """

    # Altitude
    altitude = np.array(df_am['baro_alti'])
    mean_altitude = altitude.mean()
    std_altitude = altitude.std()
    centred = altitude - mean_altitude
    mean_cross_altitude = ((centred[:-1] * centred[1:]) < 0).sum()

    # Yaw
    yaw = np.array(df_am['yaw'])
    mean_yaw = yaw.mean()
    std_yaw = yaw.std()
    centred = yaw - mean_yaw
    mean_cross_yaw = ((centred[:-1] * centred[1:]) < 0).sum()

    # Pitch
    pitch = np.array(df_am['pitch'])
    mean_pitch = pitch.mean()
    std_pitch = pitch.std()
    centred = pitch - mean_pitch
    mean_cross_pitch = ((centred[:-1] * centred[1:]) < 0).sum()

    # Roll
    roll = np.array(df_am['roll'])
    mean_roll = roll.mean()
    std_roll = roll.std()
    centred = roll - mean_roll
    mean_cross_roll = ((centred[:-1] * centred[1:]) < 0).sum()

    return std_altitude, mean_cross_altitude, \
        std_yaw, mean_cross_yaw, \
        mean_pitch, std_pitch, mean_cross_pitch, \
        mean_roll, std_roll, mean_cross_roll


def processFC(df_fc):
    """
    Flight commands processing pipeline.

    Returns the mean and standard deviation of the commands. 
    Also computes the number of times the values cross the 
    mean (indicator of rate of change).
    """

    # Commands
    # Collective
    cmd_coll = np.array(df_fc['cmd_coll'])
    mean_cmd_coll = cmd_coll.mean()
    std_cmd_coll = cmd_coll.std()
    centred = cmd_coll - mean_cmd_coll
    mean_cross_cmd_coll = ((centred[:-1] * centred[1:]) < 0).sum()
    # Yaw
    cmd_yaw = np.array(df_fc['cmd_yaw'])
    mean_cmd_yaw = cmd_yaw.mean()
    std_cmd_yaw = cmd_yaw.std()
    centred = cmd_yaw - mean_cmd_yaw
    mean_cross_cmd_yaw = ((centred[:-1] * centred[1:]) < 0).sum()
    # Pitch
    cmd_pitch = np.array(df_fc['cmd_pitch'])
    mean_cmd_pitch = cmd_pitch.mean()
    std_cmd_pitch = cmd_pitch.std()
    centred = cmd_pitch - mean_cmd_pitch
    mean_cross_cmd_pitch = ((centred[:-1] * centred[1:]) < 0).sum()
    # Roll
    cmd_roll = np.array(df_fc['cmd_roll'])
    mean_cmd_roll = cmd_roll.mean()
    std_cmd_roll = cmd_roll.std()
    centred = cmd_roll - mean_cmd_roll
    mean_cross_cmd_roll = ((centred[:-1] * centred[1:]) < 0).sum()

    # Force
    # Collective
    force_coll = np.array(df_fc['force_coll'])
    mean_force_coll = force_coll.mean()
    std_force_coll = force_coll.std()
    centred = force_coll - mean_force_coll
    mean_cross_force_coll = ((centred[:-1] * centred[1:]) < 0).sum()
    # Yaw
    force_yaw = np.array(df_fc['force_lyaw'])  # Left pedal
    mean_force_yaw = force_yaw.mean()
    std_force_yaw = force_yaw.std()
    centred = force_yaw - mean_force_yaw
    mean_cross_force_yaw = ((centred[:-1] * centred[1:]) < 0).sum()
    # Pitch
    force_pitch = np.array(df_fc['force_pitch'])
    mean_force_pitch = force_pitch.mean()
    std_force_pitch = force_pitch.std()
    centred = force_pitch - mean_force_pitch
    mean_cross_force_pitch = ((centred[:-1] * centred[1:]) < 0).sum()
    # Roll
    force_roll = np.array(df_fc['force_roll'])
    mean_force_roll = force_roll.mean()
    std_force_roll = force_roll.std()
    centred = force_roll - mean_force_roll
    mean_cross_force_roll = ((centred[:-1] * centred[1:]) < 0).sum()

    return mean_cmd_coll, std_cmd_coll, mean_cross_cmd_coll, \
        mean_cmd_yaw, std_cmd_yaw, mean_cross_cmd_yaw, \
        mean_cmd_pitch, std_cmd_pitch, mean_cross_cmd_pitch, \
        mean_cmd_roll, std_cmd_roll, mean_cross_cmd_roll, \
        mean_force_coll, std_force_coll, mean_cross_force_coll, \
        mean_force_yaw, std_force_yaw, mean_cross_force_yaw, \
        mean_force_pitch, std_force_pitch, mean_cross_force_pitch, \
        mean_force_roll, std_force_roll, mean_cross_force_roll


def processRC(df_rc, window):
    """
    Radio communication processing pipeline.

    Outputs the time spent in communications.
    """

    diff = df_rc.loc[:, ['reltime', 'p2t_pilot']].diff()
    time_spent_coms = diff.loc[
        diff['p2t_pilot'] == -1, 'reltime'
    ].sum()
    total_duration = window[1] - window[0]

    return time_spent_coms, total_duration


def processASL(df_asl):
    """
    Automatic pilot processing pipeline.

    Returns the time spent in each submode of the automatic pilot.
    """

    # Initialize dictionary from the input AP modes
    keys = [
        f'time_spent_{top_key}_{item}'
        for top_key, mode in APmodes.items()
        for _, item in mode.items()
    ]
    time_spent = dict.fromkeys(keys, 0)

    # If DataFrame is empty, return a null dictionary
    if df_asl.empty:
        return time_spent

    # Subset of the dataframe
    df_ap = df_asl.loc[:, ['reltime', 'ap_horz_mode', 'ap_vert_mode']]

    # Get AP durations
    diff = df_ap.diff().dropna()
    for mode in ['ap_horz_mode', 'ap_vert_mode']:

        sub = diff[mode]

        # Get breakpoints
        breakpoints = sub.loc[sub != 0].index.tolist()

        # Add start and end
        breakpoints = [df_ap.index[0]] + [*breakpoints] + [df_ap.index[-1]]

        for bp1, bp2 in zip(breakpoints[:-1], breakpoints[1:]):
            bp1 -= df_ap.index[0]  # Reset index of breakpoint 1
            bp2 -= df_ap.index[0]  # Reset index of breakpoint 2
            start_value = df_ap.iloc[bp1, df_ap.columns.get_loc(mode)]
            submode = APmodes[mode][start_value]
            duration = df_ap.iloc[bp2, 0] - df_ap.iloc[bp1, 0]
            time_spent[f'time_spent_{mode}_{submode}'] += duration

    return time_spent


def processEyeMovements(df_sed):
    """
    Eye movement processing pipeline.

    Retrieves eye movements from gaze + head 
    data. 
    """

    def angles(yaw, pitch, roll):
        """
        Change of reference frame: from mobile 'head' reference frame to a fixed
        reference (the cockpit). 
        """

        m11 = np.cos(yaw) * np.cos(pitch)
        m12 = np.sin(yaw) * np.cos(pitch)
        m13 = -np.sin(pitch)

        m21 = -np.sin(yaw) * np.cos(roll) + np.cos(yaw) * \
            np.sin(pitch) * np.sin(roll)
        m22 = np.cos(yaw) * np.cos(roll) + np.sin(yaw) * \
            np.sin(pitch) * np.sin(roll)
        m23 = np.cos(pitch) * np.sin(roll)

        m31 = np.sin(yaw) * np.sin(roll) + np.cos(yaw) * \
            np.sin(pitch) * np.cos(roll)
        m32 = -np.cos(yaw) * np.sin(roll) + np.sin(yaw) * \
            np.sin(pitch) * np.cos(roll)
        m33 = np.cos(pitch) * np.cos(roll)

        return np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])

    def atan2(a, b):
        """
        Trigo.
        """
        return np.arctan2(a, b)

    # Get horizontal and vertical eye movement
    x_eye = []
    y_eye = []
    for i in range(len(df_sed)):

        # 3D gaze vector
        gaze_direction = np.array([
            df_sed['gazeDir.x'].iloc[i],
            df_sed['gazeDir.y'].iloc[i],
            df_sed['gazeDir.z'].iloc[i]
        ])

        # Compute projection matrix
        M = angles(
            -np.deg2rad(df_sed['headYaw'].iloc[i]),
            -np.deg2rad(df_sed['headPitch'].iloc[i]),
            np.deg2rad(df_sed['headRoll'].iloc[i])
        )

        # Project
        gaze_direction = M.dot(gaze_direction)

        # Get horizontal and vertical components - GAZE
        # Current gaze horizontal angle (rad)
        horizontal = atan2(-gaze_direction[1], gaze_direction[0])
        # Current gaze vertical angle (rad)
        vertical = atan2(gaze_direction[2], gaze_direction[0])

        # Get horizontal and vertical components - EYES
        horizontal = np.rad2deg(horizontal) - df_sed['headYaw'].iloc[i]
        vertical = np.rad2deg(vertical) - df_sed['headPitch'].iloc[i]

        # Store new components
        x_eye.append(horizontal)
        y_eye.append(vertical)

    # Classify fixations and saccades using the I-VT algorithm
    classifier = IVT(
        df_sed['reltime'], x_eye, y_eye, threshold=40
    )
    fixations, saccades = classifier.process()

    # Get mean fixation duration
    fix_durations = [fix['duration'] for fix in fixations]
    mean_fix_dur = np.mean(fix_durations)

    # Get mean saccade duration
    sacc_durations = [sacc['duration'] for sacc in saccades]
    mean_sacc_dur = np.mean(sacc_durations)

    # Get mean saccade amplitude
    sacc_amplitudes = [sacc['amplitude'] for sacc in saccades]
    mean_sacc_amp = np.mean(sacc_amplitudes)

    return mean_fix_dur, mean_sacc_dur, mean_sacc_amp


def processAOI(df_aoi):
    """
    AOI processing pipeline.
    """

    def confidence_ellipse_area(xy_signal):
        """
        95% confidence gaze ellipse area, according to 
        Schubert and Kirchner (2014).
        """

        signal = xy_signal - np.mean(xy_signal, axis=0)

        n = len(signal)
        cov = (1/len(signal)*np.sum(signal[:, 0]*signal[:, 1]))
        s_x = np.std(signal[:, 0])
        s_y = np.std(signal[:, 1])

        confidence = 0.95
        quant = f.ppf(confidence, 2, n-2)
        coeff = ((n+1)*(n-1)) / (n*(n-2))
        det = (s_x**2)*(s_y**2) - cov**2
        area = 2 * np.pi * quant * np.sqrt(det) * coeff * (1/n)  # Modif here

        return area

    # 95% confidence gaze ellipse area
    left = np.array(df_aoi['left']).reshape(-1, 1)
    top = np.array(df_aoi['top']).reshape(-1, 1)
    gaze_ellipse = confidence_ellipse_area(
        np.concatenate([left, top], axis=1)
    )
    gaze_ellipse_normed = 100*(gaze_ellipse/(1280*768))  # Corrected norm

    # % of time spent in each category
    dt = np.mean(df_aoi['reltime'].diff())  # Mean inter-sample duration
    time_spent = df_aoi['aoi'].value_counts()*dt  # Count AOI hits
    time_spent = time_spent.to_dict()  # Convert to dictionary
    time_spent_labelled = dict.fromkeys(
        [f'proportion_time_spent_{aoi}' for aoi in AOI_groups.values()], 0
    )  # Initialize dict with standardized dictionary key names
    for aoi, value in time_spent.items():
        time_spent_labelled[
            f'proportion_time_spent_{AOI_groups[AOI_grouping[aoi]]}'
        ] += value  # Update dictionary with window values

    return gaze_ellipse_normed, time_spent_labelled


def processECG(df_ecg):
    """
    ECG processing pipeline.

    Outputs the heart rate and the IBI.
    """

    # Get ECG lead and associated time indices
    t_indices = np.array(df_ecg['reltime'])
    lead = np.array(df_ecg['lead1'])

    # Compute R-R interval
    # Distance and height parameters are optimal for all pilots
    rr_indices, _ = find_peaks(lead, distance=100, height=0.15)

    # Get the corresponding times
    beats_timestamps = t_indices[rr_indices]
    hr = []
    ibi = []

    # Compute the heart rate and inter-beat interval
    for beat1, beat2 in zip(beats_timestamps[:-1], beats_timestamps[1:]):
        hr.append(60/(beat2-beat1))
        ibi.append(1000*(beat2-beat1))

    return np.array(hr), np.array(ibi)
