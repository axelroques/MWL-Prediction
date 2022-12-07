
from .ivt import IVT

from scipy.stats import f
import numpy as np


AOI_grouping = {
    0: "0",  # Unknown
    1: "1",  # OV
    2: "x",  # Front Panel
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
    '11': "ADF, XPDR & RCU"
}


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
    # TO DO
    mean_fix_dur = None

    # Get mean saccade duration
    # TO DO
    mean_sacc_dur = None

    # Get mean saccade amplitude
    # TO DO
    mean_sacc_amp = None

    return mean_fix_dur, mean_sacc_dur, mean_sacc_amp


def processAOI(df_aoi):
    """
    AOI processing pipeline.

    """

    def confidence_ellipse_area(xy_signal):
        """
        95% confidence gaze ellipse area, supposedly 
        according to Schubert and Kirchner (2014).
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
        area = 2 * np.pi * quant * np.sqrt(det) * coeff

        return area

    # 95% confidence gaze ellipse area
    left = np.array(df_aoi['left']).reshape(-1, 1)
    top = np.array(df_aoi['top']).reshape(-1, 1)
    gaze_ellipse = confidence_ellipse_area(
        np.concatenate([left, top], axis=1)
    )
    gaze_ellipse_normed = 100*(gaze_ellipse/(1280*768))  # Corrected norm

    # % of time spent in each category
    dt = np.mean(df_aoi['reltime'].diff())
    time_spent = df_aoi['aoi'].value_counts()*dt
    time_spent_labelled = {
        f'proportion_time_spent_{AOI_groups[AOI_grouping[aoi]]}': value
        for aoi, value in time_spent.items()
    }

    return gaze_ellipse_normed, time_spent_labelled
