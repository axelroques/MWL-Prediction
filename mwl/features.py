
import numpy as np
from scipy import stats


from .preprocessing import dic_aoi_groups, to_categorical, aoi_groups


def compute_features_br(table_br):

    br = np.array(table_br["breath_rate"])
    time = np.array(table_br["reltime"])
    return {"mean_breath_rate": np.mean(br),
            "std_breath_rate": np.std(br)
            }


def compute_features_hr(table_hr):
    hr = np.array(table_hr["heart_rate"])
    time = np.array(table_hr["reltime"])
    return {"mean_heart_rate": np.mean(hr),
            }


def compute_features_ibi(table_ibi):

    ibi = np.array(table_ibi["ibi"])

    time = np.array(table_ibi["reltime"])

    hrv = ibi[1:] - ibi[:-1]
    timev = time[:-1]

    return {"mean_hrv": np.mean(hrv)}


def compute_features_sed(table_sed):

    # gazeX = table_sed['gazeDir.x']
    # gazeY = table_sed['gazeDir.y']
    # gazeZ = table_sed['gazeDir.z']
    # gaze=np.array([gazeX, gazeY, gazeZ]).T

    gazeX = table_sed["eye_movements_horizontal"]
    gazeY = table_sed["eye_movements_vertical"]
    gaze = np.array([gazeX, gazeY]).T

    time = np.array(table_sed["reltime"])

    dist_vector = np.linalg.norm(gaze[1:, :]-gaze[:-1, :], axis=1)

    speed = dist_vector / (time[1:] - time[:-1])

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1)
    # ax.hist(speed[speed<np.percentile(speed,99)], bins=100)

    # threshold_fixations = 100
    # threshold_saccades = 30

    threshold_fixations = 30
    threshold_saccades = 70
    # Qualitative and Quantitative Scoring and Evaluation of the Eye Movement Classification Algorithms

    is_fixation = (speed < threshold_fixations)
    is_saccade = (speed > threshold_saccades)

    time_fixation = np.sum(is_fixation)
    time_saccade = np.sum(is_saccade)

    fixations_duration = []
    saccades_duration = []
    saccades_amplitude = []

    number_fixations = 0
    number_saccades = 0

    for i in range(len(speed)):

        if speed[i] >= 100:
            continue

        j = i+1
        while j < len(speed):

            if speed[j] < 100:
                j += 1

            else:
                fixations_duration.append(time[j]-time[i])
                number_fixations += 1
                break

        i = j+1

    for i in range(len(speed)):

        if speed[i] <= 300:
            continue

        j = i+1
        while j < len(speed):

            if speed[j] >= 300:
                j += 1

            else:
                saccades_duration.append(time[j]-time[i])
                saccades_amplitude.append(gaze[j] - gaze[i])
                number_saccades += 1
                break

        i = j+1

    mean_fixations_duration = np.mean(fixations_duration)

    mean_saccades_duration = np.mean(saccades_duration)

    mean_saccades_amplitude = np.mean(saccades_amplitude)

    pupil_without_blinks = table_sed["pupil_without_blinks"]

    mean_pupil = np.mean(pupil_without_blinks)
    std_pupil = np.std(pupil_without_blinks)

    blinks_start = table_sed["start_blinks"]
    blinks_duration = table_sed["blinks_duration"]
    non_null_blinks_duration = blinks_duration[(blinks_duration > 0)]

    number_blinks = np.sum(blinks_start)

    duration = time[-1]-time[0]

    return {
        # "mean_pupil": mean_pupil,
        # "std_pupil": std_pupil,
        "blink_rate": number_blinks/duration,

        "blink_rate (-)": number_blinks/duration,

        "mean_blinks_duration": np.mean(blinks_duration),

        # "fixation_time_proportion": time_fixation/(time[-1] - time[0]),


        # "fixation_rate" = number_fixations/

        "mean_fixations_duration": mean_fixations_duration,
        "mean_saccades_duration": mean_saccades_duration,

        # "saccades_rate": number_saccades/duration,

        "mean_saccades_amplitude": mean_saccades_amplitude,


    }


def compute_features_aoi(table_aoi):

    aoi_mapped_to_groups = np.array(table_aoi["aoi_mapped_to_groups"])

    time = np.array(table_aoi["reltime"])

    duration = time[-1]-time[0]

    fixations = []
    gaze_outside = 0

    breaks = []

    aoi_categ = to_categorical(
        aoi_mapped_to_groups, num_classes=len(dic_aoi_groups))

    vect_aoi = np.sum(aoi_categ, axis=0)

    vect_aoi = vect_aoi[:-1]

    i = 0

    while i < len(aoi_mapped_to_groups)-2:

        j = i+1
        while j < len(aoi_mapped_to_groups)-1:
            if aoi_mapped_to_groups[i] == aoi_mapped_to_groups[j]:
                j += 1
            else:
                break
            if aoi_mapped_to_groups[j] == 1:
                gaze_outside += 1

        fixations.append(j-i)
        i = j

    time_spent_in_fixations = np.sum(fixations)

    left = np.array(table_aoi["left"]).reshape(-1, 1)
    top = np.array(table_aoi["top"]).reshape(-1, 1)

    gaze_ellipse = confidence_ellipse_area(np.concatenate([left, top], axis=1))

    dic_features_aoi = {"time_spent_"+aoi_groups[k]: 100*(
        list(vect_aoi)[k]/len(aoi_categ)) for k in range(len(aoi_groups)-1)}

    dic_features_aoi.update({"percent_time_fixations_AOI": 100*(time_spent_in_fixations/(time[-1]-time[0])),
                             "gaze_ellipse": 100*(gaze_ellipse / (1024*768))
                             })

    return dic_features_aoi


def confidence_ellipse_area(signal):

    signal = signal - np.mean(signal, axis=0)

    cov = (1/len(signal)*np.sum(signal[:, 0]*signal[:, 1]))

    s_x = np.std(signal[:, 0])
    s_y = np.std(signal[:, 1])

    n = len(signal)

    confidence = 0.95
    quant = stats.f.ppf(confidence, 2, n-2)

    coeff = ((n+1)*(n-1)) / (n*(n-2))

    det = (s_x**2)*(s_y**2) - cov**2
    area = 2 * np.pi * quant * np.sqrt(det) * coeff

    return area


def compute_radio_com(table_rc):

    p2t = np.array(table_rc["p2t_pilot"])
    time = np.array(table_rc["reltime"])

    return {"time_spent_communication": 100 * np.sum(p2t[:-1]*(time[1:]-time[:-1])) / (time[-1]-time[0])
            }


def compute_flight_command(table_fc):

    time = np.array(table_fc["reltime"])

    f1 = np.array(table_fc["cmd_yaw"])
    f2 = np.array(table_fc["cmd_roll"])
    f3 = np.array(table_fc["cmd_coll"])
    f4 = np.array(table_fc["cmd_pitch"])

    cf1 = (f1[1:] != f1[:-1])
    cf2 = (f2[1:] != f2[:-1])
    cf3 = (f3[1:] != f3[:-1])
    cf4 = (f4[1:] != f4[:-1])

    cf = cf1 + cf2 + cf3 + cf4

    return {"number_flights_commands": (np.sum(cf) / (time[-1]-time[0]))}


def compute_helicopter_movements(table_am):

    alti = np.array(table_am["baro_alti"])
    yaw = np.array(table_am["yaw"])

    return {"std_alti": np.std(alti),
            "std_yaw": np.std(yaw)
            }


def compute_all_features(dic_data):

    list_functions = [(compute_features_br, ["br"]),
                      (compute_features_hr, ["hr"]),
                      (compute_features_sed, ["sed"]),
                      (compute_features_aoi, ["aoi"]),
                      (compute_radio_com, ["rc"]),
                      (compute_flight_command, ["fc"]),
                      (compute_helicopter_movements, ["am"]),
                      (compute_features_ibi, ["ibi"])]

    dic_features = {}

    for function_infos in list_functions:

        function = function_infos[0]
        data_types = function_infos[1]

        if np.product([data_type in dic_data for data_type in data_types]) == 0:
            continue

        features = function(*[dic_data[data_type] for data_type in data_types])

        dic_features.update(features)

    return dic_features


name_br_features = ["mean_breath_rate", "std_breath_rate"]
name_hr_features = ["mean_heart_rate"]
name_ibi_features = ["mean_hrv"]
name_eye_tracking_features = [
    # "mean_pupil", "std_pupil",
    "blink_rate", "blink_rate (-)", "mean_blinks_duration", "mean_fixations_duration", \
    "mean_saccades_duration", "mean_saccades_amplitude"]


name_comportemental_features = ['time_spent_Outside view', 'time_spent_WP', 'time_spent_Block, Speed, Horiz, Vario, Rot',
                                'time_spent_PFD/ND', 'time_spent_CAD, VEMD', 'time_spent_GNS', 'time_spent_APMS', 'time_spent_ICP',
                                'time_spent_ACU', 'time_spent_ADF, XPDR, RCU', 'time_spent_Over head panel', 'time_fixations_AOI',
                                'gaze_ellipse', 'time_spent_communication', 'number_flights_commands']

list_features_to_normalize = name_br_features + name_hr_features + name_ibi_features + \
    name_eye_tracking_features + ["time_fixations_AOI", "gaze_ellipse"]


reverse_features = ["blink_rate (-)", "mean_blinks_duration", "mean_saccades_duration", "mean_saccades_amplitude",
                    "gaze_ellipse", "mean_hrv", "std_breath_rate"]


name_features = name_br_features + name_hr_features + \
    name_ibi_features + name_eye_tracking_features

inds_no_norm = []


dic_variables = {"physiological": name_br_features+name_hr_features+name_ibi_features+name_eye_tracking_features,
                 "comportemental": name_comportemental_features}
