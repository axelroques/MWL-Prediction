
import numpy as np

from scipy.signal import butter, filtfilt

# from swarii import SWARII


# def band_pass_filter(signal, freq_signal, low_freq, high_freq):

#     freqs, signal_fft = spectrum(signal, freq_signal)
#     signal_fft[np.logical_or((freqs < low_freq), (freqs > high_freq))] = 0
#     new_signal = np.fft.irfft(signal_fft, axis=0)

#     return new_signal


def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='lowpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs=100, order=5):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y


dic_aoi_groups = {"Unknown": 12,
                  "Front panel": 0,
                  "Outside view": 1,
                  "WP": 2,
                  "Block, Speed, Horiz, Vario, Rot": 3,
                  "PFD/ND": 4,
                  "CAD, VEMD": 5,
                  "GNS": 6,
                  "APMS": 7,
                  "ICP": 8,
                  "ACU": 9,
                  "ADF, XPDR, RCU": 10,
                  "Over head panel": 11}

aoi_groups = ["Front panel", "Outside view", "WP", "Block, Speed, Horiz, Vario, Rot",
              "PFD/ND", "CAD, VEMD", "GNS", "APMS", "ICP", "ACU", "ADF, XPDR, RCU", "Over head panel", "Unknown"]


def map_aoi_to_groups(aoi):

    if aoi == 0:
        return 12  # Unknown
    if aoi == 2:
        return 0  # FrontPanel
    if aoi == 1:
        return 1  # outside view
    if aoi == 3:
        return 2  # WP
    if (aoi >= 4 and aoi <= 8):
        return 3  # Block, Speed, Horiz, Vario, Rot
    if (aoi >= 9 and aoi <= 12):
        return 4  # PFD1, ND1, PFD2, ND2
    if (aoi >= 13 and aoi <= 16):  # CAD, ???, VEMD_U, VEM_L
        return 5
    if (aoi >= 17 and aoi <= 18):
        return 6  # GNS1, GNS2
    if aoi == 19:
        return 7  # APMS
    if (aoi >= 20 and aoi <= 21):  # ICP1, ICP2
        return 8
    if (aoi >= 22 and aoi <= 23):  # ACU1, ACU2
        return 9
    if (aoi >= 24 and aoi <= 26):  # ADF, XPDR, RCU
        print('all good')
        return 10
    if aoi == 27:  # OVerHeadPanel
        return 11


def find_blinks(time_pupil, pupil):

    starting_lower_threshold = 0.5*np.mean(pupil[pupil != 0])
    minimum_depth_threshold = 0

    blinks = []

    i = 0

    while i+1 < len(pupil):
        j = i+1

        if pupil[j] >= pupil[i] or (pupil[i] < starting_lower_threshold and i > 0):
            i = j
            continue

        min_value = pupil[j]
        min_index = j
#
        if pupil[i] < starting_lower_threshold:
            min_value = pupil[i]
            min_index = i

        while j+1 < len(pupil):

            # Going down

            if pupil[j] > pupil[i] or (pupil[j] > min_value):

                # Going down is over

                valid = (pupil[i] - min_value) > minimum_depth_threshold
                break

            elif pupil[j] > min_value:
                j += 1

            else:
                min_value = pupil[j]
                min_index = j
                j += 1

        if (j == len(pupil)-1 and pupil[j] < starting_lower_threshold):
            valid = True
            return blinks + [(i, pupil[i]-pupil[j], len(pupil)-1-i)]

        if not valid:

            i = j
            continue

        if j+1 >= len(pupil):
            return blinks

        j = min_index
        j += 1
        max_index = j
        max_value = pupil[j]

        while j+1 < len(pupil):

            # Going up

            if (pupil[j] < max_value):

                if (max_value <= starting_lower_threshold):
                    max_index = j
                    if pupil[j] < min_value:
                        min_value = pupil[j]
                    j += 1

                    continue

                # Going up is over

                # and (max_value> starting_lower_threshold)
                valid = (max_value - min_value >
                         minimum_depth_threshold) or (min_value == max_value == 0)

                break
            elif pupil[j] < max_value:

                if pupil[j] < min_value:
                    min_value = pupil[j]
                j += 1
            else:

                max_value = pupil[j]
                max_index = j
                if pupil[j] < min_value:
                    min_value = pupil[j]
                j += 1

        if valid:

            start_blink = i
            blink_depth = pupil[i]-min_value
            blink_duration = max_index - i  # number of points
            blinks.append((start_blink, blink_depth, blink_duration))

        i = max_index

    return blinks


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    e.g. for use with categorical_crossentropy.
    Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


# def resample(time, signal, window_size, target_frequency=100) -> None:
#     """
#     Resample the signals using SWARII
#     """

#     if len(signal.shape) == 1:
#         signal = signal.reshape(-1, 1)

#     timesig = np.concatenate([time.reshape(-1, 1), signal], axis=1)

#     resampled_signal = SWARII.resample(
#         data=timesig, desired_frequency=target_frequency, window_size=window_size)

#     resampled_time = (1/target_frequency) * \
#         np.arange(len(resampled_signal))  # Time in seconds

#     return resampled_time, resampled_signal
