
from .features import compute_all_features, list_features_to_normalize
from .preprocessing import find_blinks, map_aoi_to_groups
# from .processing import process
from .paths import datapath, savepath

from scipy import interpolate
from pathlib import Path
import pickle
import os
import pandas
import numpy as np
import h5py


path_h5_data = Path(datapath, "h5")  # directory with h5 data files
# directory with expert annotations files
path_david_files = Path(datapath, "Fdavid")


nb_aoi_groups = 13

list_data = ["br", "hr", "sed", "aoi", "rc", "fc", "am", "ibi"]


cols_NASA_tlx = ["mental_demand",
                 "physical_demand", "temporal_demand", "effort"]
cols_theoretical_NASA_tlx = ["theoretical_" + score for score in cols_NASA_tlx]
cols_time_NASA_tlx = ["time_start_NASA_tlx", "time_NASA_tlx_window"]


class Ikky_data():

    def __init__(self, reset=False, exclude_pilots=[2]):

        self.dic_tables = self.load_fichier_maitre_file(
            reset=reset, exclude_pilots=exclude_pilots)
        self.list_pilots = np.array([num for num in np.arange(
            2, 10) if num not in exclude_pilots]).astype(str)

        # print('dic tables = ', self.dic_tables)

    def load_fichier_maitre_file(self, exclude_pilots=[], reset=False):
        """
        Loads data from fichier maitre excel file
        returns : a dictionary that associates to each pilot a pandas table of events and features
        """
        name_file = "fichier_maitre_v3.xlsx"

        path_to_save = str(Path(savepath, name_file)).replace("xlsx", "pkl")

        if reset or not os.path.exists(path_to_save):

            dic_tables = pandas.read_excel(
                Path(datapath, name_file), sheet_name=None)
            dic_tables = self.prepare_dic_tables(dic_tables)

            dic_tables = self.add_infos_pilots(dic_tables)

            dic_tables = self.add_david_file_corresp(dic_tables)

            dic_tables = self.add_features(dic_tables, type_windows="NASA_tlx")

            dic_tables = self.add_features(
                dic_tables, type_windows="tc_fixed_windows")

            with open(path_to_save, "wb") as file:
                pickle.dump(dic_tables, file, protocol=pickle.HIGHEST_PROTOCOL)

        with open(path_to_save, "rb") as file:
            dic_tables = pickle.load(file)

        for num_pilot in exclude_pilots:
            del dic_tables[str(num_pilot)]

        dic_tables = self.add_normalized_scores(dic_tables)

        return dic_tables

    def load_david_file(self, num_pilot, num_scenario):

        path = Path(datapath, "david", "ClasseurPil" +
                    num_pilot+".xls", encoding="utf-8")

        tables_dic = pandas.read_excel(path, sheet_name=None)

        table = tables_dic["Pil"+num_pilot+"Sc"+num_scenario]

        time_start = pandas.to_datetime(
            table.iloc[0].values[0], format='%H:%M:%S')

        table.rename(columns={"Time\nFichier": "time_tc",
                              "Fenêtre analyse": "time_start_tc",
                              "Unnamed: 4": "time_end_tc",
                              "Durée": "duration",
                              "TC\nOral": "oral_tc",
                              "Evenement": "event"
                              }, inplace=True)

        time_cols = ["time_tc", "time_start_tc", "time_end_tc", "duration"]

        def convert_to_datetime(column):
            column = pandas.to_datetime(column, format='%H:%M:%S')

            column = column.apply(lambda d: d if not pandas.isnull(d) else d)

            return column

        table[time_cols] = table[time_cols].apply(convert_to_datetime)

        table["time_start_tc"] = table["time_start_tc"].apply(lambda d: (
            d-time_start).total_seconds() if not pandas.isnull(d) else d)
        table["duration"] = table["duration"].apply(
            lambda d: d.hour*3600 + d.minute*60 + d.second if not pandas.isnull(d) else d)

        table["time_tc_seconds"] = table["time_tc"].apply(
            lambda d: d.hour*3600 + d.minute*60 + d.second if not pandas.isnull(d) else d)

        table["oral_tc"] = table["oral_tc"].apply(lambda s: float(s) if s != "NOTC" and not "NO" in str(
            s) and s != "no" and s != "pas de réponse orale" else np.nan)

        return table

    def add_infos_pilots(self, dic_tables):

        dic_flight_hours = {"2": 2500,
                            "3": 2100,
                            "4": 2450,
                            "5": 1840,
                            "6": 300,
                            "7": 1200,
                            "8": 965,
                            "9": 1000}

        for pilot in dic_tables.keys():

            table_pilot = dic_tables[pilot]["sc2"]

            table_pilot["flight_hours"] = dic_flight_hours[pilot]

            dic_tables[pilot]["sc2"] = table_pilot

        return dic_tables

    def add_david_file_corresp(self, dic_tables):

        for pilot in dic_tables.keys():

            table_pilot = dic_tables[pilot]["sc2"]

            david_table = self.load_david_file(pilot, num_scenario="2")

            where_tc = ~(table_pilot["oral_tc"].isna())

            where_tc_david = ~(david_table["oral_tc"].isna())

            # #Verif

            print(pilot, where_tc.sum(), where_tc_david.sum())

            table_david_tc = david_table[where_tc_david].copy().reset_index()
            table_tc = table_pilot[where_tc].copy().reset_index()

            # for i in range(len(table_david_tc)):

            #     print(table_tc["time_tc"].iloc[i], table_tc["oral_tc"].iloc[i], table_david_tc["time_tc_seconds"].iloc[i], table_david_tc["oral_tc"].iloc[i])
            #     if table_tc["oral_tc"].iloc[i] != table_david_tc["oral_tc"].iloc[i]:
            #         print("WARNING")

            table_pilot["event"] = np.nan
            table_pilot.loc[where_tc,
                            "event"] = david_table.loc[where_tc_david, "event"].values

            table_pilot["time_start_tc_david"] = np.nan
            table_pilot.loc[where_tc,
                            "time_start_tc_david"] = david_table.loc[where_tc_david, "time_start_tc"].values

            table_pilot["duration_tc_david"] = np.nan
            table_pilot.loc[where_tc,
                            "duration_tc_david"] = david_table.loc[where_tc_david, "duration"].values

            table_pilot["time_end_tc_david"] = np.nan
            table_pilot.loc[where_tc, "time_end_tc_david"] = table_pilot.loc[where_tc,
                                                                             "time_start_tc_david"] + table_pilot.loc[where_tc, "duration_tc_david"]

            # si on choisit les fenêtres de David
            # table_pilot[where_tc, 'oral_tc'] = david_table.loc[where_tc_david, "oral_tc"].values

            dic_tables[pilot]["sc2"] = table_pilot

        return dic_tables

    def load_david_file_sc1_scores(self, pilot):

        path = Path(datapath, "david", "ClasseurPil" +
                    pilot+".xls", encoding="utf-8")

        tables_dic = pandas.read_excel(path, sheet_name=None)

        table = tables_dic["Pil"+pilot+"Sc1"]

        # where_tc_david = table["TC\nOral"].apply(lambda s: True if s.str.isnumeric(s) else False)

        where_tc_david = table["TC\nOral"].astype(str).apply(lambda f: True if (
            f != "nan" and f != "NOTC" and f != "pas de réponse orale" and f != "no" and "NO" not in f) else False)

        return table[where_tc_david]["TC\nOral"].astype(float).values

    def add_normalized_scores(self, dic_tables):

        for pilot in dic_tables.keys():

            values_sc1 = self.load_david_file_sc1_scores(pilot)

            table_pilot = dic_tables[pilot]["sc2"]

            table_pilot["normalized_oral_tc"] = table_pilot["oral_tc"] - \
                np.percentile(values_sc1, 75)

            table_pilot["binary_normalized_oral_tc"] = (
                table_pilot["oral_tc"] > np.percentile(values_sc1, 75)).astype(int)

            dic_tables[pilot]["sc2"] = table_pilot

        return dic_tables

    def add_features(self, dic_tables, type_windows):

        for pilot in dic_tables:
            print("reset features for pilot", pilot, type_windows)
            dic_data = {}

            normalization_features = self.load_normalization_features(pilot)

            for data_type in list_data:
                data_table = self.load_signals(
                    pilot=pilot, scenario="2", data=data_type)

                dic_data[data_type] = data_table

            table_pilot = dic_tables[pilot]["sc2"]

            for i in range(len(table_pilot)):

                if type_windows == "tc_david_windows":
                    window_start = table_pilot.iloc[i]["time_start_tc_david"]
                    window_end = table_pilot.iloc[i]["time_end_tc_david"]

                elif type_windows == "tc_fixed_windows":
                    window_start = table_pilot.iloc[i]["time_tc"] - 40
                    window_end = table_pilot.iloc[i]["time_tc"] + 10

                elif type_windows == "NASA_tlx":
                    window_start = table_pilot.iloc[i]["time_start_NASA_tlx"]
                    window_end = window_start + \
                        table_pilot.iloc[i]["time_NASA_tlx_window"]

                if (window_end - window_start) <= 10:
                    # 7 windows were skipped # 10 was chosen so that all features can be computed
                    print("skip because too short")
                    continue

                if np.isnan(window_start) or np.isnan(window_end):
                    continue

                dic_data_window = {}
                for data_type in dic_data:
                    data_table = dic_data[data_type]

                    if data_table is None:
                        continue
                    time_signal = data_table["reltime"]

                    index_start = np.where(time_signal >= window_start)[0][0]
                    index_stop = np.where(time_signal <= window_end)[0][-1]
                    dic_data_window[data_type] = data_table[index_start:index_stop+1]

                    if index_start == index_stop+1:
                        print(data_type)
                        print(data_table["reltime"].values)

                features_window = compute_all_features(dic_data_window)

                for key, value in features_window.items():
                    if key in list_features_to_normalize:
                        table_pilot.loc[i, "feature_"+type_windows +
                                        "_"+key] = value - normalization_features[key]
                    else:
                        table_pilot.loc[i, "feature_" +
                                        type_windows+"_"+key] = value

            dic_tables[pilot]["sc2"] = table_pilot

        return dic_tables

    def load_normalization_features(self, pilot):

        print("reset normalization features for pilot", pilot)

        dic_data = {}

        for data_type in list_data:
            data_table = self.load_signals(
                pilot=pilot, scenario="1", data=data_type)

            if data_table is None:
                continue
            dic_data[data_type] = data_table

        features = compute_all_features(dic_data)

        return features

    def get_theo_NASA_tlx_table(self, scenario="2"):

        if scenario != "2":
            raise ValueError("not available for scenario 1")

        events_dic = self.dic_tables

        for p, pilot in enumerate(events_dic):

            table = events_dic[pilot]["sc"+scenario]

            table["pilot"] = pilot

            all_data = ((~table[cols_theoretical_NASA_tlx +
                                cols_time_NASA_tlx].isna()).product(axis=1)).astype(bool)

            table_valid_values = table[all_data].copy()

            # print("count by pilot")
            # print(pilot, len(table_valid_values))

            if p == 0:
                concatenated_table = table_valid_values

            else:
                concatenated_table = pandas.concat(
                    [concatenated_table, table_valid_values], axis=0)

        # print("count by phase")
        # for phase in np.unique(concatenated_table["phase"]):
        #     print(phase, (concatenated_table["phase"]==phase).sum())

        concatenated_table.drop_duplicates(
            subset=cols_NASA_tlx+cols_time_NASA_tlx, inplace=True)

        return concatenated_table

    def get_NASA_tlx_table(self, scenario="2"):

        if scenario != "2":
            raise ValueError("not available for scenario 1")

        events_dic = self.dic_tables

        for p, pilot in enumerate(events_dic):

            table = events_dic[pilot]["sc"+scenario]

            table["pilot"] = pilot

            all_data = (
                (~table[cols_NASA_tlx+cols_time_NASA_tlx].isna()).product(axis=1)).astype(bool)

            table_valid_values = table[all_data].copy()

            # print("count by pilot")
            # print(pilot, len(table_valid_values))

            if p == 0:
                concatenated_table = table_valid_values

            else:
                concatenated_table = pandas.concat(
                    [concatenated_table, table_valid_values], axis=0)

        # print("count by phase")
        # for phase in np.unique(concatenated_table["phase"]):
        #     print(phase, (concatenated_table["phase"]==phase).sum())

        concatenated_table.drop_duplicates(
            subset=["time_start_NASA_tlx"], inplace=True)

        return concatenated_table

    def get_tc_table(self, scenario="2"):

        if scenario != "2":
            raise ValueError("not available for scenario 1")

        events_dic = self.dic_tables

        cols_tc = ["time_start_tc", "time_end_tc", "oral_tc"]

        for p, pilot in enumerate(events_dic):

            table = events_dic[pilot]["sc"+scenario]

            table["pilot"] = pilot

            all_data = ((~table[cols_tc].isna()).product(axis=1)).astype(bool)

            table_valid_values = table[all_data].copy()

            if p == 0:
                concatenated_table = table_valid_values

            else:
                concatenated_table = pandas.concat(
                    [concatenated_table, table_valid_values], axis=0)

            # print("pilot", pilot, table_valid_values["binary_normalized_oral_tc"].sum(
            # ), (table_valid_values["binary_normalized_oral_tc"] == 0).sum(), len(table_valid_values))

        # print("count by event")
        # for phase in np.unique(concatenated_table["event"]):
        #     print(phase, (concatenated_table["event"]==phase).sum())

        return concatenated_table

    def load_signals(self, pilot="3", scenario="1", data="st", reset=False,
                                                         plot_resampling=False):
        """
        Loads data for a particular sensor, in the form of a dataframe pandas
        pilot (str) : Pilot number
        scenario (str) : Scenario number
        data (str) : Data name (see the file TopicsDesk.json in the data directory)
        """

        file_path = Path(savepath, "data", data+"_"+pilot+"_"+scenario+".pkl")

        if reset or not os.path.exists(file_path):

            print("resetting signals table for pilot " +
                  pilot+", scenario "+scenario+", data "+data)

            num_file = str(3*(scenario == "1")+1*(scenario == "2"))

            for file_name in sorted(os.listdir(path_h5_data)):
                num_p = file_name[1]
                num_f = file_name.split(".")[0][-1]
                num_d = file_name[3]

                if num_p == pilot and num_f == num_file and num_d == scenario:

                    with h5py.File(Path(path_h5_data, file_name), "r") as fh5:

                        f = fh5[data]
                        datatypes = list(fh5.keys())
                        fkeys = list(f.keys())

                        ddict = {}
                        for key in fkeys:

                            if key == "datetime":
                                dt = pandas.to_datetime(f[key][:], unit='ms')
                            else:
                                dt = f[key][:]

                            ddict[key] = dt
                        df = pandas.DataFrame.from_dict(ddict)
                        df.drop(columns="datetime", inplace=True)

            if data == "sed":

                # Get eye movements

                eye_movements = process(df)

                df["eye_movements_horizontal"] = eye_movements["horizontal"]  # degree
                df["eye_movements_vertical"] = eye_movements["vertical"]  # degree

                # Preprocess blinks
                pupil = np.array(df["pupil"])
                time = np.array(df["reltime"])

                if np.sum(pupil) == 0:
                    df["pupil_without_blinks"] = np.nan
                    df["start_blinks"] = np.nan
                    df["end_blinks"] = np.nan
                    df["blinks_duration"] = np.nan

                else:

                    pupil_without_blinks = pupil.copy()
                    blinks = find_blinks(time, pupil)
                    start_blinks = np.zeros(len(pupil))
                    end_blinks = np.zeros(len(pupil))
                    blinks_duration = np.zeros(len(pupil))

                    for k in range(len(blinks)):
                        start_blink = blinks[k][0]
                        blink_duration = blinks[k][2]
                        end_blink = start_blink+blink_duration

                        pupil_without_blinks[start_blink:end_blink +
                                             1] = pupil[start_blink]

                        if end_blink - start_blink > 0:
                            start_blinks[start_blink] = 1
                            end_blinks[end_blink] = 1
                            blinks_duration[start_blink] = time[end_blink] - \
                                time[start_blink]

                    df["pupil_without_blinks"] = pupil_without_blinks
                    df["start_blinks"] = start_blinks
                    df["end_blinks"] = end_blinks
                    df["blinks_duration"] = blinks_duration

            if data == "aoi":

                aoi = df["aoi"]

                aoi_mapped_to_groups = aoi.apply(
                    lambda e: map_aoi_to_groups(e))
                df["aoi_mapped_to_groups"] = aoi_mapped_to_groups

            if data == "rc":

                time_s = np.array(df["reltime"])

                interp = interpolate.interp1d(
                    time_s, df["p2t_pilot"].values, kind="zero", fill_value="extrapolate")

                x = np.arange(0, int(time_s[-1])+1, 1)

                new_values = interp(x)

                df = pandas.DataFrame(np.concatenate(
                    [x.reshape(-1, 1), new_values.reshape(-1, 1)], axis=1), columns=["reltime", "p2t_pilot"])

            with open(file_path, "wb") as file:
                pickle.dump(df, file, protocol=pickle.HIGHEST_PROTOCOL)

        with open(file_path, "rb") as file:
            df = pickle.load(file)

        return df

    def print_data_frequency(self, data):

        df = self.load_signals(data=data)

        time_s = np.array(df["reltime"])

        timestamp_s = time_s[1:]-time_s[:-1]

        freqs = 1 / timestamp_s

        mean_timestamp = np.mean(timestamp_s)

        print(" ")
        print("sampling info for", data+":")
        print("mean timestamp", mean_timestamp, "std", np.std(timestamp_s))
        print("mean frequency", np.mean(freqs), "std", np.std(freqs))
        print(" ")

        # fig, ax = plt.subplots(1)
        # print(df)
        # ax.scatter(time_s, df["p2t_pilot"].values)

        # interp = interpolate.interp1d(time_s, df["p2t_pilot"], kind="zero", fill_value="extrapolate")

        # x = np.arange(0, int(time_s[-1])+1, 1)

        # new_values = interp(x)

        # ax.plot(x, new_values, color="red")

    def prepare_dic_tables(self, dic_tables):

        cols_theoretical_NASA_tlx = [
            "mental_demand", "physical_demand", "temporal_demand", "effort"]
        cols_theoretical_NASA_tlx = [
            "theoretical_"+col for col in cols_theoretical_NASA_tlx]

        cols_NASA_tlx = ["mental_demand", "physical_demand",
                         "temporal_demand", "effort", "performance", "frustration"]

        for key in dic_tables:

            dic_tables[key] = dic_tables[key].rename(columns={"time_end_s": "time_end_tc",
                                                              "time_start": "time_start_tc",
                                                              "time_test_charge_s": "time_tc",
                                                              "oral_test_charge": "oral_tc",
                                                              "slider_test_charge": "slider_tc",
                                                              "time_start_NASA_TLX_s": "time_start_NASA_tlx",
                                                              "time_NASA_TLX_window_s": "time_NASA_tlx_window",
                                                              "theorical_mental_demand": "theoretical_mental_demand",
                                                              "theorical_temporal_demand": "theoretical_temporal_demand",
                                                              "theorical_effort": "theoretical_effort",
                                                              "theorical_physical_demand": "theoretical_physical_demand",
                                                              "Phase": "phase",
                                                              "total": "mean_NASA_tlx"})

            dic_tables[key]["mean_theoretical_NASA_tlx"] = dic_tables[key][cols_theoretical_NASA_tlx].mean(
                axis=1)

        original_keys = list(dic_tables.keys())

        for key in original_keys:

            for sc in ["1", "2"]:
                if "Sc"+sc in key:

                    if key.replace("Pil", "").replace("Sc"+sc, "") not in dic_tables:

                        dic_tables[key.replace(
                            "Pil", "").replace("Sc"+sc, "")] = {}

                    dic_tables[key.replace("Pil", "").replace(
                        "Sc"+sc, "")]["sc"+sc] = dic_tables[key]

            del dic_tables[key]

        return dic_tables


if __name__ == "__main__":

    db = Ikky_data(reset=False, exclude_pilots=[2])

    # db.load_signals(data="sed", reset=True)

    db.get_tc_table()

    # NASA_tlx_table = db.get_NASA_tlx_table()

    # theo_NASA_tlx_table = db.get_theo_NASA_tlx_table()

    db.print_data_frequency("hr")
    db.print_data_frequency("br")
    db.print_data_frequency("ibi")
    db.print_data_frequency("sed")
    db.print_data_frequency("aoi")
    db.print_data_frequency("am")
    db.print_data_frequency("fc")
    db.print_data_frequency("rc")

    # sed = db.load_signals(data="sed", pilot="3", scenario="2", reset=False)
    # aoi = db.load_signals(data="aoi", pilot="3", scenario="2", reset=False)

    # sed = db.load_signals(data="sed", pilot="5", scenario="2")

    # br = db.load_signals(data="br", pilot="5", scenario="2")
