
from .paths import datapath, savepath

from pathlib import Path
import pickle
import os
import pandas
import numpy as np


class Ikky_data():

    def __init__(self, reset=False, exclude_pilots=[2]):

        self.dic_tables = self.load_fichier_maitre_file(
            reset=reset, exclude_pilots=exclude_pilots)
        self.list_pilots = np.array([num for num in np.arange(
            2, 10) if num not in exclude_pilots]).astype(str)

    def load_fichier_maitre_file(self, exclude_pilots=[], reset=False):
        """
        Loads data from fichier maitre excel file
        returns : a dictionary that associates to each pilot a pandas table of events and features
        """
        name_file = "fichier_maitre_v3.xlsx"

        path_to_save = str(Path(savepath, name_file)).replace("xlsx", "pkl")

        if reset or not os.path.exists(path_to_save):
            raise RuntimeError('See old folder.')

        with open(path_to_save, "rb") as file:
            dic_tables = pickle.load(file)

        for num_pilot in exclude_pilots:
            del dic_tables[str(num_pilot)]

        dic_tables = self.add_normalized_scores(dic_tables)

        return dic_tables

    def load_david_file_sc1_scores(self, pilot):

        path = Path(savepath, "david", "ClasseurPil" +
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
