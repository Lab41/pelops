import collections
import os
import scipy.io

import pelops.datasets.chip as chip
import pelops.utils as utils


class CompcarDataset(chip.ChipDataset):
    filenames = collections.namedtuple(
        "filenames",
        [
            "image_dir",
            "name_train",
            "name_test",
            "model_mat",
            "color_mat",
        ]
    )
    filepaths = filenames (
        "image",
        "train_surveillance.txt",
        "test_surveillance.txt",
        "sv_make_model_name.mat",
        "color_list.mat",
    )

    def __init__(self, dataset_path, set_type=None):
        super().__init__(dataset_path, set_type)
        self.__set_filepaths()         # set self.__filepaths
        self.__extract_color_labels()  # set self.__color_map
        self.__extract_model_labels()  # set self.__model_map
        self.__set_chips()

    def __set_filepaths(self):
        self.__filepaths = self.filenames(
            self.dataset_path + "/" + CompcarDataset.filepaths.image_dir,
            self.dataset_path + "/" + CompcarDataset.filepaths.name_train,
            self.dataset_path + "/" + CompcarDataset.filepaths.name_test,
            self.dataset_path + "/" + CompcarDataset.filepaths.model_mat,
            self.dataset_path + "/" + CompcarDataset.filepaths.color_mat,
        )

    def __extract_color_labels(self):
        self.__color_map = {}

        # Map color_id to its respective name
        color_map = {
            -1: None,
            0: "black",
            1: "white",
            2: "red",
            3: "yellow",
            4: "blue",
            5: "green",
            6: "purple",
            7: "brown",
            8: "champagne",
            9: "silver",
        }

        # Load the matrix of colors
        color_matrix = scipy.io.loadmat(
            self.__filepaths.color_mat)["color_list"]

        # File is an length 1 array, color_num is a 1x1 matrix
        for file_array, color_num_matrix in color_matrix:
            filepath = file_array[0]
            color_num = int(color_num_matrix[0][0])
            self.__color_map[filepath] = color_map[color_num]

    def __extract_model_labels(self):
        self.__model_map = {}

        model_matrix = scipy.io.loadmat(
            self.__filepaths.model_mat)["sv_make_model_name"]
        for car_id, model_matrix in enumerate(model_matrix):
            # correct car_id
            car_id = int(car_id) + 1
            # make contains only the make of the car and occasionally contains whitespaces after
            make = model_matrix[0][0].strip()
            # correct instance when make is misspelled that affects the model
            if make == "Zoyte":
                make = "Zotye"
            # model sometimes contains both make and model, so ensure that model only contains model
            make_and_model = model_matrix[1][0]
            model = make_and_model.replace(make, "").strip()
            # model_id contains the model id used in the web
            model_id = int(model_matrix[2][0][0])
            # correct instance when make is misspelled
            if make == "BWM":
                make = "BMW"
            self.__model_map[car_id] = [make, model, model_id]

    def __set_chips(self):
        # identify all the chips, default query to all
        all_names_filepaths = {
            utils.SetType.ALL: [self.__filepaths.name_test, self.__filepaths.name_train],
            utils.SetType.TEST: [self.__filepaths.name_test],
            utils.SetType.TRAIN: [self.__filepaths.name_train],
        }.get(self.set_type, [self.__filepaths.name_test, self.__filepaths.name_train])
        # create chip objects based on the names listed in the files
        for name_filepath in all_names_filepaths:
            for name in open(name_filepath):
                current_chip = self.__create_chip(self.__filepaths.image_dir, name.strip())
                self.chips[current_chip.filepath] = current_chip

    def __create_chip(self, img_dir, img_name):
        splitter = img_name.split("/")
        misc = dict()

        filepath = img_dir + "/" + img_name
        car_id = int(splitter[0])
        cam_id = None
        time = None
        misc["color"] = self.__color_map[img_name]
        make, model, model_id = self.__model_map[car_id]
        misc["make"] = make
        misc["model"] = model
        misc["model_id"] = model_id

        return chip.Chip(filepath, car_id, cam_id, time, misc)
