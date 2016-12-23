import collections
import os

import scipy.io

import pelops.datasets.chip as chip


class CompCarDataset(chip.ChipDataset):

    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        # define paths to files and directories
        self.filenames = collections.namedtuple(
            "filenames",
            [
                "image_dir",
                "train_txt",
                "test_txt",
                "make_model_mat",
                "color_mat",
            ]
        )
        self.filepaths = filenames(
            "iamge",
            "train_surveillance.txt",
            "test_surveillance.txt",
            "sv_make_model_name.mat",
            "color_list.mat",
        )
        self.__set_filepaths()
        self.__extract_color_labels()
        self.__set_chips()

    def __set_filepaths(self):
        self.__filepaths = self.filenames(
            self.dataset_path + "/" + self.filepaths.image_dir,
            self.dataset_path + "/" + self.filepaths.train_txt,
            self.dataset_path + "/" + self.filepaths.test_txt,
            self.dataset_path + "/" + self.filepaths.make_model_mat,
            self.dataset_path + "/" + self.filepaths.color_mat,
        )

    def __extract_color_labels(self):
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
        self.__car_color_map = {}
        color_matrix = scipy.io.loadmat(
            self.__filepaths.color_mat)["color_list"]

        # File is an length 1 array, color_num is a 1x1 matrix
        for file_array, color_num_matrix in color_matrix:
            file = file_array[0]
            color_num = color_num_matrix[0][0]
            self.__car_color_map[file] = color_map[color_num]

    def __set_chips(self):
        directory = self.__filepaths.image_dir
        for file in os.listdir(directory):
            path = directory + '/' + file

            # Only interested in certain files
            is_valid = os.path.isfile(path)
            is_png = path.endswith(".png")
            is_mask = "mask" in path
            if not is_valid or not is_png or is_mask:
                continue

            # Set all Chip variables
            car_id = get_sa_car_id(path)
            cam_id = get_sa_cam_id(path)

            time = cam_id  # Cars always pass the first camera first
            misc = None    # No miscellaneous data

            # Make chip
            current_chip = chip.Chip(
                path,
                car_id,
                cam_id,
                time,
                misc
            )

            self.chips[path] = current_chip
