import collections
import json
import os.path

import pelops.datasets.chip as chip
import pelops.utils as utils


class DGCarsDataset(chip.ChipDataset):
    filenames = collections.namedtuple(
        "filenames",
        [
            "all_list",
            "train_list",
            "test_list",
        ]
    )
    filepaths = filenames(
        "allFiles",
        "training",
        "testing",
    )

    def __init__(self, dataset_path, set_type=utils.SetType.ALL.value):
        super().__init__(dataset_path, set_type)
        self.__set_filepaths()         # set self.__filepaths
        self.__set_chips()

    def __set_filepaths(self):
        self.__filepaths = self.filenames(
            self.dataset_path + "/" + DGCarsDataset.filepaths.all_list,
            self.dataset_path + "/" + DGCarsDataset.filepaths.train_list,
            self.dataset_path + "/" + DGCarsDataset.filepaths.test_list,
        )

    def __set_chips(self):
        # identify all the chips, default query to all
        name_filepath = {
            utils.SetType.ALL.value: self.__filepaths.all_list,
            utils.SetType.TEST.value: self.__filepaths.test_list,
            utils.SetType.TRAIN.value: self.__filepaths.train_list,
        }.get(self.set_type, self.__filepaths.all_list)

        # create chip objects based on the names listed in the files
        for dg_chip in utils.read_json(name_filepath):
            filepath = os.path.join(self.dataset_path, dg_chip["filename"])
            car_id = None
            cam_id = None
            time = None
            misc = dg_chip
            current_chip = chip.Chip(filepath, car_id, cam_id, time, misc)

            self.chips[filepath] = current_chip
