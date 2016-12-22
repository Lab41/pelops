import collections
import datetime
import os

import pelops.datasets.chip as chip
import pelops.utils as utils

# ================================================================================
#  Veri Dataset
# ================================================================================


class VeriDataset(chip.ChipDataset):
    filenames = collections.namedtuple("filenames", [
        "name_query",
        "name_test",
        "name_train",
        "dir_query",
        "dir_test",
        "dir_train",
        "list_color",
        "list_type",
        "ground_truths",
        "junk_images",
        "label_train"])
    filepaths = filenames(
        "name_query.txt",
        "name_test.txt",
        "name_train.txt",
        "image_query",
        "image_test",
        "image_train",
        "list_color.txt", 
        "list_type.txt",
        "gt_image.txt",
        "jk_image.txt",
        "train_label.txt")

    def __init__(self, dataset_path, set_type=utils.SetType.ALL.value):
        super().__init__(dataset_path, set_type)
        self.__set_filepaths()
        self.__set_chips()

    def __set_filepaths(self):
        self.__filepaths =  VeriDataset.filenames(
            self.dataset_path + "/" + VeriDataset.filepaths.name_query,
            self.dataset_path + "/" + VeriDataset.filepaths.name_test,
            self.dataset_path + "/" + VeriDataset.filepaths.name_train,
            self.dataset_path + "/" + VeriDataset.filepaths.dir_query,
            self.dataset_path + "/" + VeriDataset.filepaths.dir_test, 
            self.dataset_path + "/" + VeriDataset.filepaths.dir_train,
            self.dataset_path + "/" + VeriDataset.filepaths.list_color, 
            self.dataset_path + "/" + VeriDataset.filepaths.list_type,
            self.dataset_path + "/" + VeriDataset.filepaths.ground_truths,
            self.dataset_path + "/" + VeriDataset.filepaths.junk_images, 
            self.dataset_path + "/" + VeriDataset.filepaths.label_train)

    def __set_chips(self):
        # TODO: ignore images labeled as query, so we do not have to keep tabs for identical chips
        # identify all the chips
        all_names_filepaths = {
            utils.SetType.ALL.value: [self.__filepaths.name_query, self.__filepaths.name_test, self.__filepaths.name_train],
            utils.SetType.QUERY.value: [self.__filepaths.name_query],
            utils.SetType.TEST.value: [self.__filepaths.name_test],
            utils.SetType.TRAIN.value: [self.__filepaths.name_train],
        }.get(self.set_type)
        print("all_names_filepaths: {}".format(all_names_filepaths))
        # create chip objects based on the names listed in the files
        for name_filepath in all_names_filepaths:
            if VeriDataset.filepaths.name_query in name_filepath:
                img_dir = self.__filepaths.dir_query
            elif VeriDataset.filepaths.name_test in name_filepath:
                img_dir = self.__filepaths.dir_test
            else: # VeriDataset.filepaths.name_train in filepath
                img_dir = self.__filepaths.dir_train
            for name in open(name_filepath):
                current_chip = self.__create_chip(img_dir, name.strip())
                self.chips[current_chip.filepath] = current_chip

    def __create_chip(self, img_dir, img_name):
        # information about the chip resides in the chip's name
        splitter = img_name.split("_")
        misc = dict()

        filepath = img_dir + "/" + img_name
        car_id = int(splitter[0])
        cam_id = int(utils.get_numeric(splitter[1]))
        time = datetime.datetime.fromtimestamp(int(splitter[2]))
        misc["binary"] = int(os.path.splitext(splitter[3])[0])

        return chip.Chip(filepath, car_id, cam_id, time, misc)
