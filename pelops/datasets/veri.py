# TODO: extract from .zip

import chip
import collections
import datetime
import os

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
        "gt_image.txt",
        "jk_image.txt",
        "train_label.txt")

    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.__set_filepaths()
        self.chips = self.get_all_chips()

    def __set_filepaths(self):
        self.__filepaths =  VeriDataset.filenames(
            self.dataset_path + "/" + VeriDataset.filepaths.name_query,
            self.dataset_path + "/" + VeriDataset.filepaths.name_test,
            self.dataset_path + "/" + VeriDataset.filepaths.name_train,
            self.dataset_path + "/" + VeriDataset.filepaths.dir_query,
            self.dataset_path + "/" + VeriDataset.filepaths.dir_test, 
            self.dataset_path + "/" + VeriDataset.filepaths.dir_train,
            self.dataset_path + "/" + VeriDataset.filepaths.ground_truths,
            self.dataset_path + "/" + VeriDataset.filepaths.junk_images, 
            self.dataset_path + "/" + VeriDataset.filepaths.label_train)

    def get_all_chips(self):
        # identify all the chips
        all_names_filepaths = [self.__filepaths.name_query, self.__filepaths.name_test, self.__filepaths.name_train]
        # remove identical chips by using set
        chips = set()
        # create chip objects based on the names listed in the files
        for name_filepath in all_names_filepaths:
            if VeriDataset.filepaths.name_query in name_filepath:
                img_dir = self.__filepaths.dir_query
            elif VeriDataset.filepaths.name_test in name_filepath:
                img_dir = self.__filepaths.dir_test
            else: # VeriDataset.filepaths.name_train in filepath
                img_dir = self.__filepaths.dir_train
            # put all the chip objects in a set
            this_set = set(self.__create_chip(img_dir, name.strip()) for name in open(name_filepath))
            # combine with the previous list of chip objects
            chips = chips.union(this_set)
        # return a list, not a set    
        return list(chips)

    def __create_chip(self, img_dir, img_name):
        # information about the chip resides in the chip's name
        splitter = img_name.split("_")

        filepath = img_dir + "/" + img_name
        car_id = int(splitter[0])
        camera_id = int(utils.get_numeric(splitter[1]))
        timestamp = datetime.datetime.fromtimestamp(int(splitter[2]))
        binary = int(os.path.splitext(splitter[3])[0])

        chip = VeriChip(filepath, car_id, camera_id, timestamp, binary)
        return chip

# ================================================================================
#  Veri Chip
# ================================================================================


class VeriChip(chip.ChipBase):
    def __init__(self, filepath, car_id, camera_id, timestamp, binary):
        super().__init__(filepath, car_id, camera_id, timestamp)
        self.binary = binary

    def get_timestamp(self):
        # Year-Month-Date Hour:Minute:Second
        return self.timestamp.strftime("%Y-%m-%d %H:%M:%S")



