import collections
import datetime
import os
import xml.etree.ElementTree

import pelops.datasets.chip as chip
import pelops.utils as utils

# ================================================================================
#  Veri Dataset
# ================================================================================


class VeriDataset(chip.ChipDataset):
    filenames = collections.namedtuple(
        "filenames",
        [
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
            "label_train"
        ]
    )
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
        "train_label.xml"
    )

    def __init__(self, dataset_path, set_type=None):
        super().__init__(dataset_path, set_type)
        self.__set_filepaths()  # set self.__filepaths
        self.__color_type = {}
        if self.set_type is utils.SetType.ALL or self.set_type is utils.SetType.TRAIN:
            self.__build_metadata_dict()
        self.__set_chips()

    def __build_metadata_dict(self):
        """Extract car type and color from the label file."""
        root = xml.etree.ElementTree.parse(self.__filepaths.label_train).getroot()

        colors = {
            0: None, 1: "yellow", 2: "orange", 3: "green", 4: "gray", 5: "red",
            6: "blue", 7: "white", 8: "golden", 9: "brown", 10: "black",
        }
        types = {
            0: None, 1: "sedan", 2: "suv", 3: "van", 4: "hatchback", 5: "mpv",
            6: "pickup", 7: "bus", 8: "truck", 9: "estate",
        }

        self.__color_type = {}
        for child in root.iter("Item"):
            # Get the IDs from the XML node
            vehicle_id = child.attrib["vehicleID"]
            color = child.attrib["colorID"]
            body_type = child.attrib["typeID"]

            color_id = int(color)
            body_id = int(body_type)
            str_color = colors[color_id]
            str_body = types[body_id]

            self.__color_type[vehicle_id] = (str_color, str_body)

    def __set_filepaths(self):
        self.__filepaths = VeriDataset.filenames(
            os.path.join(self.dataset_path, VeriDataset.filepaths.name_query),
            os.path.join(self.dataset_path, VeriDataset.filepaths.name_test),
            os.path.join(self.dataset_path, VeriDataset.filepaths.name_train),
            os.path.join(self.dataset_path, VeriDataset.filepaths.dir_query),
            os.path.join(self.dataset_path, VeriDataset.filepaths.dir_test),
            os.path.join(self.dataset_path, VeriDataset.filepaths.dir_train),
            os.path.join(self.dataset_path, VeriDataset.filepaths.list_color),
            os.path.join(self.dataset_path, VeriDataset.filepaths.list_type),
            os.path.join(self.dataset_path, VeriDataset.filepaths.ground_truths),
            os.path.join(self.dataset_path, VeriDataset.filepaths.junk_images),
            os.path.join(self.dataset_path, VeriDataset.filepaths.label_train),
        )

    def __set_chips(self):
        # TODO: ignore images labeled as query, so we do not have to keep tabs for identical chips
        # identify all the chips
        all_names_filepaths = {
            utils.SetType.ALL: [self.__filepaths.name_query, self.__filepaths.name_test, self.__filepaths.name_train],
            utils.SetType.QUERY: [self.__filepaths.name_query],
            utils.SetType.TEST: [self.__filepaths.name_test],
            utils.SetType.TRAIN: [self.__filepaths.name_train],
        }.get(self.set_type)
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

        filepath = os.path.join(img_dir, img_name)
        car_id = int(splitter[0])
        cam_id = int(utils.get_numeric(splitter[1]))
        time = datetime.datetime.fromtimestamp(int(splitter[2]))
        misc["binary"] = int(os.path.splitext(splitter[3])[0])

        color, vehicle_type = self.__color_type.get(car_id, (None, None))
        misc["color"] = color
        misc["vehicle_type"] = vehicle_type

        return chip.Chip(filepath, car_id, cam_id, time, misc)
