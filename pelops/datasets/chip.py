""" Base class for the chips
"""

import abc
import collections
import os

# ================================================================================
#  Chip Factory
# ================================================================================


class DatasetFactory(object):
    @staticmethod
    def create_dataset(dataset_type, dataset_path):
        for cls in ChipDataset.__subclasses__():
            if cls.check_dataset_type(dataset_type):
                return cls(dataset_path)

# ================================================================================
#  Chip Dataset
# ================================================================================


class ChipDataset(metaclass = abc.ABCMeta):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.chips = dict()

    @classmethod
    def check_dataset_type(self, dataset_type):
        return dataset_type == self.__name__

    def get_all_chips_by_car_id(self, car_id):
        return [chip for chip in self.chips.values() if chip.car_id == car_id]

    def get_all_chips_by_camera_id(self, camera_id):
        return [chip for chip in self.chips.values() if chip.cam_id == cam_id]

    def __iter__(self):
        for chip in self.chips.values():
            yield chip
        raise StopIteration()

    def __len__(self):
        return len(self.chips)

# ================================================================================
#  Chip Base
# ================================================================================


class Chip = collections.namedtuple("Chip", 
    ["filepath",
     "car_id", 
     "cam_id", 
     "time",
     "misc"])