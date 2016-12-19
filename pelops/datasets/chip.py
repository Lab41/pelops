""" Base class for the chips
"""

import abc
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
        self.chips = list()

    @classmethod
    def check_dataset_type(self, dataset_type):
        return dataset_type == self.__name__

    @abc.abstractmethod
    def get_all_chips(self):
        pass

    def get_all_chips_by_car_id(self, car_id):
        return [chip for chip in self.chips if chip.car_id == car_id]

    def get_all_chips_by_camera_id(self, camera_id):
        return [chip for chip in self.chips if chip.camera_id == camera_id]

    def __iter__(self):
        for chip in self.chips:
            yield chip
        raise StopIteration()

    def __len__(self):
        return len(self.chips)

# ================================================================================
#  Chip Base
# ================================================================================


class ChipBase(metaclass = abc.ABCMeta):
    # TODO: handle misc arguments
    def __init__(self, filepath, car_id, camera_id, timestamp):
        self.name = os.path.basename(os.path.normpath(filepath))
        self.filepath = filepath
        self.car_id = car_id
        self.camera_id = camera_id
        self.timestamp = timestamp

    @classmethod
    def check_chip_type(self, chip_type):
        return chip_type == self.__name__

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        # check that dataset_type, car_id, camera_id, and timestamp are the same
        return (type(self) == type(other)) and \
               (self.car_id == other.car_id) and \
               (self.camera_id == other.camera_id) and \
               (self.timestamp == other.timestamp)

