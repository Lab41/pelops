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

    def get_all_chips_by_cam_id(self, cam_id):
        return [chip for chip in self.chips.values() if chip.cam_id == cam_id]

    def get_distinct_cams_by_car_id(self, car_id):
        return get_distinct_cams_per_car()[car_id]

    def get_distinct_cams_per_car(self):
        list_of_cameras_per_car = collections.defaultdict(set)
        for chip in self.chips.values():
            list_of_cameras_per_car[chip.car_id].add(chip.cam_id)
        return list_of_cameras_per_car

    def get_all_cam_ids(self):
        return list(set(chip.cam_id for chip in self.chips.values()))

    def get_all_car_ids(self):
        return list(set(chip.car_id for chip in self.chips.values()))

    def __iter__(self):
        for chip in self.chips.values():
            yield chip
        raise StopIteration()

    def __len__(self):
        return len(self.chips)

# ================================================================================
#  Chip Base
# ================================================================================


# chip_id is the filepath
Chip = collections.namedtuple("Chip", 
    ["filepath",
     "car_id", 
     "cam_id", 
     "time",
     "misc"])