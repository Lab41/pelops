import abc
import collections
import os
import pelops.utils as utils

# ================================================================================
#  Chip Factory
# ================================================================================


class DatasetFactory(object):
    @staticmethod
    def create_dataset(dataset_type, dataset_path, set_type=utils.SetType.ALL.value):
        for cls in ChipDataset.__subclasses__():
            if cls.check_dataset_type(dataset_type):
                return cls(dataset_path, set_type)

# ================================================================================
#  Chip Dataset
# ================================================================================


class ChipDataset(metaclass = abc.ABCMeta):
    def __init__(self, dataset_path, set_type=utils.SetType.ALL.value):
        self.dataset_path = dataset_path
        self.set_type = set_type
        self.chips = dict()
        self.chips_by_cam_id = None
        self.chips_by_car_id = None

    @classmethod
    def check_dataset_type(self, dataset_type):
        return dataset_type == self.__name__

    def get_all_chips_by_car_id(self, car_id):
        if self.chips_by_car_id is None:
            self.chips_by_car_id = collections.defaultdict(list)
            for chip_key, chip in self.chips.items():
                self.chips_by_car_id[chip.car_id].append(chip_key)
        return [self.chips[chip_key] for chip_key in self.chips_by_car_id[car_id]]        
    
    def get_all_chips_by_car_id_camera_id(self, car_id, cam_id):
        output = []
        for chip in self.get_all_chips_by_car_id(car_id):
            if chip.cam_id == cam_id:
                output.append(chip)
        return output

    def get_all_chips_by_cam_id(self, cam_id):
        if self.chips_by_cam_id is None:
            self.chips_by_cam_id = collections.defaultdict(list)
            for chip_key, chip in self.chips.items():
                self.chips_by_cam_id[chip.cam_id].append(chip_key)
                
        return [self.chips[chip_key] for chip_key in self.chips_by_cam_id[cam_id]]

    def get_distinct_cams_by_car_id(self, car_id):
        # TODO: Look at performance
        return self.get_distinct_cams_per_car()[car_id]

    def get_distinct_cams_per_car(self):
        # TODO: Look at performance
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