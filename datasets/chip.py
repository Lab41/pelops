""" Base class for the chips
"""

import abc

# ================================================================================
#  Chip Factory
# ================================================================================


class ChipFactory(object):
    factories = {}

    @staticmethod
    def add_factory(factory_id, chip_factory):
        ChipFactory.factories.put[factory_id] = chip_factory

    @staticmethod
    def create_dataset(factory_id):
        if not ChipFactory.factories.has_key(factory_id):
            ChipFactory.factories[factory_id] = eval(factory_id + ".Factory()")
        return ChipFactory.factories[factory_id].create()


# ================================================================================
#  Chip Dataset
# ================================================================================


class ChipDataset(__metaclass__ = abc.ABCMeta):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
    
    def get_all_chips_by_car_id(self, car_id):
        return

    def get_all_chips_by_camera_id(self, camera_id):
        return

# ================================================================================
#  Chip Base
# ================================================================================


class ChipBase(__metaclass__=abc.ABCMeta):
    # handle misc arguments
    def __init__(self, filepath, car_id, camera_id, timestamp):
        self.filepath = filepath
        self.car_id = car_id
        self.camera_id = camera_id
        self.timestamp = timestamp


