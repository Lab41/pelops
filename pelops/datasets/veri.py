# TODO: extract from .zip

import chip
import collections

# ================================================================================
#  Veri Dataset
# ================================================================================


class VeriDataset(chip.chipDataset):
    filenames = collections.namedtuple("filenames", [
        "name_query", "name_test", "name_train", 
        "dir_query", "dir_test", "dir_train"])
    filepaths = filenames(
        "name_query.txt", "name_test.txt", "name_train.txt", 
        "image_query", "image_test", "image_train")

    class Factory:
        def create_dataset(self):
            return VeriDataset()

    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.__reset_filepaths()

    def __reset_filepaths(self):
        self.__filepaths = VeriDataset.filenames(
            self.dataset_path + "/" + VeriDataset.filepaths.name_query,
            self.dataset_path + "/" + VeriDataset.filepaths.name_test,
            self.dataset_path + "/" + VeriDataset.filepaths.name_train,
            self.dataset_path + "/" + VeriDataset.filepaths.dir_query,
            self.dataset_path + "/" + VeriDataset.filepaths.dir_test, 
            self.dataset_path + "/" + VeriDataset.filepaths.dir_train)

    def get_all_chips_by_car_id(self, car_id):
        return

    def get_all_chips_by_camera_id(self, camera_id):
        return

# ================================================================================
#  Veri Chip
# ================================================================================


class VeriChip(chip.chipBase):
    def __init__(self, filepath, car_id, camera_id, timestamp):
        super().__init__(filepath, car_id, camera_id, timestamp)

# ================================================================================
#  Veri Filepaths
# ================================================================================


