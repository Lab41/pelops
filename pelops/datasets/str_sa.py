import chip
import collections
import os

import pelops.utils as utils

# ================================================================================
#  STR_SA Dataset
# ================================================================================


class StrDataset(chip.ChipDataset):
    filenames = collections.namedtuple("filenames", [
        "dir_all"])
    filepaths = filenames (
        "crossCameraMatches")

    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.__set_filepaths()
        self.chips = self.get_all_chips()

    def __set_filepaths(self):
        self.__filepaths = StrDataset.filenames(
            self.dataset_path + "/" + StrDataset.filepaths.dir_all)

    def get_all_chips(self):
        chips = list()
        for file in os.listdir(self.__filepaths.dir_all):
            path = self.__filepaths.dir_all + '/' + file

            # Only interested in certain files
            is_valid = os.path.isfile(path)
            is_png = path.endswith(".png")
            is_mask = "mask" in path
            if not is_valid or not is_png or is_mask:
                continue

            # Set all Chip variables
            car_id = get_sa_car_id(path)
            cam_id = get_sa_cam_id(path)

            time = cam_id  # Cars always pass the first camera first
            chip_id = str(car_id) + "_" + str(cam_id) + "_" + str(time)
            misc = None  # No miscellaneous data

            # Make chip
            chip = StrChip(
                path,
                car_id,
                cam_id,
                time,
            )

            #chips[chip_id] = chip
            chips.append(chip)
        return chips


def int_from_string(string, start_chars, int_len):
    # We only want to use the filename, not the directory names
    base_string = os.path.basename(string)
    loc = base_string.find(start_chars)

    # Not found
    if loc < 0:
        return None

    start = loc + len(start_chars)
    end = start + int_len
    str_num = base_string[start:end]
    return int(str_num)


def get_sa_cam_id(string):
    return int_from_string(string, start_chars="_cam", int_len=2)


def get_sa_car_id(string):
    return int_from_string(string, start_chars="match", int_len=5)

# ================================================================================
#  STR_SA Chip
# ================================================================================

class StrChip(chip.ChipBase):
    def __init__(self, filepath, car_id, camera_id, timestamp):
        super().__init__(filepath, car_id, camera_id, timestamp)
