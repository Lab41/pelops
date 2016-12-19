from os import listdir
from os.path import isfile

from chipbase import Chip
from chipbase import ChipBase


class STR_SA(ChipBase):

    def __init__(self, directory, dataset_name="STR_SA", *args, **kwargs):
        super().__init__(dataset_name, args, kwargs)

        # Get some chips
        self.__load_chips(directory)

    def __load_chips(self, directory):
        for file in listdir(directory):
            path = directory + '/' + file

            # Only interested in certain files
            is_valid = isfile(path)
            is_png = path.endswith(".png")
            is_mask = "mask" in path
            if not is_valid or not is_png or is_mask:
                continue

            # Set all Chip variables
            car_id = get_sa_car_id(path)
            cam_id = get_sa_cam_id(path)
            chip_id = str(car_id) + "_" + str(cam_id)
            time = cam_id  # Cars always pass the first camera first
            misc = None  # No misc data

            # Make chip
            chip = Chip(
                chip_id,
                car_id,
                cam_id,
                time,
                path,
                misc,
            )

            self.chips[chip_id] = chip


def int_from_string(string, start_chars, int_len):
    loc = string.find(start_chars)

    # Not found
    if loc < 0:
        return None

    start = loc + len(start_chars)
    end = start + int_len
    str_num = string[start:end]
    return int(str_num)


def get_sa_cam_id(string):
    return int_from_string(string, start_chars="_cam", int_len=2)


def get_sa_car_id(string):
    return int_from_string(string, start_chars="match", int_len=5)
