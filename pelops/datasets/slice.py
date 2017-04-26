import csv
import io
import itertools
import os
import re
import sys
from datetime import datetime

import pelops.datasets.chip as chip

# ================================================================================
#  SLiCE Test Dataset (labeled by STR)
# ================================================================================


class SliceDataset(chip.ChipDataset):

    def __init__(self, dataset_path, set_type=None, debug=False):
        super().__init__(dataset_path, set_type)
        self.__noise_seq = 0
        self.__debug = debug
        self.__set_chips()

    @staticmethod
    def __decode_truth_file(truth_file):
        """The labels for the STR processed SLiCE chips are in a 'truth.txt' file which this function parses."""

        with open(truth_file) as truth_hdl:
            truth_text = truth_hdl.read()
            for char in [' ', '%']:
                truth_text = truth_text.replace(char, '')
            truth_fobj = io.StringIO(truth_text)
            return {(int(dct['obSetIdx']), int(dct['chipIdx'])): int(dct['targetID'])
                    for dct in csv.DictReader(truth_fobj)}

    @staticmethod
    def index_chip(file_path):
        """Parses an arbitrary file path and identifies paths of valid image chips.
        Returns None for non-chip file paths."""

        # We have to handle two cases:
        #
        # 1) The STR San Antonio DOT chips, which have the form:
        #     ObSet009_1473015765_IH37_Jones/images/ObSet009-014.png
        #
        # 2) The SLICE chips, which have the form:
        #     ObSet101_1473082429_day5_camera3/images/ObSet101-001-0-20160905_185543.375_1.jpg
        #
        # The epoch on the SLICE chips is per chip, whereas it is per
        # observation set for the STR chips. The SLICE chip file names have the
        # follow information after the ObSet and chip id:
        #
        # Obset-ChipID-label-time_unused

        # Split the file path into pieces to extract the information from it
        file_path = os.path.normpath(file_path)
        directory, _, file = file_path.split(os.sep)[-3:]

        # Sometimes we get the truth.txt file, which we do not want
        if file == "truth.txt":
            return

        # Get the observation set, time, and name from the directory
        obset_str, epoch_str, *name = directory.split("_")
        name = "_".join(name)

        # We slice off the first part of the string that is non-numeric, where
        # 5 = len("ObSet")
        obset_int = int(obset_str[5:])

        # Get the chip ID, and perhaps more, from the name of the file
        _, chip_id_str, *misc = file.split("-")

        # SLICE chips have more information
        if misc:
            chip_id_int = int(chip_id_str)
            _, time = misc
            # Remove file extension
            time, _ = os.path.splitext(time)
            # Remove _1 at end of each time and convert to microseconds
            time = time[:-2] + "000"
            # Get milliseconds since the unix epoch
            epoch = datetime.utcfromtimestamp(0)
            dt = datetime.strptime(time, "%Y%m%d_%H%M%S.%f")
            epoch_str = str(int((dt - epoch).total_seconds()))
        else:
            chip_id, _ = os.path.splitext(chip_id_str)
            chip_id_int = int(chip_id)

        idx_key = (obset_int, chip_id_int)
        idx_val = {
            'file': file_path,
            'meta': {
                'obSetName': name,
                'epoch': epoch_str,
            },
        }
        return idx_key, idx_val

    def __create_chip(self, file_info, truth_value):
        """Converts parsing / indexing results into a pelops.datasets.chip.Chip object"""
        if truth_value == 0:
            self.__noise_seq += 1
            car_id = 'unk-{:09d}'.format(self.__noise_seq)
        else:
            car_id = 'tgt-{:09d}'.format(truth_value)

        chip_params = [
            file_info['file'],
            car_id,
            file_info['meta']['obSetName'],
            file_info['meta']['epoch'],
            file_info['meta']
        ]
        return chip.Chip(*chip_params)

    def __set_chips(self):
        """Sets the chips dict of the superclass to contain chip files for the dataset."""

        # Scan filesystem
        root_files = [root_file for root_file in os.walk(self.dataset_path)]

        # Decode truth.txt file
        truth_files = [os.path.join(walked[0], 'truth.txt') for walked in root_files if 'truth.txt' in walked[2]]
        if len(truth_files) == 0:
            raise IOError("No truth file found.")
        elif len(truth_files) > 1:
            raise IOError("Too many truth files available.")

        truth_data = self.__decode_truth_file(truth_files.pop())
        if len(truth_data) < 1:
            raise IOError("No truth loaded")
        if self.__debug:
            print("{} truth records loaded.".format(len(truth_data)))

        # Index all image chips
        file_paths = [[os.path.join(walked[0], wfile) for wfile in walked[2]] for walked in root_files]
        chip_idx = dict(filter(lambda t: t is not None, map(self.index_chip, itertools.chain(*file_paths))))

        if len(chip_idx) != len(truth_data):
            raise IOError("Number of truth records not equal to number of chips.")
        if self.__debug:
            print("{} image chips loaded.".format(len(chip_idx)))

        # Create and store chips
        self.chips = {meta['file']: self.__create_chip(meta, truth_data[idx]) for idx, meta in chip_idx.items()}
        if self.__debug:
            print("{} chip.Chips loaded.".format(len(self.chips)))
