import os
import io
import re
import sys
import csv
import datetime
import itertools

import pelops.datasets.chip as chip

# ================================================================================
#  SLiCE Test Dataset (labled by STR)
# ================================================================================


class SliceDataset(chip.ChipDataset):

    def __init__(self, dataset_path, set_type=None, debug=False):
        super().__init__(dataset_path, set_type)
        self.__obset_txt = '(?P<obSetId>ObSet(?P<obSetIdx>\d+)_(?P<epoch>\d+)_(?P<obSetName>.+?)[/\\\\]images[/\\\\]ObSet\d+-(?P<chipId>\d+).png)'
        self.__obset_ptn = re.compile(self.__obset_txt, re.S | re.I)
        self.__noise_seq = 0
        self.__debug = debug
        self.__set_chips()

    @staticmethod
    def __decode_truth_file(truth_file):
        """The labels for the STR processed SLiCE chips are in a 'truth.txt' file which this function parses."""

        try:
            with open(truth_file) as truth_hdl:
                truth_text = truth_hdl.read()
                for char in [' ', '%']:
                    truth_text = truth_text.replace(char, '')
                truth_fobj = io.StringIO(truth_text)
                return {(int(dct['obSetIdx']), int(dct['chipIdx'])): int(dct['targetID'])
                        for dct in csv.DictReader(truth_fobj)}

        except Exception as e:
            sys.stderr.write(str(e))

        return {}

    def __index_chip(self, file_path):
        """Parses an arbitrary file path and identifies paths of valid image chips.  Returns None for non-chip file paths."""

        try:
            mch = re.search(self.__obset_ptn, file_path)
            if mch is None:
                return None

            idx_key = (int(mch.group('obSetIdx')), int(mch.group('chipId')))
            idx_val = {'file': file_path, 'meta': mch.groupdict()}
            return idx_key, idx_val

        except Exception as e:
            sys.stderr.write(str(e))

    def __create_chip(self, file_info, truth_value):
        """Converts parsing / indexing results into a pelops.datasets.chip.Chip object"""

        try:
            if truth_value == 0:
                self.__noise_seq += 1
                car_id = 'unk-%.9d' % self.__noise_seq
            else:
                car_id = 'tgt-%.9d' % truth_value

            chip_params = [
                file_info['file'],
                car_id,
                file_info['meta']['obSetName'],
                datetime.datetime.fromtimestamp(float(file_info['meta']['epoch'])).isoformat(),
                file_info['meta']
            ]
            return chip.Chip(*chip_params)

        except Exception as e:
            sys.stderr.write(str(e))

    def __set_chips(self):
        """Sets the chips dict of the superclass to contain chip files for the dataset."""

        # Scan filesystem
        root_files = [root_file for root_file in os.walk(self.dataset_path)]

        # Decode truth.txt file
        truth_files = [os.path.join(walked[0], 'truth.txt') for walked in root_files if 'truth.txt' in walked[2]]
        if len(truth_files) == 0:
            raise Exception("No truth file found.")
        elif len(truth_files) > 1:
            raise Exception("Too many truth files available.")

        truth_data = self.__decode_truth_file(truth_files.pop())
        if len(truth_data) < 1:
            raise Exception("No truth loaded")
        if self.__debug:
            print("%d truth records loaded" % len(truth_data))

        # Index all image chips
        file_paths = [[os.path.join(walked[0], wfile) for wfile in walked[2]] for walked in root_files]
        chip_idx = dict(filter(lambda t: t is not None, map(self.__index_chip, itertools.chain(*file_paths))))

        if len(chip_idx) != len(truth_data):
            raise Exception("Number of truth records not equal to number of chips.")
        if self.__debug:
            print("%d image chips loaded" % len(chip_idx))

        # Create and store chips
        self.chips = {meta['file']: self.__create_chip(meta, truth_data[idx]) for idx, meta in chip_idx.items()}
        if self.__debug:
            print("%d chip.Chips loaded" % len(self.chips))

if __name__ == '__main__':
    import pprint
    root_dir = '/Users/schuylerx/dev/data'
    slice_data = SliceDataset(root_dir, set_type='test', debug=True)
    pprint.pprint(slice_data.chips)
