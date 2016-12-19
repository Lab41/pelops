from abc import ABCMeta
from abc import abstractmethod
from collections import namedtuple

Chip = namedtuple('Chip', ['chip_id', 'car_id',
                           'cam_id', 'time', 'filename', 'misc'])


class ChipBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, dataset_name, *args, **kwargs):
        self.dataset_name = dataset_name
        self.chips = {}

    def get_all_chips_by_carid(self, car_id):
        return [chip for chip in self.chips.values() if chip.car_id == car_id]

    def get_all_chips_by_camid(self, cam_id):
        return [chip for chip in self.chips.values() if chip.cam_id == cam_id]

    def get_chip_image_path(self, chip_id):
        """Returns the image path associated with a specific chip.

        Args:
            chip_id: Unique id of the chip to retrieve the image path from.

        Returns:
            str: Returns a string indicating the full image path, or
                None if the chip is not found.
        """
        chip = self.chips.get(chip_id, None)
        if chip is not None:
            return chip.filename
        else:
            return None

    def __iter__(self):
        """
        Iterates over dataset return metadata about dataset, one record at a time
        Returns:
            (image file name, caption string, list of tag strings associated with next item)
        """
        for chip in self.chips.values():
            yield chip

        raise StopIteration()

    def __len__(self):
        return len(self.chips)
