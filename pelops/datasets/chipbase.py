from abc import ABCMeta
from abc import abstractmethod
from collections import namedtuple

Chip = namedtuple('Chip', ['chip_id', 'car_id',
                           'cam_id', 'time', 'filename', 'misc'])


class ChipBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, dataset_name, *args, **kwargs):
        self.dataset_name = dataset_name

    def get_all_chips_by_carid(self, car_id):
        return [chip for chip in self.chips if chip.car_id == car_id]

    def get_all_chips_by_camid(self, cam_id):
        return [chip for chip in self.chips if chip.cam_id == cam_id]

    @abstractmethod
    def get_chip_image_path(self, chip_id):
        pass

    def __iter__(self):
        """
        Iterates over dataset return metadata about dataset, one record at a time
        Returns:
            (image file name, caption string, list of tag strings associated with next item)
        """
        for chip in self.chips:
            yield chip

        raise StopIteration()

    def __len__(self):
        return len(self.chips)
