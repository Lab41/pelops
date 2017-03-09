import numpy as np
from PIL import Image

from pelops.datasets.chipper import Chipper
from pelops.datasets.featuredataset import FeatureDataset


class FeatureProducer(object):
    def __init__(self, chip_producer):
        self.chip_producer = chip_producer
        self.set_variables()

    def return_features(self):
        if isinstance(self.chip_producer, Chipper):
            chips = []
            chip_keys = []
            for chip_list in self.chip_producer:
                chips.extend(chip_list)
                for i, chip in enumerate(chip_list):
                    chip_keys.append('{}_{}'.format(chip.frame_number, i))

        else:
            chips = []
            chip_keys = []
            for chip_key, chip in self.chip_producer.chips:
                chips.append(chip)
                chip_keys.append(chip_key)

        feats = np.zeros((len(chips), self.feat_size), dtype=np.float32)
        for i, chip in enumerate(chips):
            feats[i] = self.produce_features(chip)
        return chip_keys, chips, feats

    @staticmethod
    def get_image(chip):
        if hasattr(chip, 'img_data'):
            img = Image.fromarray(chip.img_data)
            return img.convert('RGB')
        else:
            return Image.open(chip.filepath)

    def produce_features(self, chip):
        """Takes a chip object and returns a feature vector of size
        self.feat_size. """
        raise NotImplementedError("produce_features() not implemented")

    def save_features(self, output_filename):
        """
        Calculate features and save as a "FeatureDataset"
        Args:
            filename:

        Returns:

        """
        # TODO: See if this function should save the features in memory
        if isinstance(self.chip_producer, Chipper):
            raise NotImplementedError("Only ChipDatasets are supported at this time")
        chip_keys, chips, features = self.return_features()
        FeatureDataset.save(output_filename, chip_keys, chips, features)

    def set_variables(self):
        """Child classes should use this to set self.feat_size, and any other
        needed variables. """
        self.feat_size = None  # Set this in your inherited class
        raise NotImplementedError("set_variables() is not implemented")
