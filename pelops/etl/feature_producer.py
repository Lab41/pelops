import numpy as np


class FeatureProducer(object):

    def __init__(self, chip_producer):
        self.chip_producer = chip_producer
        self.set_variables()

    def return_features(self):
        chips = self.chip_producer.chips.values()
        feats = np.zeros((len(chips), self.feat_size), dtype=np.float32)
        for i, chip in enumerate(chips):
            feats[i] = self.produce_features(chip)
        return chips, feats

    def produce_features(self, chip):
        """Takes a chip object and returns a feature vector of size
        self.feat_size. """
        raise NotImplementedError("produce_features() not implemented")

    def set_variables(self):
        """Child classes should use this to set self.feat_size, and any other
        needed variables. """
        self.feat_size = None  # Set this in your inherited class
        raise NotImplementedError("set_variables() is not implemented")
