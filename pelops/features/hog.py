import numpy as np
from PIL import Image
from skimage import color
from skimage.feature import hog

from pelops.features.feature_producer import FeatureProducer


class HOGFeatureProducer(FeatureProducer):

    def __init__(self, chip_producer, image_size=(256,256), cells=(16, 16), orientations=8, histogram_bins_per_channel=256):
        self.image_size = image_size
        self.cells = cells
        self.orientations = orientations
        self.histogram_bins_per_channel = histogram_bins_per_channel
        super().__init__(chip_producer)

    def produce_features(self, chip):
        """Takes a chip object and returns a feature vector of size
        self.feat_size. """
        img = self.get_image(chip)
        img = img.resize(self.image_size, Image.BICUBIC)
        img_x, img_y = img.size

        # Calculate histogram of each channel
        hists = []
        for channel in img.split():
            channel_array = np.array(channel)
            values, _ = np.histogram(channel_array.flat, bins=self.histogram_bins_per_channel)
            hists.append(values)

        hists_values = np.concatenate(hists)

        # Calculate HOG features, which require a grayscale image
        img = color.rgb2gray(np.array(img))
        features = hog(
            img,
            orientations=self.orientations,
            pixels_per_cell=(img_x / self.cells[0], img_y / self.cells[1]),
            cells_per_block=self.cells,  # Normalize over the whole image
        )

        return np.concatenate((features, hists_values))

    def set_variables(self):
        hog_size = self.cells[0] * self.cells[1] * self.orientations
        hist_size = 3 * histogram_bins_per_channel
        self.feat_size = hog_size + hist_size
