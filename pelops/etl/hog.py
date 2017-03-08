from skimage.feature import hog
from skimage import colos

from pelops.etl.feature_producer import FeatureProducer


class HOGFeatureProducer(FeatureProducer):

    def __init__(self, chip_producer):
        super().__init__(chip_producer)

    def produce_features(self, chip):
        """Takes a chip object and returns a feature vector of size
        self.feat_size. """
        img = PIL_Image.open(chip.filepath)
        img = img.resize((256, 256), PIL_Image.BICUBIC)
        img_x, img_y = img.size
        img = color.rgb2gray(np.array(img))
        features = hog(
            img,
            orientations=8,
            pixels_per_cell=(int(img_x / 16), int(img_y / 16)),
            cells_per_block=(16, 16),  # Normalize over the whole image
        )
        return features

    def set_variables(self):
        """Child classes should use this to set self.feat_size, and any other
        needed variables. """
        self.feat_size = 2048  # Set this in your inherited class
