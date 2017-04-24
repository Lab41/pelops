import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import Model

from PIL import Image as PIL_Image
from pelops.features.feature_producer import FeatureProducer

# Use global so we only load the resnet model once
# TODO: find a better way to do this
resnet_model = None


class ResNet50FeatureProducer(FeatureProducer):
    def __init__(self, chip_producer):
        global resnet_model
        super().__init__(chip_producer)

        if resnet_model is None:
            # include_top needs to be True for this to work
            base_model = ResNet50(weights='imagenet', include_top=True)
            resnet_model = Model(input=base_model.input,
                      output=base_model.get_layer('flatten_1').output)

        self.resnet_model = resnet_model

    @staticmethod
    def preprocess_image(img, x_dim=224, y_dim=224):
        if img.size != (x_dim, y_dim):
            img = img.resize((x_dim,y_dim), PIL_Image.BICUBIC)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def produce_features(self, chip):
        pil_image = self.get_image(chip)
        preprocessed_image = self.preprocess_image(pil_image)
        image_features = self.resnet_model.predict(preprocessed_image)
        return image_features

    def set_variables(self):
        self.feat_size = 2048
