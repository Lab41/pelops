import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model, Model, model_from_json

from PIL import Image as PIL_Image
from pelops.features.feature_producer import FeatureProducer


class KerasModelFeatureProducer(FeatureProducer):
    def __init__(self, chip_producer, model_filename, layer_name, weight_filename=None):
        global resnet_model
        super().__init__(chip_producer)

        if weight_filename is None:
            self.original_model = load_model(model_filename)
        else:
            self.original_modle = load_model_workaround(model_filename,weight_filename)

        self.keras_model = Model(input=self.original_model.input,
                                 output=self.original_model.get_layer(layer_name).output)

    @staticmethod
    def load_model_workaround(model_filename,weight_filename):
        # load json and create model
        json_file = open(model_filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(weight_filename)
        return loaded_model

    @staticmethod
    def preprocess_image(img, x_dim=224, y_dim=224):
        img = img.resize((x_dim,y_dim), PIL_Image.BICUBIC)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def produce_features(self, chip):
        pil_image = self.get_image(chip)
        preprocessed_image = self.preprocess_image(pil_image)
        image_features = self.keras_model.predict(preprocessed_image)
        return image_features

    def set_variables(self):
        self.feat_size = 2048
