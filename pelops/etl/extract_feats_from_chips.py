import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import Model

from PIL import Image as PIL_Image
from pelops.datasets.featuredataset import FeatureDataset


def load_image(img_path, resize_x=224, resize_y=224):
    data = image.load_img(img_path, target_size=(resize_x, resize_y))
    x = image.img_to_array(data)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def load_array(img_arr, resize_x=224, resize_y=224):
    img = PIL_Image.fromarray(img_arr)
    img = img.convert('RGB')
    img = img.resize((resize_x,resize_y), PIL_Image.BICUBIC)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Make sure to only get models once
base_model = None
model = None


def get_models():
    """
    Load imagenet model and return. Make sure to only load the model once
    Returns:

    """
    global base_model
    global model
    if not base_model and not model:
        # include_top needs to be True for this to work
        base_model = ResNet50(weights='imagenet', include_top=True)
        model = Model(input=base_model.input,
                      output=base_model.get_layer('flatten_1').output)
    return (model, base_model)


# return feature vector for a given img, and model
def image_features(img, model):
    """
    Take in an image and return the feature vector for that image by running it through the model
    Args:
        img: A preprocessed image of the correct dimensions
        model: a

    Returns:

    """
    features = model.predict(img)
    return features

def extract_feats_from_chips(chipdataset, output_fname):
    model, base_model = get_models()

    features = np.zeros((len(chipdataset), 2048), dtype=np.float16)
    chips = []
    chip_keys = []
    for index, (chip_key, chip) in enumerate(chipdataset.chips.items()):
        chip_keys.append(chip_key)
        chips.append(chip)
        img_path = chip.filepath
        img_data = load_image(img_path)
        features[index] = image_features(img_data, model)
    
    FeatureDataset.save(output_fname, chip_keys, chips, features)
    return True

class FeatureProducer(object):
    def __init__(self,
                 chip_producer):
        self.chip_producer = chip_producer
        self.model, self.base_model = get_models()

    def __iter__(self):
        for chips in self.chip_producer:
            feats = np.zeros((len(chips), 2048), dtype=np.float32)
            for i, chip in enumerate(chips):
                preprocessed_chip = load_array(chip.img_data)
                feats[i] = image_features(preprocessed_chip, self.model)
            yield chips, feats
