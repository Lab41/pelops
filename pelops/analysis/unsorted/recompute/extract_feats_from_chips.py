import numpy as np
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.models import model_from_json
from keras.preprocessing import image

from pelops.datasets.featuredataset import FeatureDataset


def load_image(img_path, resizex=224, resizey=224):
    data = image.load_img(img_path, target_size=(resizex, resizey))
    x = image.img_to_array(data)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def save_model_workaround(model, model_file, weight_file):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_file, 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weight_file)


def load_model_workaround(model_file, weight_file):
    # load json and create model
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weight_file)
    return loaded_model

# load the imagenet networks


def get_models(model_file, weight_file, layer):
    # include_top needs to be True for this to work
    base_model = load_model_workaround(model_file, weight_file)
    output_layer = base_model.get_layer(layer)
    output_layer = output_layer.output
    model = Model(input=base_model.input, output=output_layer)
    # output=base_model.get_layer('flatten_1').output)
    return (model, base_model)

# return feature vector for a given img, and model


def image_features(img, model):
    features = model.predict(img)
    return features


def extract_feats_from_chips(chipdataset, output_fname, model_file, weight_file, layer):
    model, base_model = get_models(model_file, weight_file, layer)

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
