# coding: utf-8
import os
import sys
import time

import numpy as np
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.models import Model, model_from_json
from keras.preprocessing import image

DEFAULT_LAYER_NAME = 'flatten_1'


def load_image(img_path):
    data = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(data)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def save_model_workaround(model, layer, model_output_file, weights_output_file, layer_output_file):
    print('saving model   to {}'.format(model_output_file))
    print('saving weights to {}'.format(weights_output_file))
    print('saving layer   to {}'.format(layer_output_file))
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_output_file, 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weights_output_file)
    # Write layer name to text
    with open(layer_output_file, 'w') as lyr_out:
        lyr_out.write(layer)


def load_model_workaround(model_file, weight_file):
    # load json and create model
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weight_file)
    return loaded_model


def get_models(model=None, weights=None, layer=None):
    # include_top needs to be True for this to work
    if model is None or weights is None or layer is None:
        print('MODEL NOT FULLY SPECIFIED, USING RESNET FEATURES')
        base_model = ResNet50(weights='imagenet', include_top=True)
        model = Model(input=base_model.input,
                      output=base_model.get_layer(DEFAULT_LAYER_NAME).output)
    else:
        base_model = load_model_workaround(model, weights)
        base_layer_names = {lyr.name for lyr in base_model.layers}
        base_is_siamese = all([(name in base_layer_names) for name in ['dense_1', 'dense_2', 'lambda_1']])

        if base_is_siamese:
            print('Input model is siamese, extracting resnet.')
            fresh_resnet = ResNet50(weights='imagenet', include_top=True)
            fresh_resnet.set_weights(base_model.get_layer('resnet50').get_weights())
            model = Model(input=fresh_resnet.input,
                          output=fresh_resnet.get_layer(DEFAULT_LAYER_NAME).output)
        else:
            model = Model(input=base_model.input,
                          output=base_model.get_layer(layer).output)
    return model


def image_features(img, model):
    predictions = model.predict(img)
    return predictions


def find_images(topdir):
    retval = []
    exten = ['jpg', 'bmp', 'png']
    images = 'images'

    for dirpath, dirnames, files in os.walk(topdir):
        for name in files:
            if name.lower().split('.')[-1] in exten:
                if dirpath.lower().find(images):
                    retval.append(os.path.join(dirpath, name))
    return retval


def write_data(vector_file, image_file, feature):
    list_feature = feature.flatten().tolist()
    str_feature = ','.join(str(j) for j in list_feature)
    outdata = '{0},{1}\n'.format(image_file, str_feature)
    vector_file.write(outdata)
    vector_file.flush()


def main(argv=None):
    if argv is None:
        argv = sys.argv
    image_dir = argv[1]
    vector_dir = argv[2]

    model_file = os.environ.get('MODEL', None)
    weights_file = os.environ.get('WEIGHTS', None)
    layer_name = os.environ.get('LAYER', None)

    vector_file_name = os.path.join(
        vector_dir, 'vectorOutputFile_{0}.csv'.format(time.time()))
    vector_file = open(vector_file_name, 'w')

    images = find_images(image_dir)

    model = get_models(model_file, weights_file, layer_name)

    # Export model, weights, and layer if not originally supplied by the environment
    if all(map(lambda v: v is None, [model_file, weights_file, layer_name])):
        date_time = time.strftime('%Y%m%d_%H%M%S')
        make_out_file = lambda n: os.path.join(vector_dir, date_time + '.' + n)
        save_model_workaround(model, DEFAULT_LAYER_NAME, make_out_file('model'),
                              make_out_file('weights'), make_out_file('layer'))

    for image_file in images:
        img = load_image(image_file)
        feature = image_features(img, model)
        write_data(vector_file, image_file, feature)
        print('processed {0}'.format(image_file))

    vector_file.close()

if __name__ == "__main__":
    sys.exit(main())
