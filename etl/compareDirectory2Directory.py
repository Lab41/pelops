# coding: utf-8
import json
import os
import sys
import time

import numpy as np
import scipy.spatial.distance
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.preprocessing import image


def load_image(img_path):
    data = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(data)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def load_model_workaround(model_file, weight_file):
    # load json and create model
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weight_file)
    return loaded_model


def get_models(model=None,weights=None):
    model = load_model_workaround(model, weights)
    return model


def image_features(left,right, model):
    predictions = model.predict([left,right])
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


def write_data(vector_file, limage_file, rimage_file, feature):
    list_feature = feature.flatten().tolist()
    str_feature = ','.join(str(j) for j in list_feature)
    outdata = '{0},{1},{2}\n'.format(limage_file,rimage_file, str_feature)
    vector_file.write(outdata)
    vector_file.flush()


def main(argv=None):
    if argv is None:
        argv = sys.argv
    image_dir_l = argv[1]
    image_dir_r = argv[2]
    vector_dir = argv[3]

    model_file = os.environ.get('MODEL',None)
    weights_file = os.environ.get('WEIGHTS',None)
    layer = os.environ.get('LAYER',None)

    vector_file_name = os.path.join(vector_dir,
                                    'vectorOutputFile_{0}.csv'.format(time.time()))

    vector_file = open(vector_file_name, 'w')

    images_left = find_images(image_dir_l)
    images_right = find_images(image_dir_r)

    model = get_models(model_file, weights_file)

    for limage_file in images_left:
        for rimage_file in images_right:

            l_img = load_image(limage_file)
            r_img = load_image(rimage_file)

            feature = image_features(l_img,r_img, model)

            write_data(vector_file, image_file, feature)

    vector_file.close()

if __name__ == "__main__":
    sys.exit(main())
