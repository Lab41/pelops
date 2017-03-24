# coding: utf-8
import json
import os
import sys
import time

import numpy as np
import scipy.spatial.distance
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.preprocessing import image


def load_image(img_path):
    data = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(data)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def get_models():
    # include_top needs to be True for this to work
    base_model = ResNet50(weights='imagenet', include_top=True)
    model = Model(input=base_model.input,
                  output=base_model.get_layer('flatten_1').output)
    return (model, base_model)


def image_features(img, model):
    features = np.zeros((1, 2048), dtype=np.float16)
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
    vector_file_name = os.path.join(
        vector_dir, 'vectorOutputFile_{0}.csv'.format(time.time()))
    vector_file = open(vector_file_name, 'w')

    images = find_images(image_dir)

    model, base_model = get_models()

    for image_file in images:
        img = load_image(image_file)
        feature = image_features(img, model)
        write_data(vector_file, image_file, feature)
        print('processed {0}'.format(image_file))

    vector_file.close()

if __name__ == "__main__":
    sys.exit(main())
