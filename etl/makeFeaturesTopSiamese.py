import json
import os
import sys
import time

import keras.backend.tensorflow_backend as KTF
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import merge
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.models import Model
from keras.models import model_from_json
from keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical


def just_the_top(num_training_classes, model_file, weights_file):

    def load_model_workaround(model_file, weight_file):
        # load json and create model
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(weight_file)
        return loaded_model

    def s_distance(vects):
        """
        return the abs difference between vectors
        """
        x, y = vects
        s = K.abs(x - y)
        return s

    def s_shape(shapes):
        """
        return the sape of the vector being used
        """
        shape = list(shapes)
        outshape = (shape[0])
        return tuple(outshape)

    original_model = load_model_workaround(model_file, weights_file)
    d1 = original_model.get_layer('dense_1')
    d1_len = d1_len = d1.get_output_shape_for(d1.get_input_shape_at(0))[1]
    d2 = original_model.get_layer('dense_2')
    b1 = original_model.get_layer('batchnormalization_1')

    input_left = Input(shape=(1, 1, 2048))
    input_right = Input(shape=(1, 1, 2048))

    # use a distance measure for making the join
    siamese_join = Lambda(s_distance,
                          output_shape=s_shape)([input_left, input_right])
    my_layer = GlobalAveragePooling2D()(siamese_join)
    my_d1 = Dense(d1_len, activation='relu')(my_layer)
    bn = BatchNormalization()(my_d1)
    predictions = Dense(num_training_classes, activation='sigmoid')(bn)
    model = Model([input_left, input_right], output=predictions)

    print(model.summary())
    model.get_layer('dense_1').set_weights(d1.get_weights())
    model.get_layer('dense_2').set_weights(d2.get_weights())
    model.get_layer('batchnormalization_1').set_weights(b1.get_weights())

    return model


def write_data(vector_file, index, feature):
    list_feature = feature.flatten().tolist()
    str_feature = ','.join(str(j) for j in list_feature)
    outdata = '{0}|{1}\n'.format(index, str_feature)
    vector_file.write(outdata)
    vector_file.flush()


def make_top():
    a = np.ones((1, 1, 1, 2048))
    top = just_the_top(3,
                       '/pelops_root/MODEL_DIR/VeRi-siamese-weekend.model.json',
                       '/pelops_root/MODEL_DIR/VeRi-siamese-weekend.weights.hdf5')
    print('*********** test **********')
    print(top.predict([a, a])[0])
    # Out[8]: array([[ 0.98460394,  0.99653435,  0.99870515]], dtype=float32)
    print('*********** test **********')
    return top


def main(argv=None):

    #model = make_top()
    # test()

    if argv is None:
        argv = sys.argv
    image_dir_l = argv[1]
    image_dir_r = argv[2]
    output_dir = argv[3]

    input_file_name = os.environ.get('VECTORS', None)
    model_file = os.environ.get('MODEL', None)
    weights_file = os.environ.get('WEIGHTS', None)

    vector_file_name = os.path.join(
        output_dir, 'vectorOutputFile_{0}.csv'.format(time.time()))

    vector_o_file = open(vector_file_name, 'w')
    vector_i_file = open(input_file_name, 'r')

    print(3, model_file, weights_file)
    model = just_the_top(3, model_file, weights_file)

    for index, line in enumerate(vector_i_file):
        line = line.strip()
        j_line = json.loads(line)
        left = j_line['left']
        right = j_line['right']
        np_l = np.array(left)
        np_r = np.array(right)
        np_l = np_l.reshape(1, 1, 1, 2048)
        np_r = np_r.reshape(1, 1, 1, 2048)
        data = [np_l, np_r]
        feature = model.predict(data)
        feature = feature[0]
        write_data(vector_o_file, index, feature)

    vector_o_file.close()

if __name__ == "__main__":
    sys.exit(main())
