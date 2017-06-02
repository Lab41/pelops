# coding: utf-8

# In[1]:


import datetime
import glob
import hashlib
import multiprocessing as mp
import os
import queue
import random
import threading
from functools import partial

import keras.backend.tensorflow_backend as KTF
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.layers import Dense, GlobalAveragePooling2D, Input, Lambda, merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model, model_from_json
from keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

import pelops.utils as utils
from pelops.analysis import analysis
from pelops.analysis.camerautil import get_match_id, make_good_bad
from pelops.datasets.featuredataset import FeatureDataset
from pelops.datasets.veri import VeriDataset
from pelops.experiment_api.experiment import ExperimentGenerator
from pelops.utils import train_test_key_filter


# In[2]:





# In[3]:

def save_model_workaround(model, model_output_file, weights_output_file):
    print('saving model   to {}'.format(model_output_file))
    print('saving weights to {}'.format(weights_output_file))
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_output_file, 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weights_output_file)


def load_model_workaround(model_output_file, weights_output_file):
    # load json and create model
    json_file = open(model_output_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_output_file)
    return loaded_model


# In[4]:

def makework(workitems, chips, cam_id=None):
    left = chips[0]
    right = chips[1]
    same_vehicle = left.car_id == right.car_id
    same_type = left.misc['vehicle_type'] == right.misc['vehicle_type']
    same_color = left.misc['color'] == right.misc['color']
    #same_angle = cam_id(left.cam_id) == cam_id(right.cam_id)
    features = [same_vehicle, same_type, same_color]
    workitems.append((left.filepath, right.filepath, features))
    workitems.append((right.filepath, left.filepath, features))


def make_examples(gen, examples):
    workitems = []

    for _ in range(examples):
        cameras = gen.generate()
        match_id = get_match_id(cameras)
        goods, bads = make_good_bad(cameras, match_id)

        makework(workitems, goods)
        makework(workitems, bads)

    print('made', len(workitems))
    return workitems


# In[5]:

# get a GPU session and reserve memory
def get_session(gpu_fraction=0.3):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def rgb2bgr(x):
    """
    given an array representation of an RGB image, change the image
    into an BGR representtaion of the image
    """
    return(bgr2rgb(x))


def bgr2rgb(x):
    """
    given an array representation of an BGR image, change the image
    into an RGB representtaion of the image
    """
    y = np.zeros(x.shape)
    B = x[:, :, 0]
    G = x[:, :, 1]
    R = x[:, :, 2]
    y[:, :, 0] = R
    y[:, :, 1] = G
    y[:, :, 2] = B
    return y

# load an image from disk
# NOTE: input assumed to be RGB
# NOTE: output is to be BGR for resnet use.


def load_image(img_path,
               e_dims=False,
               image_flip=0.5,
               image_shift=0.20,
               image_rotate_degrees=15,
               image_zoom=0.15,
               output_BGR=True):
    """
    WARNING this funciton should only manipulation images meant for resnet50 consumption.
    To make it applicable for other environments remove preprocess_input.


    Do some image manipulation
    image input assumed to be in RGB format
    output format default is GBR unless output_BGR is set to False

    e_dims = e_dims false will output (x,y,3) sized images
             e_domes true will output (1,x,y,3) sized images
    image_flip = probability that image will be flipped rt to left
    image_shift = percent of image to randomly shift up/down & right/left
    image_rotate_degrees = rotate image randomly
                            between [-image_rotate_degrees image_rotate_degrees]
    image_zoom = randomly zoom image [1-image_zoom 1+image_zoom]
    output_BGR = True -> image output will be in BGR formate RGB otherwise
    """
    img = image.load_img(img_path, target_size=(224, 224))
    my_img = image.img_to_array(img)

    if image_flip is not None:
        if image_flip > 1 or image_flip < -1:
            raise ValueError('|image_flip:{0}| > 1'.format(image_flip))
        image_flip = abs(image_flip)
        if random.random() > image_flip:
            my_img = image.flip_axis(my_img, axis=1)

    if image_rotate_degrees is not None:
        image_rotate_degrees = int(image_rotate_degrees)

        if image_rotate_degrees > 360:
            image_rotate_degrees = image_rotate_degrees % 360

        my_img = image.random_rotation(my_img,
                                       image_rotate_degrees,
                                       row_index=0,
                                       col_index=1,
                                       channel_index=2)
    if image_shift is not None:
        if image_shift > 1 or image_shift < -1:
            raise ValueError('|image_shift:{0}| > 1'.format(image_shift))
        image_shift = abs(image_shift)

        my_img = image.random_shift(my_img,
                                    image_shift,
                                    image_shift,
                                    row_index=0,
                                    col_index=1,
                                    channel_index=2)

    if image_zoom is not None:
        if image_zoom > 1 or image_zoom < -1:
            raise ValueError('|image_zoom:{0}| > 1'.format(image_zoom))
        image_zoom = abs(image_zoom)

        low = 1 - image_zoom
        high = 1 + image_zoom
        rng = [low, high]
        my_img = image.random_zoom(my_img,
                                   rng,
                                   row_index=0,
                                   col_index=1,
                                   channel_index=2)

    if not output_BGR:
        my_img = bgr2rgb(my_img)

    my_img = np.expand_dims(my_img, axis=0)
    my_img = preprocess_input(my_img)

    if not e_dims:
        my_img = my_img.squeeze()

    return my_img


# In[6]:

def plot_run_no(history, name1, name2, rnd=None):
    """
    Take the output of a model.
    """
    v = np.array(history[name1])
    vc = np.array(history[name2])
    if rnd is not None:
        vr = np.zeros(vc.shape)
        vr.fill(rnd)
        b = np.array([v, vc, vr])
    else:
        b = np.array([v, vc])
    c = b.transpose()
    ax = plt.subplot(111)
    ax.grid(True)
    ax.plot(c)
    if rnd is not None:
        ax.legend((name1, name2, 'random'),
                  bbox_to_anchor=(1, -0.05),
                  fancybox=True, shadow=True, ncol=5)
    else:
        ax.legend((name1, name2),
                  bbox_to_anchor=(1, -0.05),
                  fancybox=True, shadow=True, ncol=5)

    plt.show()


# In[7]:

def image_class_generator(tasking, batch_size=32, augment=False):
    """
    Offload the augmentation of images, create images in batch_size chunks
    augment=False -> return image  augment=True -> return augmented image
    """
    while True:
        lefts = []
        rights = []
        ys = []
        for task in random.sample(tasking, batch_size):
            left_file = task[0]
            right_file = task[1]
            classes = task[2]
            y = np.zeros(len(classes))
            for index, c in enumerate(classes):
                y[index] = 1 if c else 0
            l_img = None
            r_img = None
            if augment:
                l_img = load_image(left_file)
                r_img = load_image(right_file)
            else:
                l_img = load_image(left_file, False, None, None, None, None)
                r_img = load_image(right_file, False, None, None, None, None)
            lefts.append(l_img)
            rights.append(r_img)
            ys.append(y)

        yield ([np.array(lefts), np.array(rights)], np.array(ys))


def buffered_gen_mp(source_gen, buffer_size=2, num_processes=4):
    """
    Generator that runs a slow source generator in a separate process.
    buffer_size: the maximal number of items to pre-generate (length of the buffer)
    """
    if buffer_size < 2:
        raise RuntimeError("Minimal buffer size is 2!")

    buffer = mp.Queue(maxsize=buffer_size - 1)
    # the effective buffer size is one less, because the generation process
    # will generate one extra element and block until there is room in the
    # buffer.

    def _buffered_generation_process(source_gen, buffer):
        for data in source_gen:
            buffer.put(data, block=True)
        buffer.put(None)  # sentinel: signal the end of the iterator
        buffer.close()  # unfortunately this does not suffice as a signal: if buffer.get()
        # was called and subsequently the buffer is closed, it will block
        # forever.

    for _ in range(num_processes):
        process = mp.Process(
            target=_buffered_generation_process, args=(source_gen, buffer))
        process.start()

    for data in iter(buffer.get, None):
        yield data


# In[8]:

def freeze(model):
    """
    Make model untrainable
    """
    for layer in model.layers:
        layer.trainable = False
    model.trainable = False


# In[9]:

def free_model_layers(model):
    """
    Make the model trainable
    """
    for layer in model.layers:
        try:
            if layer.name == 'resnet50':
                print('found resnet')
                for rn_layer in layer.layers:
                    try:
                        if not rn_layer.trainable:
                            rn_layer.trainable = True
                    except:
                        if 'merge' not in rn_layer.name:
                            print('rn layer not trainable', rn_layer.name)
            if not layer.trainable:
                layer.trainable = True
        except:
            if 'merge' not in layer.name.lower():
                print('layer not trainable:', layer.name)


# In[10]:

def make_siamese_model_concat(num_training_classes=3):
    """
    Siamese network created via concatenating resnet50 outputs

    @TODO see if less layers can now be used because of not using
    binary_crossentropy..
    """
    base_model = ResNet50(weights='imagenet', include_top=False)

    freeze(base_model)

    input_left = Input(shape=(224, 224, 3))
    input_right = Input(shape=(224, 224, 3))

    processed_left = base_model(input_left)
    processed_right = base_model(input_right)

    # join by slapping vectors together
    siamese_join = merge([processed_left, processed_right], mode='concat')

    my_layer = GlobalAveragePooling2D()(siamese_join)
    my_layer = Dense(4096, activation='relu')(my_layer)
    my_layer = BatchNormalization()(my_layer)
    my_layer = Dense(2048, activation='relu')(my_layer)
    my_layer = BatchNormalization()(my_layer)
    my_layer = Dense(2048, activation='relu')(my_layer)
    predictions = Dense(num_training_classes, activation='sigmoid')(my_layer)
    model = Model([input_left, input_right], output=predictions)

    return model


# In[11]:

def s_distance(vects):
    """
    return the abs difference between vectors
    """
    x, y = vects
    s = K.abs(x - y)
    #s =  K.sqrt(K.square(x - y))
    return (s)
    # return K.squeeze(x,1) - K.squeeze(y,1)


def s_shape(shapes):
    """
    return the sape of the vector being used
    """
    shape = list(shapes)
    outshape = (shape[0])
    return tuple(outshape)


def make_siamese_model_subtract(num_training_classes=2):
    """
    Siamese network created via subtracting resnet50 outputs
    """

    base_model = ResNet50(weights='imagenet', include_top=False)

    for layer in base_model.layers:
        layer.trainable = False
    base_model.trainable = False

    input_left = Input(shape=(224, 224, 3))
    input_right = Input(shape=(224, 224, 3))

    processed_left = base_model(input_left)
    processed_right = base_model(input_right)

    # use a distance measure for making the join
    siamese_join = Lambda(s_distance,
                          output_shape=s_shape)([processed_left, processed_right])
    my_layer = GlobalAveragePooling2D()(siamese_join)
    my_layer = Dense(1024, activation='relu')(my_layer)
    my_layer = BatchNormalization()(my_layer)
    predictions = Dense(num_training_classes, activation='sigmoid')(my_layer)
    model = Model([input_left, input_right], output=predictions)

    return model


# In[12]:

def make_callbacks(model_checkpoint_format_string, tensor_board_log_dir):
    """
    programatically make the callbacks to be used for training
    """
    callbacks = []
    if model_checkpoint_format_string is not None:
        callbacks.append(ModelCheckpoint(model_checkpoint_format_string,
                                         monitor='loss',
                                         verbose=1,
                                         save_best_only=True,
                                         save_weights_only=False,
                                         mode='min',
                                         period=1))

    if tensor_board_log_dir is not None:
        callbacks.append(TensorBoard(log_dir=tensor_board_log_dir,
                                     histogram_freq=0,
                                     write_graph=True,
                                     write_images=False))

    callbacks.append(ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       patience=4,
                                       verbose=1,
                                       mode='min',
                                       epsilon=0.001,
                                       cooldown=2,
                                       min_lr=0))

    callbacks.append(EarlyStopping(monitor='val_acc',
                                   min_delta=0.003,
                                   patience=6,
                                   verbose=1,
                                   mode='max'))
    return callbacks


# In[13]:

def checkLabels(x):
    """
    Make a warm fuzzy about the classes being balanced
    """
    s_id = 0.0
    s_type = 0.0
    s_color = 0.0
    total = len(x)
    for v in x:
        if v[2][0]:
            s_id += 1
        if v[2][1]:
            s_type += 1
        if v[2][2]:
            s_color += 1
    print('P(s_id==1):{0} P(s_type==1):{1} P(s_color==1):{2}'.format(
        s_id / total, s_type / total, s_color / total))
    return s_id / total, s_type / total, s_color / total


# In[14]:

#---------------------------------------


# In[15]:

# set some constants
ITEMSPERCAMERA = 2
YRANDOM = 13024
CAMERAS = 2
DROPPED = 0
EXPERIMENTS = int(40000 / 4)
batch_size = 16
tbld = '/local_data/dgrossman/tensorboard_logs'
mcfs = '/local_data/dgrossman/tempdir/veri-siamese.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5'


# In[16]:

veri_validate = VeriDataset(
    '/local_data/dgrossman/VeRi', set_type=utils.SetType.TEST.value)
veri_train = VeriDataset('/local_data/dgrossman/VeRi',
                         set_type=utils.SetType.TRAIN.value)
expGen_validate = ExperimentGenerator(veri_train,
                                      CAMERAS,
                                      ITEMSPERCAMERA,
                                      DROPPED,
                                      YRANDOM,
                                      key_filter=partial(train_test_key_filter, split="test"))

expGen_train = ExperimentGenerator(veri_train,
                                   CAMERAS,
                                   ITEMSPERCAMERA,
                                   DROPPED,
                                   YRANDOM,
                                   key_filter=partial(train_test_key_filter, split="train"))


# In[17]:

training_examples = make_examples(expGen_train, EXPERIMENTS)
validaiton_examples = make_examples(expGen_validate, EXPERIMENTS)  # GROSSMAN


# In[18]:

checkLabels(training_examples)


# In[19]:

checkLabels(validaiton_examples)


# In[19]:

# GROSSMAN change augment to True when running for real.

train_buffered_generator_mp = buffered_gen_mp(image_class_generator(training_examples,
                                                                    batch_size,
                                                                    augment=True),
                                              buffer_size=20,
                                              num_processes=5)

val_buffered_generator_mp = buffered_gen_mp(image_class_generator(validaiton_examples,
                                                                  batch_size,
                                                                  augment=False),
                                            buffer_size=20,
                                            num_processes=5)


# In[20]:

callbacks = make_callbacks(mcfs, tbld)


# In[21]:

KTF.set_session(get_session(.90))


# In[25]:

#model = make_siamese_model_concat(3)
model = make_siamese_model_subtract(3)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[26]:

fixed_history = model.fit_generator(train_buffered_generator_mp,
                                    samples_per_epoch=10240,
                                    nb_epoch=20,
                                    callbacks=None,
                                    nb_val_samples=10240,
                                    validation_data=val_buffered_generator_mp,
                                    verbose=2)


fixed_history.history



free_model_layers(model)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])



free_history = model.fit_generator(train_buffered_generator_mp,
                                   samples_per_epoch=10240,
                                   nb_epoch=50,
                                   callbacks=callbacks,
                                   nb_val_samples=10240,
                                   validation_data=val_buffered_generator_mp,
                                   verbose=2)


save_model_workaround(model,
                      '/local_data/dgrossman/model_save_dir/VeRi-siamese-weekend-6.model.json',
                      '/local_data/dgrossman/model_save_dir/VeRi-siamese-weekend-6.weights.hdf5')
