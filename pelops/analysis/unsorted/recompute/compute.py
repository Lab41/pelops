# coding: utf-8
import datetime
import glob
import multiprocessing as mp
import os
import queue
import random
import threading

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
from keras.models import load_model
from keras.models import Model
from keras.models import model_from_json
from keras.preprocessing import image


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


# load an image from disk
def load_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# use an image with a model to get features
def image_features(img, model, length=2048):
    features = np.zeros((1, length), dtype=np.float16)
    #model = Model(input=base_model.input, output=base_model.get_layer('flatten_1').output)
    predictions = model.predict(img)
    return predictions


def image_class_generator(image_classes_mapping, num_classes, batch_size=32):
    image_list = list(image_classes_mapping.keys())
    while True:
        xs = []
        ys = []
        for filename in random.sample(image_list, batch_size):
            xs.append(load_image(filename))
            y = np.zeros(num_classes)
            y[image_classes_mapping[filename]] = 1
            ys.append(y)
        yield (np.array(xs).squeeze(), np.array(ys))


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


def save_model_workaround(model, model_output_file, weights_output_file):
    print('saving model   to {}'.format(model_output_file))
    print('saving weignts to {}'.format(weights_output_file))
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


def prep_datasets(basepath):
    image_classes = set()
    image_class_mapping = {}
    for image_class_filepath in glob.glob(os.path.join(basepath, '*')):
        if os.path.isdir(image_class_filepath):
            image_class_num = int(os.path.basename(image_class_filepath))
            image_classes.add(image_class_num)
            for filename in glob.glob(os.path.join(image_class_filepath, '*')):
                image_class_mapping[filename] = image_class_num
    print('basepath:{0},image_classes:{1}, number_images:{2}'.format(
        basepath, len(image_classes), len(image_class_mapping)))
    return (len(image_classes), image_classes, image_class_mapping)


def make_callbacks(model_checkpoint_format_string, tensor_board_log_dir):
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
                                       patience=10,
                                       verbose=1,
                                       mode='auto',
                                       epsilon=0.0001,
                                       cooldown=0,
                                       min_lr=0))

    callbacks.append(EarlyStopping(monitor='val_acc',
                                   min_delta=0.003,
                                   patience=4,
                                   verbose=1,
                                   mode='max'))
    return callbacks

# # Start the sesion


# verbose =0 -> nothing
#         1 -> progress bar
#         2 -> 1x/epoch
def do_training(training_basepath,
                validation_basepath,
                model_output_file,
                weights_output_file,
                tensor_board_log_dir=None,
                model_checkpoint_format_string=None,
                batch_size=32,
                verbose=1):

    # # Start the sesion

    KTF.set_session(get_session(.95))

    # # get data ready to process

    num_training_classes, training_image_classes, training_image_class_mapping = prep_datasets(
        training_basepath)
    num_validation_classes, validation_image_classes, validation_image_class_mapping = prep_datasets(
        validation_basepath)

    # # Create a new model with the right number of targets and pretrain top

    base_model = ResNet50(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    my_layer = base_model.output
    my_layer = GlobalAveragePooling2D()(my_layer)

    # let's add a fully-connected my_layer
    my_layer = Dense(1024, activation='relu')(my_layer)

    predictions = Dense(num_training_classes, activation='softmax')(my_layer)

    model = Model(input=base_model.input, output=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to
    # non-trainable)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = make_callbacks(
        model_checkpoint_format_string, tensor_board_log_dir)

    train_buffered_generator_mp = buffered_gen_mp(image_class_generator(training_image_class_mapping,
                                                                        num_training_classes,
                                                                        batch_size),
                                                  buffer_size=20,
                                                  num_processes=5)

    val_buffered_generator_mp = buffered_gen_mp(image_class_generator(validation_image_class_mapping,
                                                                      num_validation_classes,
                                                                      batch_size),
                                                buffer_size=20,
                                                num_processes=5)

    # # Train

    samples_per_epoch = int(len(training_image_class_mapping) / 10)
    print('sample_per_epoch:', samples_per_epoch)

    start = datetime.datetime.time(datetime.datetime.now())
    print('started at {0}'.format(start))
    fixed_history = model.fit_generator(train_buffered_generator_mp,
                                        samples_per_epoch=samples_per_epoch,
                                        nb_epoch=30,
                                        callbacks=callbacks,
                                        nb_val_samples=10000,
                                        validation_data=val_buffered_generator_mp,
                                        verbose=verbose)

    finish = datetime.datetime.time(datetime.datetime.now())
    print('finished at {0}'.format(finish))

    # # Make all layers trainable

    for layer in model.layers:
        try:
            if not layer.trainable:
                layer.trainable = True
        except:
            print('error layer is stuck:', layer.name)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # # Continue training

    callbacks = make_callbacks(
        model_checkpoint_format_string, tensor_board_log_dir)

    start = datetime.datetime.time(datetime.datetime.now())
    print('started at {0}'.format(start))

    samples_per_epoch = int(len(training_image_class_mapping) / 10)

    full_history = model.fit_generator(train_buffered_generator_mp,
                                       samples_per_epoch=samples_per_epoch,
                                       nb_epoch=250,
                                       callbacks=callbacks,
                                       nb_val_samples=10000,
                                       validation_data=val_buffered_generator_mp,
                                       verbose=verbose)

    finish = datetime.datetime.time(datetime.datetime.now())
    print('finished at {0}'.format(finish))

    save_model_workaround(model, model_output_file, weights_output_file)

    KTF.clear_session()

    return full_history
