#!/usr/bin/env python3
# coding: utf-8

from keras import optimizers
from keras.applications.resnet50 import ResNet50
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.models import Sequential, Model, load_model
from keras.preprocessing import image as image_utils
from keras.callbacks import ModelCheckpoint
from pelops.datasets.compcar import CompcarDataset
from pelops.utils import SetType
import numpy as np
from pelops.training.utils import attributes_to_classes, key_make_model

INPUT_HEIGHT = 224
INPUT_WIDTH = 224

cc_train = CompcarDataset("/path/to/compcar/dataset", set_type=SetType.TRAIN.value)
#cc_test = CompcarDataset("/path/to/compcar/dataset", set_type=SetType.TEST.value)

# Map make and model to an index
label_to_index = attributes_to_classes(cc_train, key_make_model)

i = 0
x_train = []
y_train = []
x_val = []
y_val = []
for chip in cc_train:
    key = key_make_model(chip)
    image = image_utils.load_img(chip.filepath, target_size=(INPUT_HEIGHT, INPUT_WIDTH))
    img_arr = image_utils.img_to_array(image)

    i += 1
    if i < 15000:
        x_train.append(img_arr)
        y_train.append(label_to_index[key])
    if 15000 < i < 18000:
        x_val.append(img_arr)
        y_val.append(label_to_index[key])
    if i > 18000:
        break


x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)

# Set up a data generator for Keras
train_generator = image_utils.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    zca_whitening=False,
    rotation_range=10,  # Degrees
    width_shift_range=.25,
    height_shift_range=.25,
    shear_range=0.1,  # Radians
    zoom_range=.30,
    horizontal_flip=True,
)

train_generator.fit(x_train)

val_generator = image_utils.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    zca_whitening=False,
    rotation_range=10,  # Degrees
    width_shift_range=.25,
    height_shift_range=.25,
    shear_range=0.1,  # Radians
    zoom_range=.30,
    horizontal_flip=True,
)

val_generator.fit(x_val)

# Load model
#combined_model = load_model("/path/to/model.hdf5")
#combined_model = load_model("/path/to/model.hdf5")
combined_model = load_model("/path/to/model.hdf5")

#combined_model.compile(
#    loss='sparse_categorical_crossentropy',
#    optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
#    metrics=['accuracy']
#)

# Set up checkpoint
filepath="/path/to/resnet50_compcars_retrain_weights_improvement_{epoch:02d}_{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False)
callbacks_list = [checkpoint]

# Train
combined_model.fit_generator(
    train_generator.flow(x_train, y_train, batch_size=128),
    samples_per_epoch=len(x_train),
    nb_epoch=10,
    callbacks=callbacks_list,
    validation_data=val_generator.flow(x_val, y_val, batch_size=128),
    nb_val_samples=512,
)
