#!/usr/bin/env python3
# coding: utf-8

from keras import optimizers
from keras.applications.resnet50 import ResNet50
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.models import Sequential, Model
from keras.preprocessing import image as image_utils
from keras.callbacks import ModelCheckpoint
import numpy as np
from pelops.training.utils import attributes_to_classes
from pelops.datasets.dgcars import DGCarsDataset
from pelops.utils import SetType

INPUT_HEIGHT = 224
INPUT_WIDTH = 224
N_CLASSES = 1056  # Make and model
N_IMAGES = 836555 # Number of images in the training set

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

# Load data for fitting parameters
x_train = []
data_train = DGCarsDataset("/path/to/dgCars/dataset/", set_type=SetType.TRAIN.value)
i = 0
for chip in data_train:
    image = image_utils.load_img(chip.filepath, target_size=(INPUT_HEIGHT, INPUT_WIDTH))
    img_arr = image_utils.img_to_array(image)
    x_train.append(img_arr)
    if i > 15000:
        break

x_train = np.array(x_train)

training_data = train_generator.flow_from_directory(
    directory="/path/to/dgCars/make_model/train/",
    target_size=(INPUT_HEIGHT, INPUT_WIDTH),
    class_mode="categorical",
    batch_size=128,
    follow_links=True,
)


val_data = val_generator.flow_from_directory(
    directory="/path/to/dgCars/make_model/test/",
    target_size=(INPUT_HEIGHT, INPUT_WIDTH),
    class_mode="categorical",
    batch_size=128,
    follow_links=True,
)

# Fit the distribution of pixels in the images so we can whiten, etc.
training_data.fit(x_train)
val_data.fit(x_train)  # TODO use test data

# Load Resnet50
inputs = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3))
conv_model = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)  # TF ordering

conv_output = conv_model.output

# Freeze the conv layers
for layer in conv_model.layers:
    layer.trainable = False

# build a classifier model to put on top of the convolutional model
#INPUT_SIZE = [int(i) for i in conv_output.get_shape()[1:]]

x = Flatten()(conv_output)
#x = Dense(N_CLASSES * 2, activation='relu')(x)
#x = Dropout(0.5)(x)
output = Dense(N_CLASSES, activation='softmax')(x)


combined_model = Model(input=inputs, output=output)

combined_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    metrics=['accuracy']
)

# Set up checkpoint
filepath="/path/to/resnet50_dgcars_retrain_weights_improvement_{epoch:02d}_{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

# Train
combined_model.fit_generator(
    training_data,
    samples_per_epoch=N_IMAGES,  # set to a smaller number to debug
    nb_epoch=1000,
    callbacks=callbacks_list,
    validation_data=val_data,
    nb_val_samples=128*100,
)
