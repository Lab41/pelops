import argparse
import collections
import datetime
import enum
import glob
import logging
import numpy as np
import os
import random
import sys
import tempfile
import tensorflow as tf

import keras.backend.tensorflow_backend as KTF
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential                              
from keras.layers import Dense, Dropout, Flatten, Input, Reshape            
from keras.preprocessing import image

import pelops.const as const
from pelops.datasets.dgcars import DGCarsDataset
from pelops.utils import SetType, setup_custom_logger

def get_session(gpu_fraction=0.3):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    print("num_threads: {}".format(num_threads))
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("retrainer")

logger.info("session number: {}".format(get_session()))

train_dir_path = "./datasets/train/" 
val_dir_path = "./datasets/test/"
train_features_path = None
val_features_path = None
dataset_type = "DGCarsDataset"
conv_model_type = "ResNet50"
conv_model_name = "ResNet50"
nb_epoch = 50
dropout_rate = 0.5
batch_size = None
seed = 11

# -----------------------------------------------------------------------------
# 1. Initialize constants 
# -----------------------------------------------------------------------------

logger.info("=" * 80)
logger.info("init state")

# initialize constants ----------------------------------------------------
# ResNet50's default size
img_height = 224
img_width = 224
img_dimension = 3

# metrics_names's indexes
index_accuracy = 1
    
# initialize random number generator --------------------------------------
np.random.seed(seed)
random.seed(seed)

# -----------------------------------------------------------------------------
# 2. Load the data
# -----------------------------------------------------------------------------


logger.info("=" * 80)
logger.info("load_data state")

train_datagen = image.ImageDataGenerator()
val_datagen = image.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    directory=train_dir_path,
    target_size=(img_height, img_width),
    seed=seed, 
    follow_links=True
)
val_generator = val_datagen.flow_from_directory(
    directory=val_dir_path,
    target_size=(img_height, img_width),
    seed=seed,
    follow_links=True
)

# train_generator and val_generator output: x (data) and y (label)
# where x.shape == (32, 224, 224, 3) and y.shape == (32, 1056)

# assumption 194 images, therefore generator will output 32 * 194 images?
logger.info("number of train chips/images: {}".format(train_generator.nb_sample))
for i in train_generator:
    x, y = i
    logger.info("shape of train x: {}, y: {}".format(x.shape, y.shape))
    break

# assumption: 151 images, therefore generator will output 32 * 151 images? 
logger.info("number of validation chips/images: {}".format(val_generator.nb_sample))

"""
batch_size = 32

def load_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def image_class_generator(train_image_classes_mapping, num_classes, batch_size=32):
    image_list = list(train_image_class_mapping.keys())
    while True:
        xs = []
        ys = []
        for filename in random.sample(image_list, batch_size):
            #print("filename: {}".format(filename))
            xs.append(load_image(filename))
            y = np.zeros(num_classes)
            #print("y.shape: {}".format(y.shape))
            #print("y: {}".format(y))
            y[train_image_class_mapping[filename]] = 1
            ys.append(y)
        yield (np.array(xs).squeeze(), np.array(ys))

train_image_classes = set()
train_image_class_mapping = {}

for image_class_filepath in glob.glob(os.path.join(train_dir_path, '*')):
    if os.path.isdir(image_class_filepath):
        #print("image_class_filepath: {}".format(image_class_filepath))
        image_class_num = int(os.path.basename(image_class_filepath)) - 1
        #print("image_class_num: {}".format(image_class_num))
        train_image_classes.add(image_class_num)
        for filename in glob.glob(os.path.join(image_class_filepath, '*')):
            #print("train_image_class_mapping[{}] = {}".format(filename, image_class_num))
            train_image_class_mapping[filename] = image_class_num

#print("len(train_image_classes): {}".format(len(train_image_classes)))

nb_classes = len(train_image_classes)
train_generator = image_class_generator(train_image_class_mapping, len(train_image_classes), batch_size)

count = 0
for i in train_generator:
    x, y = i
    count = count + 1
    logger.info("count: {}, x.shape: {}, y.shape: {}".format(count, x.shape, y.shape))
    break

val_image_classes = set()
val_image_class_mapping = {}

for image_class_filepath in glob.glob(os.path.join(val_dir_path, '*')):
    if os.path.isdir(image_class_filepath):
        #print("image_class_filepath: {}".format(image_class_filepath))
        image_class_num = int(os.path.basename(image_class_filepath)) - 1
        #print("image_class_num: {}".format(image_class_num))
        val_image_classes.add(image_class_num)
        for filename in glob.glob(os.path.join(image_class_filepath, '*')):
            #print("train_image_class_mapping[{}] = {}".format(filename, image_class_num))
            val_image_class_mapping[filename] = image_class_num


val_generator = image_class_generator(val_image_class_mapping, len(val_image_classes), batch_size)
"""

# -----------------------------------------------------------------------------
# 3. Load the convolutional model 
# -----------------------------------------------------------------------------

logger.info("=" * 80)
logger.info("load_cnn state")
logger.info("load convolutional model {} that is trained on imagenet with final connected layer removed".format(conv_model_name))

model = ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=Input(
                shape=(
                    img_height, 
                    img_width, 
                    img_dimension
                )
            )
        )

# -----------------------------------------------------------------------------
# 4. Extract features from the convolutional model based on the data
# -----------------------------------------------------------------------------

logger.info("=" * 80)
logger.info("freeze_cnn state")
logger.info("set convolutional model {} to non-trainable".format(conv_model_name))

for layer in model.layers:
    layer.trainable = False

# -----------------------------------------------------------------------------
# 5. extract features from convolutional model
# -----------------------------------------------------------------------------

def __extract_features(generator, model, batch_size, set_type):
    feature_dirpath = "./features/"
    logger.info("create a feature directory to store saved features: {}".format(feature_dirpath))
    if not os.path.exists(feature_dirpath):
        os.makedirs(feature_dirpath)

    logger.info("extract features from convolutional model based on data")
    logger.debug("generator: {}_generator".format(set_type))
    logger.debug("batch_size: {}".format(batch_size))
    features = model.predict_generator(
        generator,
        batch_size
    )

    time_now = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    features_filepath = feature_dirpath + "{}_{}_{}_features_{}.npy".format(
        dataset_type,
        conv_model_type,
        set_type,
        time_now
    )
    logger.info("save features to {}".format(features_filepath))
    np.save(open(features_filepath, "wb"), features)

    return features, features_filepath

def __load_features(feature_path):
    return np.load(open(feature_path))

logger.info("=" * 80)
logger.info("extract_features state")

batch_size = train_generator.batch_size if batch_size is None else batch_size
logger.info("set batch_size: {}".format(batch_size))

if train_features_path is not None:
    logger.info("load train features")
    train_features = __load_features(train_features_path)
else:
    logger.info("extract train features")
    train_features, train_features_path = __extract_features(train_generator, model, batch_size, "train")

if val_features_path is not None:
    logger.info("load validation features")
    val_features = __load_features(val_features_path)
else: 
    logger.info("extract validation features")
    val_features, val_features_path = __extract_features(val_generator, model, batch_size, "validation")

# -----------------------------------------------------------------------------
# 6. build the classifier model
# -----------------------------------------------------------------------------

logger.info("=" * 80)
logger.info("build_classifier state")
logger.info("classifier model will be put on top of the convolutional model")

# think of features as inputs 
# model.output_shape = [1, nb_samples, nb_steps, nb_features (aka input_dimensions)]
nb_features = model.output_shape[-1] # same as train_features.shape[-1]
logger.info("input shape: {}".format(train_features.shape))
logger.info("input size: {} == {}".format(model.output_shape[-1], train_features.shape[-1]))

# think of classes as outputs
nb_classes = train_generator.nb_class
logger.info("output size: {}".format(nb_classes))

# taking the mean of the input and output size to get the hidden layer size
nb_hidden_layers = int(round(np.mean([nb_features, nb_classes])))
logger.info("hidden layer size: {}".format(nb_hidden_layers))

logger.info("{} -> [hidden layer {}] -> {}\n".format(nb_features, nb_hidden_layers, nb_classes))

# use sequential instead of computational graph as our model
top_model = Sequential()
#top_model.add(Flatten(input_shape=train_features.shape[1:]))
# add a non-linear function: recitified linear unit (relu)
top_model.add(Dense(nb_hidden_layers, activation="relu", input_shape=train_features.shape[1:]))
#top_model.add(Dense(nb_hidden_layers, activation="relu", input_shape=(None, None, nb_features)))
# dropout helps to protect the model from memorizing or overfitting the training data
top_model.add(Dropout(dropout_rate))
# "softmax" ensures the output is a valid probability distribution
# that is, the values are all non-negative and sum to 1
# think of softmax as a multi-class sigmoid
#top_model.add(Flatten())
top_model.add(Dense(nb_classes, activation="softmax")) 
#top_model.add(Dense(nb_classes, activation="softmax", input_shape=(None, None, nb_hidden_layers)))
top_model = top_model

logger.info("convolutional model: input shape = {}, output shape = {}".format(model.input_shape, model.output_shape))
logger.info("classifier model: input shape = {}, output shape = {}".format(top_model.input_shape, top_model.output_shape))

which_model = "classifier"

# -----------------------------------------------------------------------------
# 7. compile the classifier model
# -----------------------------------------------------------------------------

logger.info("=" * 80)
logger.info("compile_model state: compile {}".format("classifier"))

top_model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# -----------------------------------------------------------------------------
# 8. create checkpoints in classifier model to save network weights
# -----------------------------------------------------------------------------

logger.info("=" * 80)
logger.info("checkpoint_classifier state")
logger.info("create checkpoints in {} model to save network weights".format(which_model))

checkpoint_dirpath = "./checkpoints/"
logger.info("create a checkpoint directory to store saved checkpoints: {}".format(checkpoint_dirpath))
if not os.path.exists(checkpoint_dirpath):
    os.makedirs(checkpoint_dirpath)

checkpoint_filepath = \
    checkpoint_dirpath + \
    "{}_{}_features_".format(dataset_type, which_model) + \
    "{epoch:02d}_{val_acc:.2f}.npy"

checkpoint = ModelCheckpoint(
    checkpoint_filepath, 
    monitor="val_acc", 
    save_best_only=True, 
    mode="max"
)
callbacks_list = [checkpoint]

# -----------------------------------------------------------------------------
# 9. train the classifier model using features
# -----------------------------------------------------------------------------

def __create_generator_from_features(features, generator):
    for feature, (x, y) in zip(features, generator):
        yield (feature, y)

logger.info("=" * 80)
logger.info("train_model state: train {}".format(which_model))

"""
train_features_generator = __create_generator_from_features(train_features, train_generator)

logger.info("samples_per_epoch: {}".format(len(train_features)))

top_model.fit_generator(
    generator=train_features_generator, 
    samples_per_epoch=len(train_features),
    nb_epoch=1,
    callbacks=callbacks_list,
    verbose=2,
)
"""

labels = []
for i in train_generator:
    x, y = i
    labels.append(y)
labels = np.array(labels)
logger.debug("labels: {}".format(labels.shape))

top_model.fit(
    x=train_features,
    y=labels,
)

weight_dirpath = "./weights/"
logger.info("create a weights directory to store saved weights: {}".format(weight_dirpath))
if not os.path.exists(weight_dirpath):
    os.makedirs(weight_dirpath)

time_now = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
weight_filepath = weight_dirpath + "{}_{}_weights_{}.npy".format(
    dataset_type,
    which_model,
    time_now
)
logger.info("save weights to {}".format(weight_filepath))
top_model.save_weights(weight_filepath)


logger.info("evaluate how well the {} model performs:".format(which_model))
score = top_model.evaluate_generator(
    val_data,
    val_samples=val_data.nb_sample
)
logger.info("{}: {}".format(
    top_model.metrics_names[index_accuracy],
    score[index_accuracy]
))

# -----------------------------------------------------------------------------
# 10. add the classifier model on top of the convolutional model
# -----------------------------------------------------------------------------

logger.info("=" * 80)
logger.info("unfreeze_cnn state")
logger.info("set convolutional model {} to trainable".format(conv_model_name))

for layer in model.layers:
    layer.trainable = True

logger.info("=" * 80)
logger.info("combine_models state")
logger.info("add the classifier model on top of the convolutional model")
logger.info("combined model == (convolutional + classifier) model ")

model.add(top_model)
which_model = "combined"

# -----------------------------------------------------------------------------
# 11. compile the combined models 
# -----------------------------------------------------------------------------

logger.info("=" * 80)
logger.info("compile_model state: compile {}".format(which_model))

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# -----------------------------------------------------------------------------
# 12. train the combined models
# -----------------------------------------------------------------------------

logger.info("=" * 80)
logger.info("train_model state: train {}".format(which_model))


top_model.fit_generator(
    generator=train_generator, 
    samples_per_epoch=train_generator.nb_sample,
    nb_epoch=1,
    callbacks=callbacks_list,
    verbose=2,
)

weight_dirpath = "./weights/"
logger.info("create a weights directory to store saved weights: {}".format(weight_dirpath))
if not os.path.exists(weight_dirpath):
    os.makedirs(weight_dirpath)

time_now = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
weight_filepath = weight_dirpath + "{}_{}_weights_{}.npy".format(
    dataset_type,
    which_model,
    time_now
)
logger.info("save weights to {}".format(weight_filepath))
top_model.save_weights(weight_filepath)


logger.info("evaluate how well the {} model performs:".format(which_model))
score = top_model.evaluate_generator(
    val_data,
    val_samples=val_data.nb_sample
)
logger.info("{}: {}".format(
    top_model.metrics_names[index_accuracy],
    score[index_accuracy]
))






