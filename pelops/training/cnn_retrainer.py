import argparse
import datetime
import enum
import logging
import numpy as np
import os
import random
import sys
import tempfile
import tensorflow as tf

from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential, load_model, save_model                              
from keras.layers import Dense, Dropout, Flatten, Input, Reshape            
from keras.preprocessing import image

import pelops.const as const
from pelops.datasets.dgcars import DGCarsDataset
from pelops.utils import SetType, setup_custom_logger

# =============================================================================
# Allocate GPU memory so the program does not take over the entire GPU memory
# =============================================================================

def __get_session(gpu_fraction=0.3):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    const.logger.info("num_threads: {}".format(num_threads))
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# =============================================================================
# Define convolutional model types
# =============================================================================

class ConvType(enum.Enum):
    ResNet50 = "ResNet50"

# =============================================================================
# Helper functions
# =============================================================================

def __get_resnet50(include_top):
    return ResNet50(
            include_top=include_top,
            weights="imagenet",
            input_tensor=Input(
                shape=(
                    const.img_height, 
                    const.img_width, 
                    const.img_dimension
                )
            )
    )

def __extract_features(model, generator, set_type):
    feature_dirpath = "./features/"
    const.logger.info("create a feature directory to store saved features: {}".format(feature_dirpath))
    if not os.path.exists(feature_dirpath):
        os.makedirs(feature_dirpath)

    const.logger.info("extract features from cnn")
    const.logger.debug("batch_size: {}".format(generator.batch_size))
    features = model.predict_generator(
        generator,
        generator.batch_size
    )

    time_now = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    features_filepath = feature_dirpath + "{}_{}_{}_features_{}.npy".format(
        const.dataset_type,
        const.conv_model_type,
        set_type,
        time_now
    )
    const.logger.info("save features to {}".format(features_filepath))
    np.save(open(features_filepath, "wb"), features)

    return features, features_filepath

def __load_features(feature_filepath):
    return np.load(open(feature_filepath, "rb"))

def __create_checkpoints(which_model):
    checkpoint_dirpath = "./checkpoints/"
    const.logger.info("create a checkpoint directory to store saved checkpoints: {}".format(checkpoint_dirpath))
    if not os.path.exists(checkpoint_dirpath):
        os.makedirs(checkpoint_dirpath)

    checkpoint_filepath = \
        checkpoint_dirpath + \
        "{}_{}_best_checkpoint_".format(const.dataset_type, which_model) + \
        "{epoch:02d}_{val_acc:.4f}.npy"

    checkpoint = ModelCheckpoint(
        filepath=checkpoint_filepath, 
        monitor="val_acc", 
        save_best_only=True, 
        mode="max",
        save_weights_only=False,
        verbose=2,
    )
    
    return [checkpoint]

def __create_generator_from_features(features, generator):

    nb_sample = generator.nb_sample
    nb_class = generator.nb_class
    batch_size = generator.batch_size

    # create labels for the features
    count = 0
    labels = np.zeros((batch_size, nb_class))
    for i, class_index in zip(range(0, batch_size), generator.classes):
        labels[i][class_index] = 1

    # create generator with features and labels
    while True:
        for i in range(int(nb_sample/batch_size)):
            x = features[i * batch_size: (i+1) * batch_size]
            y = labels[i * batch_size: (i+1) * batch_size]
            yield x, y

def __get_weights_filepath(which_model):
    weight_dirpath = "./weights/"
    const.logger.info("create a weights directory to store saved weights: {}".format(weight_dirpath))
    if not os.path.exists(weight_dirpath):
        os.makedirs(weight_dirpath)

    time_now = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    weight_filepath = weight_dirpath + "{}_{}_weights_{}.npy".format(
        const.dataset_type,
        which_model,
        time_now
    )

    const.logger.info("save weights to {}".format(weight_filepath))
    return weight_filepath

# =============================================================================
# Main
# =============================================================================

def main(args):
    # setup logging
    const.logger = setup_custom_logger(__name__)

    # allocate gpu memory
    # TODO: causes memory error when run 
    #session_number = __get_session()
    #const.logger.info("allocate GPU memory, session number: {}".format(session_number))

    # extract arguments from command line
    const.train_dir_path = args.train_dir_path
    const.val_dir_path = args.val_dir_path
    const.train_features_path = args.train_features_path
    const.val_features_path = args.val_features_path
    const.dataset_type = args.dataset_type
    const.conv_model_type = args.conv_model_type
    const.nb_epoch = args.nb_epoch
    const.dropout_rate = args.dropout_rate
    const.seed = args.seed

    # initialize constants
    if const.conv_model_type == ConvType.ResNet50.value: 
        # ResNet50's default size
        const.img_height = 224
        const.img_width = 224
        const.img_dimension = 3
        # metrics_name's indexes
        const.index_accuracy = 1

    # initialize random number generator 
    np.random.seed(const.seed)
    random.seed(const.seed)

    # -------------------------------------------------------------------------
    # 1. load the data
    # -------------------------------------------------------------------------

    const.logger.info("=" * 80 + "\n1. load data")

    train_datagen = image.ImageDataGenerator()
    val_datagen = image.ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        directory=const.train_dir_path,
        target_size=(const.img_height, const.img_width),
        seed=const.seed, 
        follow_links=True
    )
    val_generator = val_datagen.flow_from_directory(
        directory=const.val_dir_path,
        target_size=(const.img_height, const.img_width),
        seed=const.seed,
        follow_links=True
    )

    # train_generator and val_generator output: x (data) and y (label)
    # where x.shape == (32, 224, 224, 3) and y.shape == (32, 1056)

    const.logger.debug("TRAINING: number of classes: {}".format(train_generator.nb_class))
    const.logger.debug("TRAINING: number of images: {}".format(train_generator.nb_sample))
    for i in train_generator:
        x, y = i
        const.logger.debug("TRAINING: shape of x: {}, y: {}".format(x.shape, y.shape))
        break

    const.logger.debug("VALIDATION: number of classes: {}".format(val_generator.nb_class))
    const.logger.debug("VALIDATION: number of images: {}".format(val_generator.nb_sample))

    # -------------------------------------------------------------------------
    # 2. load the convolutional model
    # -------------------------------------------------------------------------

    const.logger.info("=" * 80 + "\n2. load conv: {}".format(const.conv_model_type))

    model = {
        ConvType.ResNet50.value: __get_resnet50(include_top=False)
    }.get(const.conv_model_type)

    # -------------------------------------------------------------------------
    # 3. set the convolutional model to non-trainable
    # -------------------------------------------------------------------------

    const.logger.info("=" * 80 + "\n3. freeze conv")

    for layer in model.layers: 
        layer.trainable=False

    # -------------------------------------------------------------------------
    # 4. extract features from the convolutional model
    # -------------------------------------------------------------------------

    const.logger.info("=" * 80 + "\n4. extract features from conv")

    if const.train_features_path is not None:
        const.logger.info("TRAINING: load features")
        train_features = __load_features(const.train_features_path)
    else:
        const.logger.info("TRAINING: extract features")
        train_features, train_features_path = __extract_features(model, train_generator, "train")

    if const.val_features_path is not None:
        const.logger.info("VALIDATION: load features")
        val_features = __load_features(const.val_features_path)
    else: 
        const.logger.info("VALIDATION: extract features")
        val_features, val_features_path = __extract_features(model, val_generator, "validation")

    # -------------------------------------------------------------------------
    # 5. build the classifier model
    # -------------------------------------------------------------------------

    const.logger.info("=" * 80 + "\n5. build classifier")

    # think of features as inputs 
    # model.output_shape = [1, nb_samples, nb_steps, nb_features (aka input_dimensions)]
    nb_features = model.output_shape[-1] # same as train_features.shape[-1]

    # think of classes as outputs
    nb_classes = train_generator.nb_class

    # taking the mean of the input and output size to get the hidden layer size
    nb_hidden_layers = int(round(np.mean([nb_features, nb_classes])))

    const.logger.info("{} -> [hidden layer {}] -> {}\n".format(nb_features, nb_hidden_layers, nb_classes))

    # use sequential instead of computational graph as our model
    top_model = Sequential()
    # add a non-linear function: recitified linear unit (relu)
    top_model.add(Dense(nb_hidden_layers, activation="relu", input_shape=train_features.shape[1:]))
    # dropout helps to protect the model from memorizing or overfitting the training data
    #top_model.add(Dropout(const.dropout_rate))
    # flatten 3D to 1D
    top_model.add(Flatten())
    # "softmax" ensures the output is a valid probability distribution
    # that is, the values are all non-negative and sum to 1
    # think of softmax as a multi-class sigmoid
    top_model.add(Dense(nb_classes, activation="softmax")) 

    # -------------------------------------------------------------------------
    # 6. compile the classifier model
    # -------------------------------------------------------------------------

    const.logger.info("=" * 80 + "\n6. compile classifier")

    top_model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    # -------------------------------------------------------------------------
    # 7. create checkpoints to save classifier model's  weights
    # -------------------------------------------------------------------------

    const.logger.info("=" * 80 + "\n7. create checkpoints to save classifier's weights")

    callbacks_list = __create_checkpoints("classifier")

    # -------------------------------------------------------------------------
    # 8. train the classifier model using features
    # -------------------------------------------------------------------------
    
    const.logger.info("=" * 80 + "\n8. train classifier using features")

    top_model.fit_generator(
        generator=__create_generator_from_features(train_features, train_generator),
        samples_per_epoch=train_generator.batch_size,
        nb_epoch=const.nb_epoch,
        callbacks=callbacks_list,
        validation_data=__create_generator_from_features(val_features, val_generator),
        nb_val_samples=val_generator.batch_size,
        verbose=2
    )

    top_model.save_weights(__get_weights_filepath("classifier"))


    # -------------------------------------------------------------------------
    # 9. evaluate the classifier model
    # -------------------------------------------------------------------------
    
    const.logger.info("=" * 80 + "\n9. evaluate classifier")

    const.logger.info("extract labels for validation features")
    val_labels = np.zeros((val_features.shape[0], val_generator.nb_class))
    for i, class_index in zip(range(0, val_features.shape[0]), val_generator.classes):
        val_labels[i][class_index] = 1

    const.logger.info("evaluate classifier with validation features and labels")
    score = top_model.evaluate(
        x=val_features,
        y=val_labels,
        batch_size=val_generator.batch_size,
    )

    const.logger.info("{}: {}".format(
        top_model.metrics_names[const.index_accuracy],
        score[const.index_accuracy]
    ))

    # -------------------------------------------------------------------------
    # 10. set the convolutional model to trainable
    # -------------------------------------------------------------------------

    const.logger.info("=" * 80 + "\n10. unfreeze conv")

    for layer in model.layers: 
        layer.trainable=True

    # -------------------------------------------------------------------------
    # 11. combine the convolutional model and classifier model
    # -------------------------------------------------------------------------

    const.logger.info("=" * 80 + "\n11. add classifier on top of conv == combined")

    top_model = load_model("./top_model.h5")
    model = Model(input=model.input, output=top_model(model.output))

    # -------------------------------------------------------------------------
    # 12. compile the combined model
    # -------------------------------------------------------------------------

    const.logger.info("=" * 80 + "\n12. compile combined")

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    # -------------------------------------------------------------------------
    # 13. create checkpoints to save classifier model's  weights
    # -------------------------------------------------------------------------

    const.logger.info("=" * 80 + "\n7. create checkpoints to save classifier's weights")

    callbacks_list = __create_checkpoints("combined")

    # -------------------------------------------------------------------------
    # 14. train the combined model
    # -------------------------------------------------------------------------

    const.logger.info("=" * 80 + "\n13. train combined using data")

    model.fit_generator(
        generator=train_generator,
        samples_per_epoch=train_generator.batch_size,
        nb_epoch=const.nb_epoch, 
        callbacks=callbacks_list,
        validation_data=val_generator,
        nb_val_samples=val_generator.nb_sample,
        verbose=2
    )

    model.save_weights(__get_weights_filepath("combined"))

    # -------------------------------------------------------------------------
    # 15. evaluate the combined model
    # -------------------------------------------------------------------------

    const.logger.info("=" * 80 + "\n14. evaluate combined")

    score = model.evaluate_generator(
        generator=val_generator,
        val_samples=val_generator.nb_sample
    )

    const.logger.info("{}: {}".format(
        model.metrics_names[1],
        score[1]
    ))

# =============================================================================
# Start here
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="cnn_retrainer.py", 
        description="Generate a convolutional neural network retrained on car images", 
        formatter_class=argparse.RawTextHelpFormatter
    )
    # arguments
    parser.add_argument("train_dir_path", default="train_dir_path", action="store", type=str,
        help="Specify the directory where training dataset lies.")
    parser.add_argument("val_dir_path", default="val_dir_path", action="store", type=str,
        help="Specify the directory where validation dataset lies.")
    # options
    parser.add_argument("-v", "--version", action="version",
        version="CNN Retrainer 1.0")
    parser.add_argument("--train_features_path", dest="train_features_path", action="store", type=str,
        default=None,
        help="Specify the file that contains the train features.")
    parser.add_argument("--val_features_path", dest="val_features_path", action="store", type=str,
        default=None,
        help="Specify the file that contains the validation features.")
    parser.add_argument("-w", dest="dataset_type", action="store", choices=["DGCarsDataset"], type=str,
        default="DGCarsDataset",
        help="Specify the datasets to use.")
    parser.add_argument("-c", dest="conv_model_type", action="store", choices=["ResNet50"], type=str,
        default="ResNet50",
        help="Specify the convolutional model to use.")
    parser.add_argument("-e", dest="nb_epoch", action="store", type=int, 
        default=10,
        help="Specify epoch, the total number of iterations on the data.")
    parser.add_argument("-d", dest="dropout_rate", action="store", type=float,
        default=0.5,
        help="Specify dropout rate. Dropout helps model from not memorizing or overfitting the data.")
    parser.add_argument("-s", dest="seed", action="store", type=int, 
        default=random.randint(1, 100),
        help="(OPTIONAL) SEED is used for random number generator.")
    main(parser.parse_args())