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

# =============================================================================
# Allocate GPU memory so Tensorflow does not take the entire GPU memory
# =============================================================================

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

# =============================================================================
# State Machine
# =============================================================================

class StateMachine():
    def __init__(self):
        self.handlers = {}
        self.start_state = None
        self.end_states = []

    def add_state(self, name, handler, end_state=False):
        self.handlers[name] = handler
        if end_state: 
            self.end_states.append(name)

    def set_start(self, name):
        self.start_state = name

    def run(self, cargo):
        try:
            handler = self.handlers[self.start_state]
        except:
            raise InitializationError("ERROR in StateMachine: must set the start state first")

        if not self.end_states: 
            raise InitializationError("ERROR in StateMachine: at least one state must be an end state")

        while True:
            new_state, cargo = handler(cargo)
            if new_state in self.end_states:
                break
            else:
                handler = self.handlers[new_state]

# =============================================================================
# States of State Machine
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Initialize constants 
# -----------------------------------------------------------------------------

def init(cargo):
    const.logger.info("=" * 80)
    const.logger.info("init state")

    # initialize constants ----------------------------------------------------
    if const.conv_model_type == ConvType.ResNet50.value: 
        # ResNet50's default size
        const.img_height = 224
        const.img_width = 224
        const.img_dimension = 3

    # metrics_names's indexes
    const.index_accuracy = 1
    
    # initialize random number generator --------------------------------------
    np.random.seed(const.seed)
    random.seed(const.seed)

    return ("load_data", cargo)

# -----------------------------------------------------------------------------
# 2. Load the data
# -----------------------------------------------------------------------------

def load_data(cargo):
    const.logger.info("=" * 80)
    const.logger.info("load_data state")

    train_datagen = image.ImageDataGenerator()
    val_datagen = image.ImageDataGenerator()

    cargo.train_generator = train_datagen.flow_from_directory(
        directory=const.train_dir_path,
        target_size=(const.img_height, const.img_width),
        seed=const.seed, 
        follow_links=True
    )
    cargo.val_generator = val_datagen.flow_from_directory(
        directory=const.val_dir_path,
        target_size=(const.img_height, const.img_width),
        seed=const.seed,
        follow_links=True
    )

    # train_generator and val_generator output: x (data) and y (label)
    # where x.shape == (32, 224, 224, 3) and y.shape == (32, 1056)

    const.logger.info("number of train chips/images: {}".format(cargo.train_generator.nb_sample))
    for i in cargo.train_generator:
        x, y = i
        const.logger.info("shape of train x: {}, y: {}".format(x.shape, y.shape))
        break

    const.logger.info("number of validation chips/images: {}".format(cargo.val_generator.nb_sample))

    return ("load_cnn", cargo)

# -----------------------------------------------------------------------------
# 3. Load the convolutional model 
# -----------------------------------------------------------------------------

def load_cnn(cargo):
    const.logger.info("=" * 80)
    const.logger.info("load_cnn state")
    const.logger.info("load convolutional model {} that is trained on imagenet with final connected layer removed".format(const.conv_model_name))

    def __get_resnet50():
        return ResNet50(
                include_top=False,
                weights="imagenet",
                input_tensor=Input(
                    shape=(
                        const.img_height, 
                        const.img_width, 
                        const.img_dimension
                    )
                )
        )


    cargo.model = {
        ConvType.ResNet50.value: __get_resnet50()
    }.get(const.conv_model_type)

    return ("freeze_cnn", cargo)

# -----------------------------------------------------------------------------
# 4. Extract features from the convolutional model based on the data
# -----------------------------------------------------------------------------

def freeze_cnn(cargo):
    const.logger.info("=" * 80)
    const.logger.info("freeze_cnn state")
    const.logger.info("set convolutional model {} to non-trainable".format(const.conv_model_name))

    for layer in cargo.model.layers:
        layer.trainable = False

    return("extract_features", cargo)

# -----------------------------------------------------------------------------
# 5. extract features from convolutional model
# -----------------------------------------------------------------------------

def extract_features(cargo):
    const.logger.info("=" * 80)
    const.logger.info("extract_features state")

    cargo.batch_size = cargo.train_generator.batch_size if const.batch_size is None else const.batch_size
    const.logger.info("set batch_size: {}".format(cargo.batch_size))

    if const.train_features_path is not None:
        const.logger.info("load train features")
        cargo.train_features = __load_features(const.train_features_path)
    else:
        const.logger.info("extract train features")
        cargo.train_features, cargo.train_features_path = __extract_features(cargo.train_generator, cargo.model, cargo.batch_size, "train")

    if const.val_features_path is not None:
        const.logger.info("load validation features")
        cargo.val_features = __load_features(const.val_features_path)
    else: 
        const.logger.info("extract validation features")
        cargo.val_features, cargo.val_features_path = __extract_features(cargo.val_generator, cargo.model, cargo.batch_size, "validation")

    return ("build_classifier", cargo)

def __extract_features(generator, model, batch_size, set_type):
    feature_dirpath = "./features/"
    const.logger.info("create a feature directory to store saved features: {}".format(feature_dirpath))
    if not os.path.exists(feature_dirpath):
        os.makedirs(feature_dirpath)

    const.logger.info("extract features from convolutional model based on data")
    const.logger.debug("generator: {}_generator".format(set_type))
    const.logger.debug("batch_size: {}".format(batch_size))
    features = model.predict_generator(
        generator,
        batch_size
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

def __load_features(feature_path):
    return np.load(open(feature_path))

# -----------------------------------------------------------------------------
# 6. build the classifier model
# -----------------------------------------------------------------------------

def build_classifier(cargo):
    const.logger.info("=" * 80)
    const.logger.info("build_classifier state")
    const.logger.info("classifier model will be put on top of the convolutional model")
    
    # think of features as inputs 
    # model.output_shape = [1, nb_samples, nb_steps, nb_features (aka input_dimensions)]
    nb_features = cargo.model.output_shape[-1] # same as cargo.train_features.shape[-1]
    const.logger.info("input shape: {}".format(cargo.train_features.shape))
    const.logger.info("input size: {} == {}".format(cargo.model.output_shape[-1], cargo.train_features.shape[-1]))

    # think of classes as outputs
    nb_classes = cargo.train_generator.nb_class
    const.logger.info("output size: {}".format(nb_classes))

    # taking the mean of the input and output size to get the hidden layer size
    nb_hidden_layers = int(round(np.mean([nb_features, nb_classes])))
    const.logger.info("hidden layer size: {}".format(nb_hidden_layers))

    const.logger.info("{} -> [hidden layer {}] -> {}\n".format(nb_features, nb_hidden_layers, nb_classes))

    # use sequential instead of computational graph as our model
    top_model = Sequential()
    #top_model.add(Flatten(input_shape=cargo.train_features.shape[1:]))
    # add a non-linear function: recitified linear unit (relu)
    top_model.add(Dense(nb_hidden_layers, activation="relu", input_shape=cargo.train_features.shape[1:]))
    # dropout helps to protect the model from memorizing or overfitting the training data
    top_model.add(Dropout(const.dropout_rate))
    # "softmax" ensures the output is a valid probability distribution
    # that is, the values are all non-negative and sum to 1
    # think of softmax as a multi-class sigmoid
    top_model.add(Dense(nb_classes, activation="softmax")) 
    cargo.top_model = top_model

    const.logger.info("convolutional model: input shape = {}, output shape = {}".format(cargo.model.input_shape, cargo.model.output_shape))
    const.logger.info("classifier model: input shape = {}, output shape = {}".format(cargo.top_model.input_shape, cargo.top_model.output_shape))

    # define which model to compile -> checkpoint -> train
    cargo.which_model = "classifier"

    return ("compile_model", cargo)

# -----------------------------------------------------------------------------
# 7. compile the classifier model
# 12. compile the combined models 
# -----------------------------------------------------------------------------

def compile_model(cargo):
    const.logger.info("=" * 80)
    const.logger.info("compile_model state: compile {}".format(cargo.which_model))

    model = __select_which_model(cargo)

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    next_state = "checkpoint_model" if cargo.which_model == "classifier" else "train_combined_model"
    return ("checkpoint_model", cargo)

# -----------------------------------------------------------------------------
# 8. create checkpoints in classifier model to save network weights
# -----------------------------------------------------------------------------

def checkpoint_model(cargo):
    const.logger.info("=" * 80)
    const.logger.info("checkpoint_classifier state")
    const.logger.info("create checkpoints in {} model to save network weights".format(cargo.which_model))

    checkpoint_dirpath = "./checkpoints/"
    const.logger.info("create a checkpoint directory to store saved checkpoints: {}".format(checkpoint_dirpath))
    if not os.path.exists(checkpoint_dirpath):
        os.makedirs(checkpoint_dirpath)

    checkpoint_filepath = \
        checkpoint_dirpath + \
        "{}_{}_features_".format(const.dataset_type, cargo.which_model) + \
        "{epoch:02d}_{val_acc:.2f}.npy"

    checkpoint = ModelCheckpoint(
        checkpoint_filepath, 
        monitor="val_acc", 
        save_best_only=True, 
        mode="max"
    )
    cargo.callbacks_list = [checkpoint]

    return ("train_model", cargo)

# -----------------------------------------------------------------------------
# 9. train the classifier model using features
# 13. train the combined models
# -----------------------------------------------------------------------------

def train_model(cargo):
    const.logger.info("=" * 80)
    const.logger.info("train_model state: train {}".format(cargo.which_model))

    model = __select_which_model(cargo)
    generator = __select_which_generator_data(cargo)

    const.logger.debug(cargo.train_features.shape)

    """
    count = 0
    for i in generator:
        x, y = i
        count = count + 1
        const.logger.debug("count: {}, x: {}, y: {}".format(count, x.shape, y.shape))
        if x is None: const.logger.debug("x is none")
        if y is None: const.logger.debug("y is none")

    model.fit_generator(
        generator=generator,
        samples_per_epoch=2, # number of images to process before going to next epoch
        nb_epoch=10,
        callbacks=cargo.callbacks_list,
        verbose=2,
    )
    """

    labels = []
    for i in cargo.train_generator:
        x, y = i
        labels.append(y)


    labels = np.array(labels)
    const.logger.debug("labels: {}".format(labels.shape))

    count = 0
    for x, y in zip(cargo.train_features, labels):
        count = count + 1
        const.logger.debug("count: {}, x: {}, y: {}".format(count, x.shape, y.shape))


    model.fit(
        x=cargo.train_features,
        y=labels,
        batch_size=2,
        nb_epoch=10,
        verbose=2,
    )

    weight_dirpath = "./weights/"
    const.logger.info("create a weights directory to store saved weights: {}".format(weight_dirpath))
    if not os.path.exists(weight_dirpath):
        os.makedirs(weight_dirpath)

    time_now = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    weight_filepath = weight_dirpath + "{}_{}_weights_{}.npy".format(
        const.dataset_type,
        const.which_model,
        time_now
    )
    const.logger.info("save weights to {}".format(weight_filepath))
    model.save_weights(weight_filepath)

    return("happy_ending", None)

    const.logger.info("evaluate how well the {} model performs:".format(cargo.which_model))
    score = model.evaluate_generator(
        val_data,
        val_samples=val_data.nb_sample
    )
    const.logger.info("{}: {}".format(
        model.metrics_names[const.index_accuracy],
        score[const.index_accuracy]
    ))

    # determine what is the next state
    next_state, next_cargo = ("unfreeze_cnn", cargo) if cargo.which_model == "classifier" else ("happy_ending", None)
    return (next_state, next_cargo)

def __create_generator_from_features(features, generator):
    for feature, (x, y) in zip(features, generator):
        yield (feature, y)
        #yield (feature.flatten(), y)

# -----------------------------------------------------------------------------
# 10. unfreeze the convolutional model
# -----------------------------------------------------------------------------

def unfreeze_cnn(cargo):
    const.logger.info("=" * 80)
    const.logger.info("unfreeze_cnn state")
    const.logger.info("set convolutional model {} to trainable".format(const.conv_model_name))

    for layer in cargo.model.layers:
        layer.trainable = True

    return("combine_models", cargo)

# -----------------------------------------------------------------------------
# 11. add the classifier model on top of the convolutional model
# -----------------------------------------------------------------------------

def combine_models(cargo):
    const.logger.info("combine_models state")
    const.logger.info("add the classifier model on top of the convolutional model")
    const.logger.info("combined model == (convolutional + classifier) model ")

    cargo.model.add(cargo.top_model)
    cargo.which_model = "combined"

    return("compile_model", cargo)

# =============================================================================
# Helper functions
# =============================================================================

def __select_which_model(cargo):
    return {
        "cnn": cargo.model,
        "classifier": cargo.top_model,
        "combined": cargo.model
    }.get(cargo.which_model)

def __select_which_generator_data(cargo):
    return {
        "cnn": cargo.train_generator,
        "classifier": __create_generator_from_features(cargo.train_features, cargo.train_generator),
        "combined": cargo.train_generator,
    }.get(cargo.which_model)


# =============================================================================
# Define convolutional model types
# =============================================================================

class ConvType(enum.Enum):
    ResNet50 = "ResNet50"

# =============================================================================
# Main
# =============================================================================

def main(args):
    # setup logging
    const.logger = setup_custom_logger(__name__)

    # allocate gpu memory
    session_number = get_session()
    const.logger.info("allocate GPU memory, session number: {}".format(session_number))

    # extract arguments from command line
    const.train_dir_path = args.train_dir_path
    const.val_dir_path = args.val_dir_path
    const.train_features_path = args.train_features_path
    const.val_features_path = args.val_features_path
    const.dataset_type = args.dataset_type
    const.conv_model_type = args.conv_model_type
    const.conv_model_name = {
        ConvType.ResNet50.value: ResNet50.__name__
    }.get(const.conv_model_type)
    const.nb_epoch = args.nb_epoch
    const.dropout_rate = args.dropout_rate
    const.batch_size = args.batch_size
    const.seed = args.seed

    logging.debug("INPUTS")
    logging.debug("train_dir_path: {}".format(const.train_dir_path))
    logging.debug("val_dir_path: {}".format(const.val_dir_path))
    logging.debug("dataset_type: {}".format(const.dataset_type))
    logging.debug("conv_model_type: {}".format(const.conv_model_type))
    logging.debug("nb_epoch: {}".format(const.nb_epoch))
    logging.debug("dropout_rate: {}".format(const.dropout_rate))
    logging.debug("batch_size: {}".format(const.batch_size))
    logging.debug("seed: {}\n".format(const.seed))

    # setup state machine
    m = StateMachine()
    m.add_state("init", init)
    m.add_state("load_data", load_data)
    m.add_state("load_cnn", load_cnn)
    m.add_state("freeze_cnn", freeze_cnn)
    m.add_state("extract_features", extract_features)
    m.add_state("build_classifier", build_classifier)
    m.add_state("compile_model", compile_model)
    m.add_state("checkpoint_model", checkpoint_model)
    m.add_state("train_model", train_model)
    m.add_state("unfreeze_cnn", unfreeze_cnn)
    m.add_state("combine_models", combine_models)
    m.add_state("happy_ending", None, end_state=1)
    m.add_state("error_state", None, end_state=1)
    m.set_start("init")

    # setup cargo
    cargo = collections.namedtuple("cargo", [])

    # run state machine
    m.run(cargo)

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
        default=50,
        help="Specify epoch, the total number of iterations on the data.")
    parser.add_argument("-d", dest="dropout_rate", action="store", type=float,
        default=0.5,
        help="Specify dropout rate. Dropout helps model from not memorizing or overfitting the data.")
    parser.add_argument("-b", dest="batch_size", action="store", type=int,
        default=None,
        help="Specify batch size, the number of images to look at a time. If not specified, use keras's default batch_size.")
    parser.add_argument("-s", dest="seed", action="store", type=int, 
        default=random.randint(1, 100),
        help="(OPTIONAL) SEED is used for random number generator.")
    main(parser.parse_args())


