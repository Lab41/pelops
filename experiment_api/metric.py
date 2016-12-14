""" Run metrics to determine how well the ResNet model identify the same car

This script will create a plot that will have
 * x-axis: the range for the number of times we have to go through in the set sorted by the shortest
           cosine distance to match the target car image
 * y-axis: cummulative sum of
       (the number of times we have to go through in the set to match the target car image)
       divided by
       (the number of run)

Reliance:
    The script will rely on ExperimentGenerator to build a list of sets (of images) for the experiment.

Metric:
    1. CMC: compare the target car with a set of 10 car images
       Step #1: Using the ExperimentGenerator, create a set of 10 car images with zero drop rate,
                meaning the target car will exist in the set.
       Step #2: Match each feature vector to the images in the set.
       Step #3: Calculate the cosine distance of the target car's feature vector and the feature vector
                of each image in the set.
       Step #4: Figure out which image has the shortest cosine distance to the target car.
       Step #5: Sort the list of images in the set by the shortest cosine distance.
       Step #6: Determine how many images it has to go through in the set to match the target car.
                Add this value into a list.
       Step #7: Plot out the graph.

    2. STR: compare a set of 10 car images with another set of 10 car images
       Similar to CMC, except that 10 car images are compared to another 10 car images instead of
       the target car image compared to 10 car images.

Usage:
    metric.py [-hv]
    metric.py [-e <SEED> -y <TYPE> -l -c -s] -r <NUM_RUN> <VERI> <FEATURE>

Arguments:
    VERI                            : Path to the VeRi dataset unzipped
    FEATURE                         : Path to the feature json file
                                      Make sure that feature is the same type as TYPE

Options:
    -h, --help                      : Show this help message.
    -v, --version                   : Show the version number.
    -e, --seed=<SEED>               : Seed to be used for random number generator (default: random [1-100])
    -y, --type=<TYPE>               : Determine which type of images and features to use.
                                      0: all, 1: query, 2: test, 3: train (default: test)
    -r, --num_run=<NUM_RUN>         : How many iterations to run the ranking
    -l, --log                       : Flag to output debug messages (default: info messages)
    -c, --cmc                       : Run CMC metric
    -s, --str                       : Run STR metric

"""

import collections
import docopt
import logging
import matplotlib.pyplot as plt
import os
import random
import scipy
import scipy.spatial
import sys
import utils

from experiment import ExperimentGenerator


class MetricRunner(object):
    def __init__(self, veri_unzipped_path, feature_path, seed, typ, num_run, is_debug):
        # mandatory
        self.veri_unzipped_path = veri_unzipped_path
        self.feature_path = feature_path
        self.num_run = num_run
        # optional but initialized
        self.seed = seed
        self.typ = typ
        # for logging purposes
        self.logger = logging.getLogger("Metric Runner")
        if is_debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        # initalizing

    # @timewrapper
    def run_str(self):
        """ STR will compare a set of 10 car images with another set of 10 car images
        """
        self.logger.info("=" * 80)
        self.logger.info("Running STR metric")
        # instantiate ExperimentGenerator
        num_cams = 2
        num_cars_per_cam = 10
        drop_percentage = 0
        time = 5  # TODO: currently this value does not matter
        self.logger.info("-" * 80)
        self.logger.info("Instantiate ExperimentGenerator")
        self.logger.info("num_cams = {}, num_cars_per_cam = {}, drop_percentage = {}".format(num_cams, num_cars_per_cam, drop_percentage))
        self.logger.info("-" * 80)
        exp = ExperimentGenerator(self.veri_unzipped_path, num_cams, num_cars_per_cam, drop_percentage, self.seed, time, self.typ)
        # run the metric
        self.__run_ranker(exp)
        self.logger.info("=" * 80)
        return

    # @timewrapper
    def run_cmc(self):
        """ CMC will compare target car image with a set of 10 car images
        """
        self.logger.info("=" * 80)
        self.logger.info("Running CMC metric")
        # instantiate ExperimentGenerator
        num_cams = 1
        num_cars_per_cam = 10
        drop_percentage = 0
        time = 5  # TODO: currently this value does not matter
        self.logger.info("-" * 80)
        self.logger.info("Instantiate ExperimentGenerator")
        self.logger.info("num_cams = {}, num_cars_per_cam = {}, drop_percentage = {}".format(num_cams, num_cars_per_cam, drop_percentage))
        self.logger.info("-" * 80)
        exp = ExperimentGenerator(self.veri_unzipped_path, num_cams, num_cars_per_cam, drop_percentage, self.seed, time, self.typ)
        # run the metric
        self.__run_ranker(exp)
        self.logger.info("=" * 80)
        return

    def __run_ranker(self, exp):
        # get feature vectors
        feature_vectors = self.__get_feature_vectors()
        # calculate which image in the set has the shortest cosine distance
        # and add the number of times we have to go through the sorted set to find the matching target car
        attempts = list()
        for i in range(0, self.num_run):
            self.logger.info("Run #{}".format(i + 1))
            attempts = attempts + self.__get_attempts(exp, feature_vectors)
            self.logger.info("Adding the number of attempts to find the matching target car into the list:")
            self.logger.info(attempts)
            self.logger.info("-" * 80)
        # plot the output
        self.__plot(collections.Counter(attempts))
        return

    def __get_attempts(self, exp, feature_vectors):
        # index reference later to be used
        CAR_ID_INDEX = 0
        COSINE_DISTANCE_INDEX = 1
        self.logger.info("Generate set of images")
        camsets = exp.generate()
        self.logger.info("Match the target car to its respective vector")
        target_car_vector = feature_vectors[exp.target_car.name]
        self.logger.info("Match each car image to its respective vector and calculate the cosine distance against the target car")
        attempts = list()
        cosine_distances = list()
        for camset in camsets:
            for image in camset:
                image_vector = feature_vectors[image.name]
                cosine_distance = scipy.spatial.distance.cosine(target_car_vector, image_vector)
                cosine_distances.append((image.car_id, cosine_distance))
                self.logger.info("Name: Target = {}, Image = {}".format(exp.target_car.name, image.name))
                self.logger.info("Cosine distance: {}".format(cosine_distance))
                self.logger.info("Sort the cosine distances")
            cosine_distances = sorted(cosine_distances, key=lambda tupl: tupl[COSINE_DISTANCE_INDEX])
            self.logger.info("Determine how many times we have to go through the sorted list to find the matching target car")
            attempt = utils.get_index_of_tuple(cosine_distances, CAR_ID_INDEX, exp.target_car.car_id)
            attempts.append(attempt)
            # reset
            cosine_distances = list()
        return attempts

    def __get_feature_vectors(self):
        # assume the json file has
        # imageName that references the image's name
        # resnet50 that references the image's feature vector
        feature_vectors = dict()
        objs = utils.read_json(self.feature_path)
        for obj in objs:
            feature_vectors[obj["imageName"]] = obj["resnet50"]
        return feature_vectors

    def __plot(self, num_per_index):
        def get_y(x):
            y = list()
            total = 0.
            for key in x:
                total = total + (float(num_per_index[key]) / float(self.num_run))
                y.append(total)
            return y

        x = sorted(num_per_index.keys())
        y = get_y(x)

        self.logger.info("x value: {}".format(x))
        self.logger.info("y value: {}".format(y))

        plt.plot(x, y, '-o')
        plt.xlabel("number of attempts to find the matching target car")
        plt.ylabel("cummulative sum")
        plt.axis([1, max(x), 0, max(y)])
        for i, y_val in enumerate(y):
            plt.annotate(y_val, xy=(x[i], y[i]))
        plt.savefig("cmc_metric.pdf")
        return


# -----------------------------------------------------------------------------
#  Execution example
# -----------------------------------------------------------------------------

def main(args):
    # extract arguments from command line
    try:
        # which metrics
        if args["--cmc"]:
            is_cmc = True
        else:
            is_cmc = False
        if args["--str"]:
            is_str = True
        else:
            is_str = False
        # mandatory
        veri_unzipped_path = args["<VERI>"]
        feature_path = args["<FEATURE>"]
        num_run = int(args["--num_run"])
        # optionals
        if args["--log"]:
            is_debug = True
        else:
            is_debug = False
        if args["--seed"]:
            seed = int(args["--seed"])
        else:  # set default
            seed = random.randint(1, 100)
        if args["--type"]:
            typ = int(args["--type"])
        else:  # set default
            typ = utils.ImageType.TEST
    except docopt.DocoptExit as e:
        sys.exit("ERROR: input invalid options: %s" % e)

    # check that input_path points to a directory
    if not os.path.exists(veri_unzipped_path) or not os.path.isdir(veri_unzipped_path):
        sys.exit("ERROR: filepath to VeRi directory (%s) is invalid" % veri_unzipped_path)

    # check that a metric is selected to run
    if not is_cmc and not is_str:
        sys.exit("ERROR: you need to specify which metric to run")

    # create the metric runner
    runner = MetricRunner(veri_unzipped_path, feature_path, seed, typ, num_run, is_debug)

    # run the experiment
    if is_cmc:
        runner.run_cmc()
    if is_str:
        runner.run_str()

# -----------------------------------------------------------------------------
#  Entry
# -----------------------------------------------------------------------------


if __name__ == '__main__':
    args = docopt.docopt(__doc__, version="Metric Runner 1.0")
    main(args)
