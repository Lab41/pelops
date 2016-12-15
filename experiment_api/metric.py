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
    metric.py [-e <SEED> -y <TYPE> -c -s] -r <NUM_RUN> <VERI> <FEATURE>

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
    CMC = 0
    STR = 1

    def __init__(self, veri_unzipped_path, feature_path, seed, typ, num_run):
        # mandatory
        self.veri_unzipped_path = veri_unzipped_path
        self.feature_path = feature_path
        self.num_run = num_run
        # optional but initialized
        self.seed = seed
        self.typ = typ
        logging.basicConfig(filename="metric.log", level=logging.DEBUG)

    # @timewrapper
    def run_str(self):
        """ STR will compare a set of 10 car images with another set of 10 car images
        """
        logging.info("=" * 80)
        logging.info("Running STR metric")
        # instantiate ExperimentGenerator
        num_cams = 2
        num_cars_per_cam = 10
        drop_percentage = 0
        time = 5  # TODO: currently this value does not matter
        logging.info("-" * 80)
        logging.info("Instantiate ExperimentGenerator")
        logging.info("num_cams = {}, num_cars_per_cam = {}, drop_percentage = {}".format(num_cams, num_cars_per_cam, drop_percentage))
        logging.info("-" * 80)
        exp = ExperimentGenerator(self.veri_unzipped_path, num_cams, num_cars_per_cam, drop_percentage, self.seed, time, self.typ)
        # run the metric
        self.__run(exp, MetricRunner.STR)
        logging.info("=" * 80)
        return

    # @timewrapper
    def run_cmc(self):
        """ CMC will compare target car image with a set of 10 car images
        """
        logging.info("=" * 80)
        logging.info("Running CMC metric")
        # instantiate ExperimentGenerator
        num_cams = 1
        num_cars_per_cam = 10
        drop_percentage = 0
        time = 5  # TODO: currently this value does not matter
        logging.info("-" * 80)
        logging.info("Instantiate ExperimentGenerator")
        logging.info("num_cams = {}, num_cars_per_cam = {}, drop_percentage = {}".format(num_cams, num_cars_per_cam, drop_percentage))
        logging.info("-" * 80)
        exp = ExperimentGenerator(self.veri_unzipped_path, num_cams, num_cars_per_cam, drop_percentage, self.seed, time, self.typ)
        # run the metric
        self.__run(exp, MetricRunner.CMC)
        logging.info("=" * 80)
        return

    def __run(self, exp, which_metric):
        # get feature vectors
        feature_vectors = self.__get_feature_vectors()
        # calculate which image in the set has the shortest cosine distance
        # and add the number of times we have to go through the sorted set to find the matching target car
        attempts = list()
        for i in range(0, self.num_run):
            logging.info("Run #{}".format(i + 1))
            attempts.append(self.__get_attempt(exp, feature_vectors, which_metric))
            logging.info("Adding the number of attempts to find the matching target car into the list:")
            logging.info(attempts)
            logging.info("-" * 80)
        # plot the output
        self.__plot(collections.Counter(attempts))
        return

    def __get_attempt(self, exp, feature_vectors, which_metric):
        # index reference for set creation
        CMC = MetricRunner.CMC
        STR = MetricRunner.STR
        # index reference for cosine_distances which is in the format of [(car_id, cosine_distance)]
        CHOSEN_CAR_INDEX = 0
        COMP_CAR_INDEX = 1
        COSINE_DISTANCE_INDEX = 2

        logging.info("Generate set of images")
        camsets = exp.generate()

        logging.info("Match target car to its respective vector")
        target_car_vector = feature_vectors[exp.target_car.name]

        logging.info("Identify chosen set vs comparison set")
        # chosen
        chosen_set = {
            CMC: [exp.target_car],
            STR: camsets[0],
        }.get(which_metric)
        # comparison
        comp_sets = {
            CMC: camsets,
            STR: camsets.pop(0),
        }.get(which_metric)

        print(len(chosen_set))
        print(len(comp_sets))
        index = 1
        for i in chosen_set:
            logging.info("chosen_set {}: {}".format(index, i.name))
            index = index + 1

        index_set = 1
        for comp_set in comp_sets:
            index = 1
            for i in comp_set: 
                logging.info("{} comp_set {}: {}".format(index_set, index, i.name))
                index = index + 1
            index_set = index_set + 1


        logging.info("Calculate cosine distances between the sets")
        cosine_distances = list()
        for chosen_car in chosen_set:
            logging.info(">> Match chosen car to its respective vector")
            chosen_car_vector = feature_vectors[chosen_car.name]
            for comp_set in comp_sets:
                for comp_car in comp_set:
                    logging.info(">> Match comparison car to its respective vector")
                    comp_car_vector = feature_vectors[comp_car.name]
                    logging.info(">> Calculate the cosine distance")
                    cosine_distance = scipy.spatial.distance.cosine(chosen_car_vector, comp_car_vector)
                    cosine_distances.append((chosen_car.car_id, comp_car.car_id, cosine_distance))
                    logging.info(">> chosen {}, comp {}, cosine distance {}".format(chosen_car.name, comp_car.name, cosine_distance))
        
        logging.info("Sort the cosine distances")
        cosine_distances = sorted(cosine_distances, key=lambda tupl:tupl[COSINE_DISTANCE_INDEX])

        logging.info("Determine how many times we have to go through the sorted list to find the matching target car")
        attempt = utils.get_index_of_tuple(cosine_distances, CHOSEN_CAR_INDEX, COMP_CAR_INDEX, exp.target_car.car_id)

        return attempt

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

        logging.info("x value: {}".format(x))
        logging.info("y value: {}".format(y))

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
        if args["--seed"]:
            seed = int(args["--seed"])
        else:  # set default
            seed = random.randint(1, 100)
        if args["--type"]:
            typ = int(args["--type"])
        else:  # set default
            typ = utils.ImageType.TEST.value
    except docopt.DocoptExit as e:
        sys.exit("ERROR: input invalid options: %s" % e)

    # check that input_path points to a directory
    if not os.path.exists(veri_unzipped_path) or not os.path.isdir(veri_unzipped_path):
        sys.exit("ERROR: filepath to VeRi directory (%s) is invalid" % veri_unzipped_path)

    # check that a metric is selected to run
    if not is_cmc and not is_str:
        sys.exit("ERROR: you need to specify which metric to run")

    # create the metric runner
    runner = MetricRunner(veri_unzipped_path, feature_path, seed, typ, num_run)

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
