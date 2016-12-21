""" Run metrics to determine how well the ResNet model identify the same car

This script will create a plot that will have
 * x-axis: the range for the number of times we have to go through in the set sorted by the shortest
           cosine distance to match the target car image
 * y-axis: cummulative sum of
       (the number of times we have to go through in the set to match the target car image)
       divided by
       (the number of run)

Assumption:
    * FEATURE is a json file with imageName (filename, not filepath) and resNet50 (feature vectors)
    * FEATURE includes all the feature vectors for both test and training images

Output:
    * cmc_metric.pdf: plot of CMC metric
    * str_metric.pdf: plot of STR metric
    * metric.log: log messages 

Reliance:
    The script will rely on ExperimentGenerator to build a list of sets (of images) for the experiment.

Metric:
    1. CMC: compare the target car with a set of 10 car images
       Step #1: Using the ExperimentGenerator, create a set of 10 car images with zero drop rate,
                meaning the target car will exist in the set.
       Step #2: Match each feature vector to the images in the set.
       Step #3: Calculate the cosine distance of the target car's feature vector and the feature vector
                of each image in the set, resulting in 10 computations.
       Step #4: Sort the list of images in the set by the shortest cosine distance.
       Step #5: Determine how many images it has to go through the sorted cosine distance list to 
                find the target car. 
       Step #6: Plot out the graph.

    2. STR: compare a set of 10 car images with another set of 10 car images
       Step #1: Using the ExperimentGenerator, create two sets of 10 car images with zero drop rate,
                meaning the target car will exist in both set.
       Step #2: Match each feature vector to the images in the sets.
       Step #3: Calculate the cosine distance of each car's feature vector in set 1 and set 2,
                resulting in 100 computations. 
       Step #4: Sort the list of images in the set by the shortest cosine distance.
       Step #5: Determine how many image pairs it has to go through the sorted cosine distance list to 
                find the target car image pair. 
       Step #6: Plot out the graph.

Usage:
    metric.py [-hv]
    metric.py [-e <SEED> -c -s] -w <DATASET_TYPE> -r <NUM_RUN> <VERI> <FEATURE>

Arguments:
    VERI                            : Path to the VeRi dataset unzipped
    FEATURE                         : Path to the feature json file

Options:
    -h, --help                      : Show this help message.
    -v, --version                   : Show the version number.
    -c, --cmc                       : Run CMC (Cummulative Matching Curve) metric
    -s, --str                       : Run STR (N^2) metric
    -r, --num_run=<NUM_RUN>         : How many iterations to run the ranking
    -w, --dataset_type=<DATASET_TYPE>       : Specify the datasets to use. 
                                      ["CompcarsDataset", "StrDataset", "VeriDataset"]
    -e, --seed=<SEED>               : Seed to be used for random number generator (default: random [1-100])

"""

import argparse
import collections
import logging
import matplotlib.pyplot as plt
import os
import random
import scipy
import scipy.spatial
import sys

import pelops.utils as utils
from experiment import ExperimentGenerator


class MetricRunner(object):
    CMC = 0
    STR = 1

    def __init__(self, dataset_path, feature_path, seed, dataset_type, num_run):
        # mandatory
        self.dataset_path = dataset_path
        self.feature_path = feature_path
        self.num_run = num_run
        # optional but initialized
        self.seed = seed
        self.dataset_type = dataset_type
        # logging
        log_file = "metric.log"
        utils.remove_file(log_file)
        logging.basicConfig(filename=log_file, level=logging.DEBUG)

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
        logging.info("-" * 80)
        logging.info("Instantiate ExperimentGenerator")
        logging.info("num_cams = {}, num_cars_per_cam = {}, drop_percentage = {}".format(num_cams, num_cars_per_cam, drop_percentage))
        logging.info("-" * 80)
        exp = ExperimentGenerator(self.dataset_path, self.dataset_type, num_cams, num_cars_per_cam, drop_percentage, self.seed)
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
        logging.info("-" * 80)
        logging.info("Instantiate ExperimentGenerator")
        logging.info("num_cams = {}, num_cars_per_cam = {}, drop_percentage = {}".format(num_cams, num_cars_per_cam, drop_percentage))
        logging.info("-" * 80)
        exp = ExperimentGenerator(self.dataset_path, self.dataset_type, num_cams, num_cars_per_cam, drop_percentage, self.seed)
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
        output_name = {
            MetricRunner.CMC: "cmc_metric.pdf",
            MetricRunner.STR: "str_metric.pdf", 
        }.get(which_metric)
        self.__plot(collections.Counter(attempts), output_name)
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
        target_car_filename = utils.get_basename(exp.target_car.filepath)
        logging.info("target {}".format(target_car_filename))
        target_car_vector = feature_vectors[target_car_filename]

        logging.info("Identify chosen set vs comparison set")
        # chosen
        chosen_set = {
            CMC: [exp.target_car],
            STR: camsets[0],
        }.get(which_metric)
        # comparison
        comp_sets = {
            CMC: camsets,
            STR: camsets[1:],
        }.get(which_metric)

        logging.info("Calculate cosine distances between the sets")
        cosine_distances = list()
        for chosen_car in chosen_set:
            logging.info(">> Match chosen car to its respective vector")
            chosen_car_filename = utils.get_basename(chosen_car.filepath)
            chosen_car_vector = feature_vectors[chosen_car_filename]
            for comp_set in comp_sets:
                for comp_car in comp_set:
                    logging.info(">> Match comparison car to its respective vector")
                    comp_car_filename = utils.get_basename(comp_car.filepath)
                    comp_car_vector = feature_vectors[comp_car_filename]
                    logging.info(">> Calculate the cosine distance")
                    cosine_distance = scipy.spatial.distance.cosine(chosen_car_vector, comp_car_vector)
                    cosine_distances.append((chosen_car.car_id, comp_car.car_id, cosine_distance))
                    logging.info(">> chosen {}, comp {}, cosine distance {}".format(chosen_car.filepath, comp_car.filepath, cosine_distance))
        
        logging.info("Sort the cosine distances")
        cosine_distances = sorted(cosine_distances, key=lambda tupl:tupl[COSINE_DISTANCE_INDEX])

        logging.info("Determine how many times we have to go through the sorted list to find the matching target car")
        attempt = {
            CMC: utils.get_index_of_tuple(cosine_distances, COMP_CAR_INDEX, exp.target_car.car_id),
            STR: utils.get_index_of_pairs(cosine_distances, CHOSEN_CAR_INDEX, COMP_CAR_INDEX, exp.target_car.car_id),
        }.get(which_metric)

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

    def __plot(self, num_per_index, output_name):
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
        """
        # annotate all value points
        for i, y_val in enumerate(y):
            plt.annotate(y_val, xy=(x[i], y[i]))
        """
        # annotate only the first point
        plt.annotate(y[0], xy=(x[0], y[0]))
        plt.savefig(output_name)
        return

# -----------------------------------------------------------------------------
#  Execution example
# -----------------------------------------------------------------------------


def main(args):
    # extract arguments from command line
    dataset_path = args.dataset_path
    feature_path = args.feature
    is_cmc = args.cmc
    is_str = args.str
    num_run = args.num_run
    dataset_type  = args.dataset_type
    seed = args.seed

    # check that input_path points to a directory
    if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
        sys.exit("ERROR: filepath to VeRi directory (%s) is invalid" % dataset_path)

    # check that a metric is selected to run
    if not is_cmc and not is_str:
        sys.exit("ERROR: you need to specify which metric to run")

    # create the metric runner
    runner = MetricRunner(dataset_path, feature_path, seed, dataset_type, num_run)

    # run the experiment
    if is_cmc:
        runner.run_cmc()
    if is_str:
        runner.run_str()

# -----------------------------------------------------------------------------
#  Entry
# -----------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="metric.py", description="Run metrics to determine how well the ResNet model identify the same car. Outputs: ", formatter_class=argparse.RawTextHelpFormatter)
    # arguments
    parser.add_argument("dataset_path", default="dataset_path", action="store", type=str,
                        help="Path to the dataset unzipped.")
    parser.add_argument("feature", default="feature", action="store", type=str,
                        help="Path to the feature json file.\nMake sure that feature is the same type as TYPE.")
    # options
    parser.add_argument("-v", "--version", action="version", version="Metric Runner 1.0")
    parser.add_argument("-w", dest="dataset_type", choices=["CompcarsDataset", "StrDataset", "VeriDataset"])
    parser.add_argument("-c", "--cmc", dest="cmc", action="store_true", default=False, 
                        help="Run CMC metric.")
    parser.add_argument("-s", "--str", dest="str", action="store_true", default=False,
                        help="Run STR metric.")
    parser.add_argument("-r", dest="num_run", action="store", type=int,
                        help="NUM_RUN defines how many iterations to run the metric.")
    parser.add_argument("-e", dest="seed", action="store", type=int,
                        default=random.randint(1, 100),
                        help="(OPTIONAL) SEED is used for random number generator.")    
    main(parser.parse_args())
