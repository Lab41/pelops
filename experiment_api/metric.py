""" Run metrics to determine how well the ResNet model identify the same car

This script will create a plot that will show 

Reliance: 
    The script will rely on ExperimentGenerator to build a list of sets (of images) for the experiment.

Metric: 
    1. CMC
    
    2. STR
    

Usage:
    metric.py [-hv]
    metric.py [-e <SEED> -y <TYPE> -l -c -s] -r <NUM_RUN> <VERI> <FEATURE>

Arguments:
    VERI                            : Path to the VeRi dataset unzipped
    FEATURE                         : Path to the feature json file

Options:
    -h, --help                      : Show this help message.
    -v, --version                   : Show the version number.
    -e, --seed=<SEED>               : Seed to be used for random number generator.
    -y, --type=<TYPE>               : Determine which type of images and features to use.
                                      0: all, 1: query, 2: test, 3: train (default: test)
    -r, --num_run=<NUM_RUN>         : How many iterations to run the ranking
    -l, --log                       : Flag to output debug messages.
    -c, --cmc                       : Run CMC metric
    -s, --str                       : Run STR metric

"""

import collections
import docopt
import logger
import matplotlib.pyplot as plt
import os
import re
import scipy
import scipy.spatial
import sys

from experiment import ExperimentGenerator
from utils import * 


class MetricRunner(object):
    def __init__(self, veri_unzipped_path, feature_path, seed, typ, num_run, is_log):
        self.veri_unzipped_path = veri_unzipped_path
        self.feature_path = feature_path
        self.seed = seed
        self.typ = typ
        self.num_run = num_run
        self.logger = logging.getLogger("Metric Runner")
        if is_log:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

    # TODO: running STR is different than running CMC in get_first_ranks
    def run_str(self):
        """ STR will compare a set of 10 car images with another set of 10 car images
        """
        self.logger.info("=" * 80)
        self.logger.info("Running STR metric")
        # instantiate ExperimentGenerator
        num_cams = 2
        num_cars_per_cam = 10
        drop_percentage = 0
        time = 5
        self.logger.info("-" * 80)
        self.logger.info("Instantiate ExperimentGenerator")
        self.logger.info("num_cams = {}, num_cars_per_cam = {}, drop_percentage = {}".format(num_cams, num_cars_per_cam, drop_percentage))
        self.logger.info("-" * 80)
        exp = ExperimentGenerator(self.veri_unzipped_path, num_cams, num_cars_per_cam, drop_percentage, self.seed, time, self.typ)
        # run the metric
        self.__run_ranker(exp)
        self.logger.info("=" * 80)
        return

    def run_cmc(self):
        """ CMC will compare target car image with a set of 10 car images including the car
        """
        self.logger.info("=" * 80)
        self.logger.info("Running CMC metric")
        # instantiate ExperimentGenerator
        num_cams = 1
        num_cars_per_cam = 10
        drop_percentage = 0
        time = 5
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
        first_ranks = list()
        for i in range(0, self.num_run): 
            self.logger.info("Run #{}".format(i + 1))
            first_ranks = first_ranks + self.__get_first_ranks(exp)
            self.logger.info("Adding the first ranks into the list:")
            self.logger.info(first_ranks)
            self.logger.info("-" * 80)
        
        self.__plot(collections.Counter(first_ranks), len(first_ranks))
        return

    def __get_first_ranks(self, exp):
        self.logger.info("Generate set of images")
        camsets = exp.generate()
        self.logger.info("Match the target car to its respective vector")
        target_car_vector = self.__match_vector(exp.target_car)
        self.logger.info("Match each car image to its respective vector")
        first_ranks = list()
        cosine_distances = list()
        index = 1
        for camset in camsets:
            for image in camset:
                image_vector = self.__match_vector(image)
                cosine_distance = scipy.spatial.distance.cosine(target_car_vector, image_vector)
                cosine_distances.append((index, cosine_distance))
                index = index + 1            
            self.logger.info("Calculate the cosine distance of the image's vector against the target car's vector")
            self.logger.info("Rank the cosine distance with the shortest distance being #1")
            self.logger.debug("Before cosine distance is sorted [(index, cosine_distance)]:")
            self.logger.debug(cosine_distances)
            sorted(cosine_distances, key=lambda tupl: tupl[1])
            self.logger.debug("After cosine distance is sorted [(index, cosine distance)]:")
            self.logger.debug(cosine_distances)
            self.logger.info("The index of the image with the shortest cosine distance is:")
            self.logger.info(cosine_distances[0][0])
            first_rank = cosine_distances[0][0] 
            first_ranks.append(first_rank)
            # reset
            cosine_distances = list()
            index = 1
        return first_ranks

    def __match_vector(self, image):
        objs = read_json(self.feature_path)
        for obj in objs:
            # find the first image that has the same vehicle id and camera id
            if obj["imageName"] == image.name:
                return obj["resnet50"]
            self.logger.debug(">> matching image car: {} vs {}".format(image.car_id, obj["imageName"]))

    def __plot(self, num_each_rank, len_first_ranks):
        def get_y(x):
            y = list()
            total = 0.
            for key in x:
                total = total + (float(num_each_rank[key]) / float(len_first_ranks))
                y.append(total)
            return y

        x = sorted(num_each_rank.keys())
        y = get_y(x)

        self.logger.info("x value: {}".format(x))
        self.logger.info("y value: {}".format(y))

        plt.plot(x, y, 'ro')
        plt.xlabel("rank")
        plt.ylabel("cummulative")
        plt.axis([1, max(x), 1, max(y)])
        plt.savefig("cmc_metric.pdf") # TODO: save it in a better manner
        return


# -----------------------------------------------------------------------------
#  Execution example
# -----------------------------------------------------------------------------

def main(args):
    # extract arguments from command line
    try:
        veri_unzipped_path = args["<VERI>"]
        feature_path = args["<FEATURE>"]
        if args["--cmc"]:
            is_cmc = True
        else:
            is_cmc = False
        if args["--str"]:
            is_str = True
        else:
            is_str = False
        if args["--log"]:
            is_log = True
        else:
            is_log = False
        if args["--seed"]:
            seed = int(args["--seed"])
        else: # set default
            seed = random.randint(1, 100)
        if args["--type"]:
            typ = int(args["--type"])
        else: # set default
            typ = ImageType.TEST
        num_run = int(args["--num_run"])
    except docopt.DocoptExit as e:
        sys.exit("ERROR: input invalid options: %s" % e)

    # check that input_path points to a directory
    if not os.path.exists(veri_unzipped_path) or not os.path.isdir(veri_unzipped_path):
        sys.exit("ERROR: filepath to VeRi directory (%s) is invalid" % veri_unzipped_path)

    # check that a metric is selected to run
    if not is_cmc and not is_str:
        sys.exit("ERROR: you need to specify which metric to run")

    # create the metric runner
    runner = MetricRunner(veri_unzipped_path, feature_path, seed, typ, num_run, is_log)

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
