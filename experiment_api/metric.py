""" Run metrics to determine how well the ResNet model identify the same car

Reliance: 
    The script will rely on ExperimentGenerator to build a list of sets for the experiment.


Usage:
    metric.py [-hv]
    metric.py 

Arguments:
    VERI                            : Path to the VeRi dataset unzipped
    TRAIN_FEATURE                   : Path to the train feature json file
    TEST_FEATURE                    : Path to the test feature json file

Options:
    -h, --help                      : Show this help message.
    -v, --version                   : Show the version number.

"""

import collections
import matplotlib.pyplot as plt
import os
import re
import scipy
import scipy.spatial

from experiment import ExperimentGenerator
from utils import * 


class MetricRunner(object):
    def __init__(self, veri_unzipped_path, seed, typ, num_run):
        self.veri_unzipped_path = veri_unzipped_path
        self.seed = seed
        self.typ = typ
        self.num_run = num_run

    def run_str(self):
        """ STR will compare a set of 10 car images with another set of 10 car images
        """
        # generate set of images 
        num_cams = 2
        num_cars_per_cam = 10
        drop_percentage = 0
        time = 5
        exp = ExperimentGenerator(veri_unzipped_path, num_cams, num_cars_per_cam, drop_percentage, seed, time, typ)
        # run the metrics
        self.__run_ranker(exp)
        return

    def run_cmc(self):
        """ CMC will compare target car image with a set of 10 car images including the car
        """
        # generate set of images
        num_cams = 1
        num_cars_per_cam = 10
        drop_percentage = 0
        time = 5
        exp = ExperimentGenerator(veri_unzipped_path, num_cams, num_cars_per_cam, drop_percentage, seed, time, typ)
        # run the metrics
        self.__run_ranker(exp)
        return

    def __run_ranker(self, exp):

        rank = list()
        counter = 1

        print("after experiment generator")
        first_ranks = list()
        for i in range(0, num_run): 
            first_rank = get_first_rank(exp)
            first_ranks.append(first_rank)

        print("calculate num_ranks")
        num_ranks = collections.Counter(first_ranks)
        print(num_ranks)

        plot(num_ranks)

def plot(num_ranks):
    x = sorted(num_ranks.keys())
    print("x", x)
    y = get_y(num_ranks)
    print("y", y)

    plt.plot(x, y, 'ro')
    plt.xlabel("rank")
    plt.ylabel("number of ranks")
    plt.axis([1, max(x), 1, max(y)])
    plt.show()
    return

def get_y(num_ranks):
    y = list()
    total_ranks = len(num_ranks)

    for key in sorted(num_ranks.keys()):
        y.append(num_ranks[key] / total_ranks)
    return y

def get_first_rank(exp):

    exp_generator = exp.generate()
    target_car_vector = match_target_car_vector(exp.target_car)
    rank = list()
    counter = 1
    print("start get_first_rank")
    for camset in exp_generator:
        for image in camset:
            match_vector(image)
            print("image.vector = ")
            print(image.vector)
            image_cos = scipy.spatial.distance.cosine(target_car_vector, image.vector)
            rank.append((counter, image_cos))
            counter = counter + 1

    print("before sort")
    print(rank)
    # sort 
    sorted(rank, key=lambda tupl: tupl[1])
    print("after sort")
    print(rank)

    print("end get_first_rank")

    # return the first item in the sort and give the rank number
    return rank[0][0]
 
def match_vector(image):

    print("start match_vector")
    objs = read_json(test)
    print(objs)
    for obj in objs:
        if read_car_id(obj["vehicleID"]) == image.car_id and read_camera_id(obj["cameraID"]) == image.camera_id:
            print("obj[resnet50]", obj["resnet50"])
            image.set_vector(obj["resnet50"])
            return

        print("obj[vehicleID]", read_car_id(obj["vehicleID"]), image.car_id)
        print("obj[cameraID]", read_camera_id(obj["cameraID"]), image.camera_id)

    print("finish match vector")


def match_target_car_vector(target_car_id):

    print("start match_vector")
    objs = read_json(test)
    print(objs)
    for obj in objs:
        if read_car_id(obj["vehicleID"]) == target_car_id:
            return obj["resnet50"]

        print("obj[vehicleID]", read_car_id(obj["vehicleID"]), target_car_id)

    print("finish match vector")

# -----------------------------------------------------------------------------
#  Execution example
# -----------------------------------------------------------------------------

def main(args):
    # extract arguments from command line
    try:
        veri_unzipped_path = args["<VERI>"]
        if args["--cmc"]:
            is_cmc = True
        if args["--str"]:
            is_str = True
        seed = int(args["--seed"])
        typ = int(args["--type"])
        num_run = int(args["--num_run"])
    except docopt.DocoptExit as e:
        sys.exit("ERROR: input invalid options: %s" % e)

    # check that input_path points to a directory
    if not os.path.exists(input_path) or not os.path.isdir(input_path):
        sys.exit("ERROR: input path (%s) is invalid" % input_path)

    # run the experiment
    if is_cmc: 
        run_cmc(veri_unzipped_path, seed, typ, num_run)
    if is_str: 
        run_str(veri_unzipped_path, seed, typ, num_run)

# -----------------------------------------------------------------------------
#  Entry
# -----------------------------------------------------------------------------


if __name__ == '__main__':
    cmc()
