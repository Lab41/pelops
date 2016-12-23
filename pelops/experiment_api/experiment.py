""" Generate sets of images for experiment

This script will build a list of sets (of images) for the experiment.

Input:
    We use the following datasets: CompCars, GD (from Google), STR, and VeRi.

    VeRi dataset contains:
        * 49358 images (1679 query images, 11580 test images, 37779 train images)
        * 776 vehicles
        * 20 cameras
        * covering 1.0 km^2 area in 24 hours

    Liu, X., Liu, W., Ma, H., Fu, H.: Large-scale vehicle re-identification in urban surveillance videos.
    In: IEEE International Conference on Multimedia and Expo. (2016) accepted.

Output:
    * Define a target car image
    * Outputs sets containing images
        * The number of set is defined by NUM_CAMS.
        * Each set contains a number of car images defined by NUM_CARS_PER_CAM.
        * All images in a set is taken by the same camera.
        * The target car, which will be randomly selected, will exist in each set depending on DROP_PERCENTAGE.
        * Each set contains distinct cars unless the dataset only contains one car.

Example:
    * num_cams = 10, which means we will create 10 camera sets
    * num_cars_per_cam = 5, which means we will have 5 distinct cars within the set
    * drop = 0.3, which means the target car will be dropped 30% of the time
    [
     set(Image(target_car), Image(1), Image(2), Image(3), Image(4)), # set 1
     set(Image(target_car), Image(1), Image(2), Image(3), Image(4)), # set 2
     set(Image(1), Image(2), Image(3), Image(4), Image(5)),          # set 3, target car is dropped
     set(Image(target_car), Image(1), Image(2), Image(3), Image(4)), # set 4
     set(Image(target_car), Image(1), Image(2), Image(3), Image(4)), # set 5
     set(Image(target_car), Image(1), Image(2), Image(3), Image(4)), # set 6
     set(Image(1), Image(2), Image(3), Image(4), Image(5)),          # set 7, target car is dropped
     set(Image(target_car), Image(1), Image(2), Image(3), Image(4)), # set 8
     set(Image(1), Image(2), Image(3), Image(4), Image(5)),          # set 9, target car is dropped
     set(Image(target_car), Image(1), Image(2), Image(3), Image(4)), # set 10
    ]

Usage:
    experiment.py [-hv]
    experiment.py [-e <SEED>] -s <NUM_CAMS> -c <NUM_CARS_PER_CAM> -d <DROP_PERCENTAGE> -w <DATASET_TYPE> -y <SET_TYPE> <DATASET_PATH>

Arguments:
    DATASET_PATH                    : Path to the dataset unzipped

Options:
    -h, --help                      : Show this help message.
    -v, --version                   : Show the version number.
    -s NUM_CAMS                     : Each camera maps to a set. NUM_CAMS specify the number of camera sets to be outputted.
    -c NUM_CARS_PER_CAM             : Each set has a list of images. NUM_CARS_PER_CAM specify the number of car images in each camera set.
    -d DROP_PERCENTAGE              : The likelihood that the target car image is dropped (float from [0,1])
    -e SEED                         : Seed to be used for random number generator.
    -y SET_TYPE                     : Determine which type of images to use.
                                      0: all, 1: query, 2: test, 3: train
    -w DATASET_TYPE                 : Specify the datasets to use. 
                                      ["CompcarsDataset", "StrDataset", "VeriDataset"]

"""
import argparse
import collections
import datetime
import os
import random
import sys

import pelops.datasets.chip as chip
import pelops.datasets.str as str_sa
import pelops.datasets.veri as veri 
import pelops.utils as utils


class ExperimentGenerator(object):

    def __init__(self, dataset_path, dataset_type, num_cams, num_cars_per_cam, drop_percentage, seed, set_type):
        # set inputs
        self.dataset = chip.DatasetFactory.create_dataset(dataset_type, dataset_path, set_type)
        self.num_cams = num_cams
        self.num_cars_per_cam = num_cars_per_cam
        self.drop_percentage = drop_percentage
        self.seed = seed
        # stuff that needs to be initialized
        random.seed(self.seed)
        self.list_of_cameras_per_car = self.dataset.get_distinct_cams_per_car()
        self.list_of_cameras = self.dataset.get_all_cam_ids()
        self.list_of_cars = self.dataset.get_all_car_ids()

    def __is_only_one_car(self):
        return len(self.list_of_cars) == 1

    def __is_taken_by_only_one_camera(self, car_id):
        return len(self.list_of_cameras_per_car[car_id]) == 1

    def __set_target_car(self):
        # count the number of times distinct cameras spot the car
        # has to be greater than or equal to num_cams
        # or not drop percentage is going to be higher
        list_valid_target_cars = []
        for car_id, cam_ids in self.list_of_cameras_per_car.items():
            if len(cam_ids) >= self.num_cams or self.num_cams > len(self.list_of_cameras):
                list_valid_target_cars.append(car_id)

        # get the target car
        car_id = random.choice(list_valid_target_cars)
        self.target_car = random.choice(
            list(self.dataset.get_all_chips_by_car_id(car_id)))

    def __create_similar_target_car(self):
        while True:
            similar_target_car = random.choice(
                list(self.dataset.get_all_chips_by_car_id(self.target_car.car_id)))
            # WARNING: If the car is only taken by one camera, this will go into an infinite loop
            # prevent that from happening with this check
            if self.__is_taken_by_only_one_camera(self.target_car.car_id):
                break
            # make sure the car is taken by a different camera
            if self.target_car.cam_id != similar_target_car.cam_id:
                break
        return similar_target_car

    def __get_camset(self):
        num_imgs_per_camset = self.num_cars_per_cam
        camset = list()
        which_cam_id = random.choice(list(self.list_of_cameras))
        # determine whether or not to add a different target car image
        if not utils.should_drop(self.drop_percentage):
            num_imgs_per_camset = num_imgs_per_camset - 1
            similar_target_car = self.__create_similar_target_car()
            which_cam_id = similar_target_car.cam_id
            camset.append(similar_target_car)
        # grab images
        exist_car = list()
        for i in range(0, num_imgs_per_camset):
            while True:
                random_car = random.choice(
                    list(self.dataset.get_all_chips_by_cam_id(which_cam_id)))
                # WARNING: If the dataset only contains one car, this will go into an infinite loop
                # prevent that from happening with this check
                if self.__is_only_one_car():
                    break
                # check if same car already existed
                if(random_car.car_id not in exist_car):
                    # different car
                    break
            exist_car.append(random_car.car_id)
            camset.append(random_car)
        return camset

    def generate(self):
        self.__set_target_car()
        output = list()
        for i in range(0, self.num_cams):
            output.append(self.__get_camset())
        return output


# -----------------------------------------------------------------------------
#  Execution example
# -----------------------------------------------------------------------------


# @timewrapper
def main(args):
    # extract arguments from command line
    dataset_path = args.dataset_path
    dataset_type = args.dataset_type
    num_cams = args.num_cams
    num_cars_per_cam = args.num_cars_per_cam
    drop_percentage = args.drop_percentage
    set_type = args.set_type
    seed = args.seed

    # check that input_path points to a directory
    if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
        sys.exit("ERROR: filepath to directory (%s) is invalid" %
                 dataset_path)

    # create the generator
    exp = ExperimentGenerator(
        dataset_path, dataset_type, num_cams, num_cars_per_cam, drop_percentage, seed, set_type)

    # generate the experiment
    set_num = 1
    print("=" * 80)
    for camset in exp.generate():
        print("Set #{}".format(set_num))
        print("Target car: {}".format(utils.get_basename(exp.target_car.filepath)))
        print("-" * 80)
        for image in camset:
            print("filepath: {}".format(image.filepath))
            print("car id: {}".format(image.car_id))
            print("camera id: {}".format(image.cam_id))
            print("timestamp: {}".format(utils.get_timestamp(image.time)))
            if image.misc is not None:
                for key, value in image.misc.items():
                    print("{}: {}".format(key, value))
            print("-" * 80)
        print("=" * 80)
        set_num = set_num + 1
    return

# -----------------------------------------------------------------------------
#  Entry
# -----------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="experiment.py", description="Generate sets of images for metrics", formatter_class=argparse.RawTextHelpFormatter)
    # arguments
    parser.add_argument("dataset_path", default="dataset_path", action="store", type=str,
                        help="Path to the dataset unzipped.")
    # options
    parser.add_argument("-v", "--version", action="version",
                        version="Experiment Generator 1.0")
    parser.add_argument("-w", dest="dataset_type", action="store", choices=["CompcarDataset", "StrDataset", "VeriDataset"], type=str,
                        help="Specify the datasets to use.")
    parser.add_argument("-s", dest="num_cams", action="store", type=int,
                        help="Each camera maps to a set.\nNUM_CAMS specifies the number of camera sets to be outputted.")
    parser.add_argument("-c", dest="num_cars_per_cam", action="store", type=int,
                        help="Each set has a list of car images. The cars in a set are distinct.\nNUM_CARS_PER_CAM specifies the number of car images in each camera set.")
    parser.add_argument("-d", dest="drop_percentage", action="store", type=float,
                        help="DROP_PERCENTAGE specifies the likelihood that\nthe target car image is dropped (float from [0,1]).")
    parser.add_argument("-y", dest="set_type", action="store", choices=[0, 1, 2, 3], type=int,
                        help="TYPE determines which type of images to use.\n0: all, 1: query, 2: test, 3: train")
    parser.add_argument("-e", dest="seed", action="store", type=int,
                        default=random.randint(1, 100),
                        help="(OPTIONAL) SEED is used for random number generator.")

    main(parser.parse_args())
