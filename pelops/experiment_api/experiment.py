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
from itertools import combinations

import pelops.datasets.chip as chip
import pelops.datasets.str as str_sa
import pelops.datasets.veri as veri 
import pelops.utils as utils
from pelops.datasets.featuredataset import FeatureDataset

class ExperimentGenerator(object):

    def __init__(self, dataset, num_cams, num_cars_per_cam, drop_percentage, seed, key_filter= lambda x: True):
        # set inputs
        self.dataset = dataset
        self.num_cams = num_cams
        self.num_cars_per_cam = num_cars_per_cam
        self.drop_percentage = drop_percentage
        self.seed = seed
        # stuff that needs to be initialized
        random.seed(self.seed)
        d = self.dataset.get_distinct_cams_per_car()
        self.list_of_cameras_per_car = {k:v for (k,v) in d.items() if key_filter(k)}
        self.list_of_cameras = self.dataset.get_all_cam_ids()
        self.list_of_cars = list(filter(key_filter, self.dataset.get_all_car_ids()))
        self.valid_target_cars = None

    def __is_only_one_car(self):
        return len(self.list_of_cars) == 1

    def __is_taken_by_only_one_camera(self, car_id):
        return len(self.list_of_cameras_per_car[car_id]) == 1

    def __set_target_car(self):
        # count the number of times distinct cameras spot the car
        # has to be greater than or equal to num_cams
        # or not drop percentage is going to be higher
        if self.valid_target_cars is None:
            self.valid_target_cars = {}
            for car_id, cam_ids in self.list_of_cameras_per_car.items():
                valid_cameras = self.__get_valid_potential_cameras(car_id)
                if len(valid_cameras) > 0:
                    self.valid_target_cars[car_id] = valid_cameras
            self.list_valid_target_cars = list(self.valid_target_cars.keys())
        # get the target car
        
        car_id = random.choice(self.list_valid_target_cars)
        
        self.target_car = random.choice(
            list(self.dataset.get_all_chips_by_car_id(car_id)))
    
    def __get_valid_potential_cameras(self, target_car):
        # TODO: list_of_cameras should be filtered by having more than # cars in camera
        
        chips_with_target_car = self.dataset.get_all_chips_by_car_id(target_car)
        potential_cameras = set([chip.cam_id for chip in chips_with_target_car])
        valid_combinations = []
        for cameras in combinations(potential_cameras, self.num_cams):
            used_cars = set()
            isValid = True
            for camera in cameras:
                # Get cars for this cars
                chips = self.dataset.get_all_chips_by_cam_id(camera)
                car_ids = set([chip.car_id for chip in chips])
                if len(car_ids - used_cars) < self.num_cars_per_cam:
                    isValid = False
                used_cars.update(car_ids)
            if isValid:
                valid_combinations.append(cameras)
                # TODO: THis should not be here
                return valid_combinations
        return valid_combinations
    
    def __get_valid_sample(cars, num_items):
        for i in range(1000):
            sample = random.sample(other_cars, num_items)
            if len(set([chip.car_id for chip in sample])) == num_items:
                return sample
        raise ValueError('Exceed maximum sample attempts, possibly no valid experiment')
    
    def __get_camset(self, target_car):
        valid_camera_sets = self.valid_target_cars[target_car]
        camera_set = random.choice(valid_camera_sets)
        camera_sets = []
        used_cars = set([target_car])
        for camera in camera_set:
            camera_set = []
            # Add target car to set
            potential = self.dataset.get_all_chips_by_car_id_camera_id(target_car, camera)
            camera_set.append(random.choice(potential))
            
            # Add other cars
            other_cars = self.dataset.get_all_chips_by_cam_id(camera)
            # Filter to exclude already includd cars
            other_cars = [chip for chip in other_cars if chip.car_id not in used_cars]
            selected_chips = random.sample(other_cars, self.num_cars_per_cam -1)
            used_cars.update([chip.car_id for chip in selected_chips])
            camera_set.extend(selected_chips)
            
            camera_sets.append(camera_set)
        return camera_sets
    

    def generate(self):
        self.__set_target_car()
        return self.__get_camset(self.target_car.car_id)

# -----------------------------------------------------------------------------
#  Execution example
# -----------------------------------------------------------------------------


# @timewrapper
def main(args):
    # extract arguments from command line
    dataset_path = args.dataset_path
    set_type = args.set_type
    dataset_type = args.dataset_type
    dataset = chip.DatasetFactory.create_dataset(dataset_type, dataset_path, set_type)
    num_cams = args.num_cams
    num_cars_per_cam = args.num_cars_per_cam
    drop_percentage = args.drop_percentage
    
    seed = args.seed

    # create the generator
    exp = ExperimentGenerator(
        dataset, num_cams, num_cars_per_cam, drop_percentage, seed)

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
