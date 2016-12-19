""" Generate sets of images for experiment

This script will build a list of sets (of images) for the experiment.

Input:
    We use VeRi dataset that contains the following information:
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
    experiment.py [-e <SEED>] -s <NUM_CAMS> -c <NUM_CARS_PER_CAM> -d <DROP_PERCENTAGE> -y <TYPE> <VERI>

Arguments:
    VERI                            : Path to the VeRi dataset unzipped

Options:
    -h, --help                      : Show this help message.
    -v, --version                   : Show the version number.
    -s, --cams=<NUM_CAMS>           : Each camera maps to a set. NUM_CAMS specify the number of camera sets to be outputted.
    -c, --cars=<NUM_CARS_PER_CAM>   : Each set has a list of images. NUM_CARS_PER_CAM specify the number of car images in each camera set.
    -d, --drop=<DROP_PERCENTAGE>    : The likelihood that the target car image is dropped (float from [0,1])
    -y, --type=<TYPE>               : Determine which type of images to use.
                                      0: all, 1: query, 2: test, 3: train
    -e, --seed=<SEED>               : Seed to be used for random number generator.

"""
import argparse
import collections
import datetime
import os
import random
import sys

import pelops.utils as utils

class ExperimentGenerator(object):

    def __init__(self, veri_unzipped_path, num_cams, num_cars_per_cam, drop_percentage, seed, typ):
        # set inputs
        self.set_filepaths(veri_unzipped_path)
        self.num_cams = num_cams
        self.num_cars_per_cam = num_cars_per_cam
        self.drop_percentage = drop_percentage
        self.typ = typ
        self.seed = seed
        # stuff that needs to be initialized
        random.seed(self.seed)
        self.list_of_cameras_per_car = collections.defaultdict(set)
        self.list_of_cameras = list()
        self.list_of_images_by_car_id = collections.defaultdict(set)
        self.list_of_images_by_camera_id = collections.defaultdict(set)

    def set_filepaths(self, veri_unzipped_path):
        self.name_query_filepath = veri_unzipped_path + \
            "/" + utils.Veri.name_query_filepath
        self.name_test_filepath = veri_unzipped_path + \
            "/" + utils.Veri.name_test_filepath
        self.name_train_filepath = veri_unzipped_path + \
            "/" + utils.Veri.name_train_filepath
        self.image_query_filepath = veri_unzipped_path + \
            "/" + utils.Veri.image_query_filepath
        self.image_test_filepath = veri_unzipped_path + \
            "/" + utils.Veri.image_test_filepath
        self.image_train_filepath = veri_unzipped_path + \
            "/" + utils.Veri.image_train_filepath

    def __merge_names(self, *filepaths):
        names = set()
        for filepath in filepaths:
            # determine image's type and directory
            if utils.Veri.name_query_filepath in filepath:
                img_type = "query"
                img_dir = self.image_query_filepath
            elif utils.Veri.name_test_filepath in filepath:
                img_type = "test"
                img_dir = self.image_test_filepath
            else:
                img_type = "train"
                img_dir = self.image_train_filepath
            # put all the names in the file into a set
            this_set = set(Image(name.strip(), img_dir, img_type)
                           for name in open(filepath))
            # combine with the previous list of names
            names = names.union(this_set)
        return names

    def __get_images(self):
        return {
            utils.ImageType.ALL.value: self.__merge_names(self.name_query_filepath, self.name_test_filepath, self.name_train_filepath),
            utils.ImageType.QUERY.value: self.__merge_names(self.name_query_filepath),
            utils.ImageType.TEST.value: self.__merge_names(self.name_test_filepath),
            utils.ImageType.TRAIN.value: self.__merge_names(self.name_train_filepath),
        }.get(self.typ)
        # }.get(self.typ, 0) # default to all

    def __set_lists(self):
        # TODO: figure out a better way to go about doing this
        # list_of_cameras_per_car: map each car with a list of distinct cameras that spot the car
        # list_of_car_names_per_camera: map each camera with a list of cars it spots
        # list_of_images_by_car_id: map each car image to its respective car id
        # list_of_images_by_camera_id: map each car image to its respective
        # camera id
        for image in self.__get_images():
            car_id = image.car_id
            camera_id = image.camera_id
            # list needed for finding target cars
            self.list_of_cameras_per_car[car_id].add(camera_id)
            self.list_of_images_by_car_id[car_id].add(image)
            # list needed for finding random cars
            self.list_of_cameras.append(camera_id)
            self.list_of_images_by_camera_id[camera_id].add(image)

    def __is_only_one_car(self):
        return len(self.list_of_images_by_car_id.keys()) == 1

    def __is_taken_by_only_one_camera(self, car_id):
        return len(self.list_of_cameras_per_car[car_id]) == 1

    def __set_target_car(self):
        # count the number of times distinct cameras spot the car
        # has to be greater than or equal to num_cams, or not drop percentage
        # is going to be higher
        list_valid_target_cars = []
        for car_id, camera_ids in self.list_of_cameras_per_car.items():
            if len(camera_ids) >= self.num_cams:
                list_valid_target_cars.append(car_id)
        # get the target car
        car_id = random.choice(list_valid_target_cars)
        self.target_car = random.choice(
            list(self.list_of_images_by_car_id[car_id]))

    def __create_similar_target_car(self, target_car):
        while True:
            similar_target_car = random.choice(
                list(self.list_of_images_by_car_id[target_car.car_id]))
            # WARNING: If the car is only taken by one camera, this will go into an infinite loop
            # prevent that from happening with this check
            if self.__is_taken_by_only_one_camera(target_car.car_id):
                break
            # make sure the car is taken by a different camera
            if target_car.camera_id != similar_target_car.camera_id:
                break
        return similar_target_car

    def __get_camset(self):
        num_imgs_per_camset = self.num_cars_per_cam
        camset = set()
        which_camera_id = random.choice(self.list_of_cameras)
        # determine whether or not to add a different target car image
        if not utils.should_drop(self.drop_percentage):
            num_imgs_per_camset = num_imgs_per_camset - 1
            similar_target_car = self.__create_similar_target_car(
                self.target_car)
            which_camera_id = similar_target_car.camera_id
            camset.add(similar_target_car)
        # grab images
        exist_car = list()
        for i in range(0, num_imgs_per_camset):
            while True:
                random_car = random.choice(
                    list(self.list_of_images_by_camera_id[which_camera_id]))
                # WARNING: If the dataset only contains one car, this will go into an infinite loop
                # prevent that from happening with this check
                if self.__is_only_one_car():
                    break
                # check if same car already existed
                if(random_car.car_id not in exist_car):
                    # different car
                    break
            exist_car.append(random_car.car_id)
            camset.add(random_car)
        return camset

    def generate(self):
        self.__set_lists()
        self.__set_target_car()
        output = list()
        for i in range(0, self.num_cams):
            output.append(self.__get_camset())
        return output


class Image(object):
    # assume image's name is in the format of
    # carId_cameraId_timestamp_binary.jpg

    def __init__(self, img_name, img_dir, img_type):
        self.name = img_name
        self.filepath = img_dir + "/" + img_name
        self.type = img_type
        self.__splitter = self.name.split("_")
        self.car_id = int(self.__splitter[0])
        self.camera_id = int(utils.get_numeric(self.__splitter[1]))
        self.timestamp = datetime.datetime.fromtimestamp(
            int(self.__splitter[2]))
        self.binary = int(os.path.splitext(self.__splitter[3])[0])

    def get_timestamp(self):
        # Year-Month-Date Hour:Minute:Second
        return self.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

# -----------------------------------------------------------------------------
#  Execution example
# -----------------------------------------------------------------------------


# @timewrapper
def main(args):
    # extract arguments from command line
    veri_unzipped_path = args.veri
    num_cams = args.num_cams
    num_cars_per_cam = args.num_cars_per_cam
    drop_percentage = args.drop_percentage
    typ = args.type
    seed = args.seed

    # check that input_path points to a directory
    if not os.path.exists(veri_unzipped_path) or not os.path.isdir(veri_unzipped_path):
        sys.exit("ERROR: filepath to VeRi directory (%s) is invalid" %
                 veri_unzipped_path)

    # create the generator
    exp = ExperimentGenerator(
        veri_unzipped_path, num_cams, num_cars_per_cam, drop_percentage, seed, typ)

    # generate the experiment
    set_num = 1
    print("=" * 80)
    for camset in exp.generate():
        print("Set #{}".format(set_num))
        print("Target car: {}".format(exp.target_car.name))
        print("-" * 80)
        for image in camset:
            print("name: {}".format(image.name))
            """
            print("filepath: {}".format(image.filepath))
            print("type: {}".format(image.type))
            print("car id: {}".format(image.car_id))
            print("camera id: {}".format(image.camera_id))
            print("timestamp: {}".format(image.get_timestamp()))
            print("binary: {}".format(image.binary))
            """
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
    parser.add_argument("veri", default="veri_unzipped_path", action="store", type=str,
                        help="Path to the VeRi dataset unzipped.")
    # options
    parser.add_argument("-v", "--version", action="version",
                        version="Experiment Generator 1.0")
    parser.add_argument("-s", dest="num_cams", action="store", type=int,
                        help="Each camera maps to a set.\nNUM_CAMS specifies the number of camera sets to be outputted.")
    parser.add_argument("-c", dest="num_cars_per_cam", action="store", type=int,
                        help="Each set has a list of car images. The cars in a set are distinct.\nNUM_CARS_PER_CAM specifies the number of car images in each camera set.")
    parser.add_argument("-d", dest="drop_percentage", action="store", type=float,
                        help="DROP_PERCENTAGE specifies the likelihood that\nthe target car image is dropped (float from [0,1]).")
    parser.add_argument("-y", dest="type", action="store", type=int,
                        help="TYPE determines which type of images to use.\n0: all, 1: query, 2: test, 3: train")
    parser.add_argument("-e", dest="seed", action="store", type=int,
                        default=random.randint(1, 100),
                        help="(OPTIONAL) SEED is used for random number generator.")

    main(parser.parse_args())
