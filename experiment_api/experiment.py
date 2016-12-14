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
    This script will output a list of sets based on the number of cameras specified.
    Within each set, it will contain a list of images based on the number of cars per camera specified.
    Each set contains car images taken by the same camera.
    These images are stored inside an Image object.
    The target car, which will be randomly selected, will exist in each set depending on the drop rate.
    Currently, each set contains distinct car ids. If you uncommented the code section in
    ExperimentGenerator's __get_camset(), it will generate a set with repeated car ids that are at least
    MINUTES apart specified by the user.
    For example:
    * num_cams = 10, which means we will create 10 camera sets
    * num_cars_per_cam = 5, which means we will have 5 Image objects within the set
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
    experiment.py -s <NUM_CAMS> -c <NUM_CARS_PER_CAM> -d <DROP_PERCENTAGE> -t <MINUTES> -e <SEED> -y <TYPE> <INPUT_PATH>

Arguments:
    INPUT_PATH                      : Path to the VeRi dataset unzipped

Options:
    -h, --help                      : Show this help message.
    -v, --version                   : Show the version number.
    -s, --cams=<NUM_CAMS>           : Each camera maps to a set. NUM_CAMS specify the number of camera sets to be outputted.
    -c, --cars=<NUM_CARS_PER_CAM>   : Each set has a list of images. NUM_CARS_PER_CAM specify the number of car images in each camera set.
    -d, --drop=<DROP_PERCENTAGE>    : The likelihood that the target car image is dropped (float from [0,1])
    -e, --seed=<SEED>               : Seed to be used for random number generator.
    -t, --time=<MINUTES>            : If the same car image exists in a set, only allows it after a certain amount of time in minutes.
    -y, --type=<TYPE>               : Determine which type of images to use.
                                      0: all, 1: query, 2: test, 3: train

"""

import collections
import datetime
import docopt
import os
import random
import sys
import utils


class ExperimentGenerator(object):
    def __init__(self, veri_unzipped_path, num_cams, num_cars_per_cam, drop_percentage, seed, time, typ):
        # set inputs
        self.set_filepaths(veri_unzipped_path)
        self.num_cams = num_cams
        self.num_cars_per_cam = num_cars_per_cam
        self.drop_percentage = drop_percentage
        self.seed = seed
        self.time = time
        self.typ = typ
        # stuff that needs to be initialized
        random.seed(self.seed)
        self.list_of_cameras_per_car = collections.defaultdict(set)
        self.list_of_cameras = list()
        self.list_of_images_by_car_id = collections.defaultdict(set)
        self.list_of_images_by_camera_id = collections.defaultdict(set)

    def set_filepaths(self, veri_unzipped_path):
        self.name_query_filepath = veri_unzipped_path + "/" + utils.Veri.name_query_filepath
        self.name_test_filepath = veri_unzipped_path + "/" + utils.Veri.name_test_filepath
        self.name_train_filepath = veri_unzipped_path + "/" + utils.Veri.name_train_filepath
        self.image_query_filepath = veri_unzipped_path + "/" + utils.Veri.image_query_filepath
        self.image_test_filepath = veri_unzipped_path + "/" + utils.Veri.image_test_filepath
        self.image_train_filepath = veri_unzipped_path + "/" + utils.Veri.image_train_filepath

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
            this_set = set(Image(name.strip(), img_dir, img_type) for name in open(filepath))
            # combine with the previous list of names
            names = names.union(this_set)
        return names

    def __get_images(self):
        return {
            utils.ImageType.ALL: self.__merge_names(self.name_query_filepath, self.name_test_filepath, self.name_train_filepath),
            utils.ImageType.QUERY: self.__merge_names(self.name_query_filepath),
            utils.ImageType.TEST: self.__merge_names(self.name_test_filepath),
            utils.ImageType.TRAIN: self.__merge_names(self.name_train_filepath),
        }.get(self.typ)
        # }.get(self.typ, 0) # default to all

    def __unset_lists(self):
        self.list_of_cameras_per_car = collections.defaultdict(set)
        self.list_of_cameras = list()
        self.list_of_images_by_car_id = collections.defaultdict(set)
        self.list_of_images_by_camera_id = collections.defaultdict(set)

    def __set_lists(self):
        # TODO: figure out a better way to go about doing this
        # list_of_cameras_per_car: map each car with a list of distinct cameras that spot the car
        # list_of_car_names_per_camera: map each camera with a list of cars it spots
        # list_of_images_by_car_id: map each car image to its respective car id
        # list_of_images_by_camera_id: map each car image to its respective camera id
        for image in self.__get_images():
            car_id = image.car_id
            camera_id = image.camera_id
            # list needed for finding target cars
            self.list_of_cameras_per_car[car_id].add(camera_id)
            self.list_of_images_by_car_id[car_id].add(image)
            # list needed for finding random cars
            self.list_of_cameras.append(camera_id)
            self.list_of_images_by_camera_id[camera_id].add(image)
            cars.add(car_id)

    def __is_only_one_car(self):
        return len(self.list_of_images_by_car_id.keys()) == 1

    def __is_taken_by_only_one_camera(self, car_id):
        return len(self.list_of_cameras_per_car[car_id]) == 1

    def __set_target_car(self):
        # count the number of times distinct cameras spot the car
        # has to be greater than or equal to num_cams, or not drop percentage is going to be higher
        list_valid_target_cars = []
        for car_id, camera_ids in self.list_of_cameras_per_car.items():
            if len(camera_ids) >= self.num_cams:
                list_valid_target_cars.append(car_id)
        # get the target car
        car_id = random.choice(list_valid_target_cars)
        self.target_car = random.choice(list(self.list_of_images_by_car_id[car_id]))

    def __get_camset(self):
        num_imgs_per_camset = self.num_cars_per_cam
        camset = set()
        which_camera_id = random.choice(self.list_of_cameras)
        # determine whether or not to add a different target car image
        if not utils.should_drop(self.drop_percentage):
            num_imgs_per_camset = num_imgs_per_camset - 1
            while True:
                similar_target_car = random.choice(list(self.list_of_images_by_car_id[self.target_car.car_id]))
                # WARNING: If the car is only taken by one camera, this will go into an infinite loop
                # prevent that from happening with this check
                if self.__is_taken_by_only_one_camera(self.target_car.car_id):
                    break
                # make sure the car is taken by a different camera
                if self.target_car.camera_id != similar_target_car.camera_id:
                    break
            which_camera_id = similar_target_car.camera_id
            camset.add(similar_target_car)
        # grab images
        exist_car = list()
        # exist_timestamp = list()
        for i in range(0, num_imgs_per_camset):
            while True:
                random_car = random.choice(list(self.list_of_images_by_camera_id[which_camera_id]))
                # WARNING: If the dataset only contains one car, this will go into an infinite loop
                # prevent that from happening with this check
                if self.__is_only_one_car:
                    break
                # check if same car already existed
                if(random_car.car_id not in exist_car):
                    # different car
                    break
                """
                else:
                    # same car already exists, make sure the timestamp is greater than 5 minutes
                    old_timestamp = exist_timestamp[exist_car.index(random_car.car_id)]
                    new_timestamp = random_car.timestamp
                    valid_timestamp = abs(new_timestamp - old_timestamp) / datetime.timedelta(minutes=1)
                    if valid_timestamp > self.time:
                        break
                """
            exist_car.append(random_car.car_id)
            # exist_timestamp.append(random_car.timestamp)
            camset.add(random_car)
            # see if the image added is the same and if it is check the timestamp
        # return camset
        return camset

    def generate(self):
        self.__set_lists()
        self.__set_target_car()
        output = list()
        for i in range(0, self.num_cams):
            output.append(self.__get_camset())
        self.__unset_lists()
        return output


class Image(object):
    # assume image's name is in the format of carId_cameraId_timestamp_binary.jpg
    def __init__(self, img_name, img_dir, img_type):
        self.name = img_name
        self.filepath = img_dir + "/" + img_name
        self.type = img_type
        self.__splitter = self.name.split("_")
        self.car_id = int(self.__splitter[0])
        self.camera_id = int(utils.get_numeric(self.__splitter[1]))
        self.timestamp = datetime.datetime.fromtimestamp(int(self.__splitter[2]))
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
    try:
        veri_unzipped_path = args["<INPUT_PATH>"]
        num_cams = int(args["--cams"])
        num_cars_per_cam = int(args["--cars"])
        drop_percentage = float(args["--drop"])
        seed = int(args["--seed"])
        time = int(args["--time"])
        typ = int(args["--type"])
    except docopt.DocoptExit as e:
        sys.exit("ERROR: input invalid options: %s" % e)

    # check that input_path points to a directory
    if not os.path.exists(veri_unzipped_path) or not os.path.isdir(veri_unzipped_path):
        sys.exit("ERROR: filepath to VeRi directory (%s) is invalid" % veri_unzipped_path)

    # create the generator
    exp = ExperimentGenerator(veri_unzipped_path, num_cams, num_cars_per_cam, drop_percentage, seed, time, typ)

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
    args = docopt.docopt(__doc__, version="Experiment Generator 1.0")
    main(args)
