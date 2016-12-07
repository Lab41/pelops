"""
datasets contain 
* 50,000 images
* 776 vehicles
* 20 cameras
* 1.0 km^2 area
* 24 hours
"""

import collections
import datetime
import os
import random
import re

class Veri(object):
    # class variables
    name_query_filepath = "name_query.txt"
    name_test_filepath = "name_test.txt"
    name_train_filepath = "name_train.txt"
    image_query_filepath = "image_query"
    image_test_filepath = "image_test"
    image_train_filepath = "image_train"
    num_cars = 776
    num_cams = 20 
    total_images = 49358 

class ExperimentGenerator(object):
    def __init__(self, veri_unzipped_path, num_cams, num_cars_per_cam, drop_percentage, seed):
        # set inputs
        self.set_filepaths(veri_unzipped_path)
        self.num_cams = num_cams
        self.num_cars_per_cam = num_cars_per_cam
        self.drop_percentage = drop_percentage
        self.seed = seed
        # stuff that needs to be initialized
        random.seed(seed)
        self.images = self.__get_images()
        self.list_of_cameras_per_car = collections.defaultdict(set)
        self.list_of_cars_per_camera = collections.defaultdict(set)
        self.list_of_cars = collections.defaultdict(list)

    def set_filepaths(self, veri_unzipped_path):
        self.name_query_filepath = veri_unzipped_path + Veri.name_query_filepath
        self.name_test_filepath = veri_unzipped_path + Veri.name_test_filepath
        self.name_train_filepath = veri_unzipped_path + Veri.name_train_filepath
        self.image_query_filepath = veri_unzipped_path + Veri.image_query_filepath
        self.image_test_filepath = veri_unzipped_path + Veri.image_test_filepath
        self.image_train_filepath = veri_unzipped_path + Veri.image_train_filepath

    def __merge_names(self, *filepaths):
        names = set()
        for filepath in filepaths:
            # determine image's type and directory
            if Veri.name_query_filepath in filepath:
                img_type = "query"
                img_dir = self.image_query_filepath
            elif Veri.name_test_filepath in filepath:
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
        return self.__merge_names(self.name_query_filepath, self.name_test_filepath, self.name_train_filepath)

    def get_images(self):
        return self.images

    def __unset_lists(self):
        self.list_of_cameras_per_car = collections.defaultdict(set)
        self.list_of_cars_per_camera = collections.defaultdict(set)
        self.list_of_cars = collections.defaultdict(list)

    def __set_lists(self):
        # TODO: figure out a better way to go about doing this
        # list_of_cameras_per_car: extract a list of distinct cameras that spot the car
        # list_of_cars_per_camera: extract a list of cars that the camera spots
        # 
        for image in self.images:
            car_id = image.get_car_id()
            camera_id = image.get_camera_id()
            self.list_of_cameras_per_car[car_id].add(camera_id)
            self.list_of_cars_per_camera[camera_id].add(car_id)
            self.list_of_cars[car_id].append(image)

    def __set_target_car(self):
        # count the number of times distinct cameras spot the car
        # has to be greater than equal to num_cams, or not drop percentage is going to be higher
        list_valid_target_cars = []
        for car_id, camera_ids in self.list_of_cameras_per_car.items():
            if len(camera_ids) >= self.num_cams:
                list_valid_target_cars.append(car_id)
        # return a car id
        return random.choice(list_valid_target_cars)

    def get_target_car(self):
        return self.target_car

    def __get_camset(self):
        num_imgs_per_camset = self.num_cars_per_cam
        camset = set()
        # determine whether or not to add the target car
        if not should_drop(self.drop_percentage):
            num_imgs_per_camset = num_imgs_per_camset - 1
            target_car_img = random.choice(self.list_of_cars[self.target_car])
            camset.add(target_car_img)
        # grab images
        random_cam = random.choice(list(self.list_of_cameras_per_car[self.target_car]))
        for i in range(0, num_imgs_per_camset):
            random_car = random.choice(list(self.list_of_cars_per_camera[random_cam]))
            random_car_img = random.choice(self.list_of_cars[random_car])
            camset.add(random_car_img)
        # return camset
        return camset

    def generate(self):
        self.__set_lists()
        self.target_car = self.__set_target_car()
        output = list()
        for i in range(0, self.num_cams):
            output.append(self.__get_camset())
        self.__unset_lists()
        return output

class Image(object):
    """ Assume image's name is in the format of carId_cameraId_timestamp_binary.jpg
    """
    def __init__(self, img_name, img_dir, img_type):
        self.name = img_name
        self.filepath = img_dir + "/" + img_name
        self.type = img_type
        self.__splitter = self.name.split("_")
        self.car_id = self.__set_car_id()
        self.camera_id = self.__set_camera_id()
        self.timestamp = self.__set_timestamp()
        self.binary = self.__set_binary()

    def get_name(self):
        return self.name

    def __set_car_id(self):
        car_id = int(self.__splitter[0])
        return car_id

    def get_car_id(self):
        return self.car_id

    def __set_camera_id(self):
        camera_id = int(get_numeric(self.__splitter[1]))
        return camera_id

    def get_camera_id(self):
        return self.camera_id

    def __set_timestamp(self):
        timestamp = datetime.datetime.fromtimestamp(int(self.__splitter[2]))
        return timestamp

    def get_timestamp(self):
        return self.timestamp

    def get_timestamp_in_str(self):
        # Year-Month-Date Hour:Minute:Second
        return self.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    def __set_binary(self):
        binary = int(os.path.splitext(self.__splitter[3])[0])
        return binary

    def get_binary(self):
        return binary

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

# -----------------------------------------------------------------------------
#  Helper Functions
# -----------------------------------------------------------------------------

def get_numeric(string):
    """ Extract the numeric value in a string.
    Args:
        string
    Returns:
        a string with only the numeric value extracted
    """
    return re.sub('[^0-9]','', string)

def should_drop(drop_percentage):
    """ Based on the given percentage, provide an answer 
    whether or not to drop the image.
    Args:
        drop_percentage: the likelihood of a drop 
    Returns:
        a boolean whether to drop or not drop the image
    """
    return random.randrange(100) < drop_percentage

def select_cameras(num_cams):
    """ Select camera ids based on the number of cameras specified.
    Args:
        num_cams: number of cameras 
    Returns:
        a list of camera ids
    """
    return random.sample(range(1, 21), num_cams)

# -----------------------------------------------------------------------------
#  Execution
# -----------------------------------------------------------------------------

def main():
    # inputs
    veri_unzipped_path = ""
    num_cams = 5
    num_cars_per_cam = 3
    drop_percentage = 30
    seed = 11

    # create the generator
    exp = ExperimentGenerator(veri_unzipped_path, num_cams, num_cars_per_cam, drop_percentage, seed)

    # generate the experiment
    """
    for i in exp.get_images():
        print("%s: %s: %s: %s: %s: %s: %s" % (i.name, i.filepath, i.type, i.car_id, i.camera_id, i.get_timestamp_in_str(), i.binary))
    print(len(exp.get_images()))
    """
    #print(exp.get_target_car())
    print(exp.generate())
    for camset in exp.generate():
        for i in camset:
            print("%s: %s: %s: %s: %s: %s: %s" % (i.name, i.filepath, i.type, i.car_id, i.camera_id, i.get_timestamp_in_str(), i.binary))

    return 

# -----------------------------------------------------------------------------
#  Entry
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    main()
