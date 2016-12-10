import datetime
import json
import logging
import random
import re
import time

class Veri(object):
    """ Structure of the Veri dataset unzipped and miscellaneous information
    """
    name_query_filepath = "name_query.txt"
    name_test_filepath = "name_test.txt"
    name_train_filepath = "name_train.txt"
    image_query_filepath = "image_query"
    image_test_filepath = "image_test"
    image_train_filepath = "image_train"
    ground_truth_filepath = "gt_image.txt"
    junk_images_filepath = "jk_image.txt"
    train_label_filepath = "train_label.xml"
    num_cars = 776
    num_cams = 20
    num_query_images = 1679
    num_test_images = 11580
    num_train_images = 37779
    total_images = 49358


class ImageType(object):
    """ Types of images
    """
    ALL = 0
    QUERY = 1
    TEST = 2
    TRAIN = 3


def get_numeric(string):
    """ Extract the numeric value in a string.
    Args:
        string
    Returns:
        a string with only the numeric value extracted
    """
    return re.sub('[^0-9]', '', string)


def should_drop(drop_percentage):
    """ Based on the given percentage, provide an answer
    whether or not to drop the image.
    Args:
        drop_percentage: the likelihood of a drop in the form of a float from [0,1]
    Returns:
        a boolean whether to drop or not drop the image
    """
    return random.random() < drop_percentage


def read_camera_id(camera_id):
    """ Assuming that camera_id is a string in the format of c001,
    convert camera_id into an int.
    Args:
        camera_id: string in the format of c001
    Returns:
        an int value of camera_id
    """
    return int(get_numeric(camera_id))


def read_car_id(car_id):
    """ Assuming that car_id is a string in the format of 0001,
    convert car_id into an int.
    Args:
        car_id: string in the format of 0001
    Returns:
        an int value of car_id
    """
    return int(car_id)


def read_timestamp(timestamp):
    """ Assuming that timestamp is a unix timestamp string,
    convert it into a datetime object.
    """
    return datetime.datetime.fromtimestamp(int(timestamp))


def read_json(filepath):
    """ Assuming the json file contains a dictionary per line,
    read the json file and create a generator that yields each
    dictionary per line.
    Args:
        filepath: path to the json file
    Returns:
        a generator that yields dictionary line by line
    """
    with open(filepath) as file:
        for line in file:
            yield json.loads(line)


def timeit(func, *args, **kwargs):
    """ This is a wrapper function to calculate how fast each operation takes.
    Note that we are assuming that we do not need to do anything with the return
    value of the function.
    Args: 
        func: function pointer
        args: arguments to the function
        kwargs: named arguments not defined in advance to be passed in to the function
    """
    start = time.time()
    func(*args, **kwargs)
    elapsed = time.time() - start
    return elapsed
