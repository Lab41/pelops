import cProfile
import datetime
import enum
import json
import os
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


class ImageType(enum.Enum):
    """ Types of images
    """
    ALL = 0
    QUERY = 1
    TEST = 2
    TRAIN = 3


def get_index_of_tuple(list_of_tuple, index_of_tuple, value):
    """ Determine how far through the list to find the value.
    If the value does not exist in the list, then return the
    length of the list.
    Args:
        list_of_tuple: a list of tuples i.e. [(index1, index2, index3)]
        index_of_tuple_1: which index in the tuple you want to compare the value to
        value: the value to search
    Return:
        the number of items in the list it has compared
    """
    for index_of_list, tupl in enumerate(list_of_tuple):
        if tupl[index_of_tuple] == value:
            return index_of_list + 1
    # could not find value in list_of_tuple, so return length of tuple
    return len(list_of_tuple)


def get_index_of_pairs(list_of_tuple, index_of_tuple_1, index_of_tuple_2, value):
    """ Determine how far through the list to find the value.
    If the value does not exist in the list, then return the
    length of the list.
    Args:
        list_of_tuple: a list of tuples i.e. [(index1, index2, index3)]
        index_of_tuple_1: which index in the tuple you want to compare the value to
        index_of_tuple_2: which index in the tuple you want to compare the value to
        value: the value to search
    Return:
        the number of items in the list it has compared
    """
    for index_of_list, tupl in enumerate(list_of_tuple):
        if tupl[index_of_tuple_1] == value and tupl[index_of_tuple_2] == value:
            return index_of_list + 1
    # could not find value in list_of_tuple, so return length of tuple
    return len(list_of_tuple)


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


def read_camera_id(name):
    """ Assuming that name is a string in the format of 0002_c002_00030670_0.jpg,
    find the camera_id in name and convert it into an int.
    Args:
        name: string in the format of 0002_c002_00030670_0.jpg
    Returns:
        an int value of camera_id
    """
    splitter = name.split("_")
    return int(get_numeric(splitter[1]))


def read_car_id(name):
    """ Assuming that name is a string in the format of 0002_c002_00030670_0.jpg,
    find the car_id in name and convert it into an int.
    Args:
        name: string in the format of 0002_c002_00030670_0.jpg
    Returns:
        an int value of car_id
    """
    splitter = name.split("_")
    return int(splitter[0])


def read_timestamp(name):
    """ Assuming that name is a string in the format of 0002_c002_00030670_0.jpg,
    find the timestamp in name and convert it into a datetime object.
    Args:
        name: string in the format of 0002_c002_00030670_0.jpg
    Returns:
        a datetime object of timestamp
    """
    splitter = name.split("_")
    return datetime.datetime.fromtimestamp(int(splitter[2]))


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


def remove_file(filename):
    """ Assuming the filename exists where the application is run,
    remove the file.
    Args:
        filename
    Returns:
        filename is removed
    """
    try:
        os.remove(filename)
    except OSError:
        pass


def timewrapper(func):
    """ This is a decorator to calculate how fast each operation takes.
    Args:
        func: function pointer
        args: arguments to the function
        kwargs: named arguments not defined in advance to be passed in to the function
    """
    def timer(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print("{} took {} seconds".format(func.__name__, elapsed))
        return result
    return timer


def profilewrapper(func):
    """ This is a decorator to profile a function.
    Args:
        func: function pointer
        args: arguments to the function
        kwargs: named arguments not defined in advance to be passed in to the function
    """
    def profiler(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiler
