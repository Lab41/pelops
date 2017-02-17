import cProfile
import datetime
import enum
import json
import logging
import os
import random
import re
import time
import sys


class SetType(enum.Enum):
    """ Types of set, i.e. training set
    """
    ALL = 0
    QUERY = 1
    TEST = 2
    TRAIN = 3


def setup_custom_logger(name):
    """ Setup a custom logger that will output to the console
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # create a file handler
    file_handler = logging.FileHandler("./log_{}".format(name))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


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


def get_basename(string):
    """ Extract the basename from the filepath.
    Args:
        filepath in the format of a string
    Args:
        filename in the format of a string
    """
    return os.path.basename(os.path.normpath(string))


def get_numeric(string):
    """ Extract the numeric value in a string.
    Args:
        string
    Returns:
        a string with only the numeric value extracted
    """
    return re.sub('[^0-9]', '', string)


def get_timestamp(timestamp):
    """ Convert datetime object into a string in the format of
    Year-Month-Date Hour:Minute:Second
    Args: 
        datetime 
    Returns: 
        string in the format of Year-Month-Date Hour:Minute:Second
    """
    return timestamp.strftime("%Y-%m-%d %H:%M:%S") if isinstance(type(timestamp), datetime.datetime) else timestamp


def should_drop(drop_percentage):
    """ Based on the given percentage, provide an answer
    whether or not to drop the image.
    Args:
        drop_percentage: the likelihood of a drop in the form of a float from [0,1]
    Returns:
        a boolean whether to drop or not drop the image
    """
    return random.random() < drop_percentage


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
