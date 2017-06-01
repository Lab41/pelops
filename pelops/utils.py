import cProfile
import datetime
import enum
import json
import logging
import os
import random
import re
import time
import itertools
import csv


class SetType(enum.Enum):
    """ Types of set, i.e. training set
    """
    ALL = "all"
    QUERY = "query"
    TEST = "test"
    TRAIN = "train"

    
def get_session(gpu_fraction=0.3):
    import tensorflow as tf # Shadow import for testing
    """
    Helper function to ensure that Keras only uses some fraction of the memory
    Args:
        gpu_fraction: Fraction of the GPU memory to use

    Returns:
        A tensorflow session to be passed into tensorflow_backend.set_session
    """

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

      
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


def get_split(key, pivots):
    if not isinstance(pivots, list):
        pivots = list(pivots)

    pivots.sort()
    hash_val = hash(key)%100
    for split, pivot in enumerate(pivots):
        if hash_val < pivot:
            return split
    return len(pivots)


def train_test_key_filter(key, split="train"):
    hash_val = get_split(key, [90])
    split = split.lower()
    if split == "train":
        desired_val = 0
    elif split == "test":
        desired_val = 1
    else:
        raise ValueError('Unknown Split Type: %s'%split)

    if hash_val == desired_val:
        return True
    else:
        return False


def prep_for_siamese(*csv_files, json_file='./out.json', full_combos=False):
    """
    Prepares a json file containing pairwise feature vectors for input to the siamese docker container.
    
    :param csv_files: List of CSV files containing i2v produced feature vectors 
    :param json_file: Optional output json file path
    :param full_combos: Boolean indcating whether full combinations or observation set cartesian product should be used.
    """

    # Generator for csv rows from a single csv file.
    def iter_rows(csv_file):
        with open(csv_file, newline='') as csv_hdl:
            for row in csv.reader(csv_hdl):
                yield row

    # Generator for flattened access to rows from multiple csv files.
    def iter_many(row_gens):
        for gen in row_gens:
            for row in gen:
                yield row

    if len(csv_files) == 1:
        if not full_combos:
            raise NotImplemented("Full combinations must be applied if only one csv is supplied")
        combos = itertools.combinations(iter_rows(csv_files[0]), 2)
    else:
        if full_combos:
            combos = itertools.combinations(iter_many(map(iter_rows, csv_files)), 2)
        elif len(csv_files) == 2:
            combos = itertools.product(iter_rows(csv_files[0]), iter_rows(csv_files[1]))
        else:
            raise NotImplemented("Full combinations must be applied if most than two csvs are supplied")

    with open(json_file, 'w') as json_hdl:
        for left, right in combos:
            try:
                dct = {
                    'left': list([float(v) for v in left[1:]]),
                    'right': list([float(v) for v in right[1:]]),
                    'left_img': left[0],
                    'right_img': right[0]
                }
                json_hdl.write(json.dumps(dct) + '\n')
            except IOError:
                raise IOError("Error occurred writing vectors to json")
