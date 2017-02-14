""" utilities when working with cameras"""

from collections import defaultdict


def nameit_cam(first, second):
    """
    concatenate chip names together in a seperable way
    first(chip) - first item
    second(chip) - second item
    """
    return '{}|{}'.format(first.cam_id, second.cam_id)


def nameit_car(first, second):
    """
    concatenate chip.car names together in a seperable way
    first(chip) - first item
    second(chip) - second imte
    """
    return '{}|{}'.format(first.car_id, second.car_id)


def get_match_id(cameras):
    """
    find the car of interest from a set of cameras

    cameras(list(list(chips)))): list of the cameras with cars in each camera
    """
    chosendict = defaultdict(int)
    for camera in cameras:
        for car in camera:
            chosendict[car.car_id] += 1
    mymax = -1
    myid = None
    for k in chosendict.keys():
        if chosendict[k] > mymax:
            mymax = chosendict[k]
            myid = k
    return myid


def make_good_bad(cameras, car_id):
    """
    make a list of cars of interest, and a list of other

    cameras(list(list(chips))): list of the cameras with the cars in each cameras
    car_id(): the id of the car of interest
    """
    goodlist = list()
    bad_list = list()
    for camera in cameras:
        for car in camera:
            if car.car_id == car_id:
                goodlist.append(car)
            else:
                bad_list.append(car)
    return (goodlist, bad_list)


def glue(vec_a, vec_b):
    """
    concatenate two smaller vectors to a larger vector
    vec_a : first vector
    vec_b : second vector
    """
    retval = list()
    retval.extend(vec_a)
    retval.extend(vec_b)
    return retval
