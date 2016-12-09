""" Ranker
"""

import collections
import matplotlib.pyplot as plt
import re
import scipy
import scipy.spatial

from experiment import ExperimentGenerator
from utils import * 

def main():

    veri_unzipped_path = "/net/dev-vsrv-fs4.b.internal/data1/teams/pelops/veri_unzipped"
    num_cams = 1
    num_cars_per_cam = 10
    drop_percentage = 0
    seed = 11
    time = 5
    typ = 2
    exp = ExperimentGenerator(veri_unzipped_path, num_cams, num_cars_per_cam, drop_percentage, seed, time, typ)
    rank = list()
    counter = 1

    print("after experiment generator")
    first_ranks = list()
    for i in range(0, 10000): 
        first_rank = get_first_rank(exp)
        first_ranks.append(first_rank)

    print("calculate num_ranks")
    num_ranks = collections.Counter(first_ranks)
    print(num_ranks)

    plot(num_ranks)

def plot(num_ranks):
    x = sorted(num_ranks.keys())
    print("x", x)
    y = get_y(num_ranks)
    print("y", y)

    plt.plot(x, y, 'ro')
    plt.xlabel("rank")
    plt.ylabel("number of ranks")
    plt.axis([1, max(x), 1, max(y)])
    plt.show()
    return

def get_y(num_ranks):
    y = list()
    total_ranks = len(num_ranks)

    for key in sorted(num_ranks.keys()):
        y.append(num_ranks[key] / total_ranks)
    return y

def get_first_rank(exp):

    exp_generator = exp.generate()
    target_car_vector = match_target_car_vector(exp.target_car)
    rank = list()
    counter = 1
    print("start get_first_rank")
    for camset in exp_generator:
        for image in camset:
            match_vector(image)
            print("image.vector = ")
            print(image.vector)
            image_cos = scipy.spatial.distance.cosine(target_car_vector, image.vector)
            rank.append((counter, image_cos))
            counter = counter + 1

    print("before sort")
    print(rank)
    # sort 
    sorted(rank, key=lambda tupl: tupl[1])
    print("after sort")
    print(rank)

    print("end get_first_rank")

    # return the first item in the sort and give the rank number
    return rank[0][0]
 
def match_vector(image):

    train = "/net/dev-vsrv-fs4.b.internal/data1/teams/pelops/veri_features/training_features.json"
    test = "/net/dev-vsrv-fs4.b.internal/data1/teams/pelops/veri_features/test_features.json"

    print("start match_vector")
    objs = read_json(test)
    print(objs)
    for obj in objs:
        if read_car_id(obj["vehicleID"]) == image.car_id and read_camera_id(obj["cameraID"]) == image.camera_id:
            print("obj[resnet50]", obj["resnet50"])
            image.set_vector(obj["resnet50"])
            return

        print("obj[vehicleID]", read_car_id(obj["vehicleID"]), image.car_id)
        print("obj[cameraID]", read_camera_id(obj["cameraID"]), image.camera_id)

    print("finish match vector")


def match_target_car_vector(target_car_id):
    train = "/net/dev-vsrv-fs4.b.internal/data1/teams/pelops/veri_features/training_features.json"
    test = "/net/dev-vsrv-fs4.b.internal/data1/teams/pelops/veri_features/test_features.json"

    print("start match_vector")
    objs = read_json(test)
    print(objs)
    for obj in objs:
        if read_car_id(obj["vehicleID"]) == target_car_id:
            return obj["resnet50"]

        print("obj[vehicleID]", read_car_id(obj["vehicleID"]), target_car_id)

    print("finish match vector")


# -----------------------------------------------------------------------------
#  Entry
# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
