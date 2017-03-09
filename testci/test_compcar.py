import pytest
import datetime
import numpy as np
import scipy.io
import os
from shutil import copyfile

import pelops.utils as utils
from pelops.datasets.compcar import CompcarDataset

@pytest.fixture
def compcar(tmpdir):
    """ Set up some test files and an instance of CompcarDataset() """
    # Write a file to read back
    FILE_NAMES = (
        # filepath, car_id, cam_id, time, misc
        ("1/asdasf1123123.jpg", 1, None, None, {"color": "blue", "make": "BMW", "model": "X5", "model_id": 105}),
        ("1/qjfas123189798.jpg", 1, None, None, {"color": "black", "make": "BMW", "model": "X5", "model_id": 105}),
        ("2/345sdjkhjlsh33.jpg", 2, None, None, {"color": "red", "make": "Zotye", "model": "Z300", "model_id": 1035}),
        ("3/werfsdbfuw3223.jpg", 3, None, None, {"color": "yellow", "make": "Hyundai", "model": "Santafe", "model_id": 961}),
        ("3/asdfj21348wesd.jpg", 3, None, None, {"color": "champagne", "make": "Hyundai", "model": "Santafe", "model_id": 961}),
        ("4/kjdfgjhlsdg322.jpg", 4, None, None, {"color": "champagne", "make": "Toyota", "model": "Crown", "model_id": 1322}),
    )
    # The contents of the files do not matter, the name is enough
    name_test = tmpdir.join("test_surveillance.txt")
    name_train = tmpdir.join("train_surveillance.txt")
    name_train.write("TEST")
    tmpdir.mkdir("image")
    model_mat = tmpdir.join("sv_make_model_name.mat")
    model_matrix = np.array([
        # make, model, web_id
        [["BWM"], ["BWM X5"], 105],
        [["Zoyte"], ["Zotye Z300"], 1035],
        [["Hyundai"], ["Santafe"], 961],
        [["Toyota"], ["Crown"], 1322],
    ])
    scipy.io.savemat(model_mat.dirname + "/sv_make_model_name.mat", mdict={"sv_make_model_name": model_matrix})
    color_mat = tmpdir.join("color_list.mat")
    color_matrix = np.array([
        # filepath, color_num
        [["1/asdasf1123123.jpg"], 4],
        [["1/qjfas123189798.jpg"], 0],
        [["2/345sdjkhjlsh33.jpg"], 2],
        [["3/werfsdbfuw3223.jpg"], 3],
        [["3/asdfj21348wesd.jpg"], 8],
        [["4/kjdfgjhlsdg322.jpg"], 8],
    ])
    scipy.io.savemat(color_mat.dirname + "/color_list.mat", mdict={"color_list": color_matrix})
    names = ""
    for name, _, _, _, _ in FILE_NAMES:
        names += name + "\n"
    name_test.write(names)

    # Setup the class
    instantiated_class = CompcarDataset(name_test.dirname, utils.SetType.TEST)

    # Rename filepath
    FILE_NAMES = (
        # filepath, car_id, cam_id, time, misc
        (name_test.dirname + "/image/" + "1/asdasf1123123.jpg", 1, None, None, {"color": "blue", "make": "BMW", "model": "X5", "model_id": 105}),
        (name_test.dirname + "/image/" + "1/qjfas123189798.jpg", 1, None, None, {"color": "black", "make": "BMW", "model": "X5", "model_id": 105}),
        (name_test.dirname + "/image/" + "2/345sdjkhjlsh33.jpg", 2, None, None, {"color": "red", "make": "Zotye", "model": "Z300", "model_id": 1035}),
        (name_test.dirname + "/image/" + "3/werfsdbfuw3223.jpg", 3, None, None, {"color": "yellow", "make": "Hyundai", "model": "Santafe", "model_id": 961}),
        (name_test.dirname + "/image/" + "3/asdfj21348wesd.jpg", 3, None, None, {"color": "champagne", "make": "Hyundai", "model": "Santafe", "model_id": 961}),
        (name_test.dirname + "/image/" + "4/kjdfgjhlsdg322.jpg", 4, None, None, {"color": "champagne", "make": "Toyota", "model": "Crown", "model_id": 1322}),
    )
    return (instantiated_class, FILE_NAMES)


def test_compcar_chips_len(compcar):
    """ Test that CompcarDataset.chips is the correct length """
    instantiated_class = compcar[0]
    FILE_NAMES = compcar[1]
    # check that self.chips has been created, is not empty, and has the right
    # number of entries
    assert len(FILE_NAMES)
    assert len(FILE_NAMES) == len(instantiated_class.chips)


def test_compcar_chips_vals(compcar):
    """ Test that CompcarDatset chips have the correct values. """
    instantiated_class = compcar[0]
    FILE_NAMES = compcar[1]

    # Check that the correct chips exist
    for filepath, car_id, cam_id, time, misc in FILE_NAMES:
        chip = instantiated_class.chips[filepath]
        assert car_id == chip.car_id
        assert cam_id is None
        assert time is None
        assert misc["color"] == chip.misc["color"]
        assert misc["make"] == chip.misc["make"]
        assert misc["model"] == chip.misc["model"]
        assert misc["model_id"] == chip.misc["model_id"]

        # Filepath should be filled
        assert chip.filepath


def test_get_all_chips_by_car_id(compcar):
    """ Test CompcarDatset.get_all_chips_by_car_id() """
    instantiated_class = compcar[0]
    FILE_NAMES = compcar[1]

    seen_ids = []
    for filepath, car_id, cam_id, time, misc in FILE_NAMES:
        # Generate all the chips by hand, and compare
        if car_id in seen_ids:
            continue
        seen_ids.append(car_id)
        chips = []
        for key, val in instantiated_class.chips.items():
            if val.car_id == car_id:
                chips.append(val)

        chips.sort()
        test_chips = sorted(instantiated_class.get_all_chips_by_car_id(car_id))
        assert chips == test_chips


def test_get_all_chips_by_cam_id(compcar):
    """ Test CompcarDatset.get_all_chips_by_cam_id() """
    instantiated_class = compcar[0]
    FILE_NAMES = compcar[1]

    seen_ids = []
    for filepath, car_id, cam_id, time, misc in FILE_NAMES:
        # Generate all the chips by hand, and compare
        if cam_id in seen_ids:
            continue
        seen_ids.append(cam_id)
        chips = []
        for key, val in instantiated_class.chips.items():
            if val.cam_id == cam_id:
                chips.append(val)

        chips.sort()
        test_chips = sorted(instantiated_class.get_all_chips_by_cam_id(cam_id))
        assert chips == test_chips


def test_get_distinct_cams_by_car_id(compcar):
    """ Test CompcarDatset.get_distinct_cams_by_car_id() and get_distinct_cams_per_car """
    instantiated_class = compcar[0]
    CAR_ID = 1
    TEST_CAMS = []
    for test_cam, cam in zip(TEST_CAMS, sorted(instantiated_class.get_distinct_cams_by_car_id(CAR_ID))):
        assert test_cam == cam


def test_get_all_cam_ids(compcar):    
    """ Test CompcarDatset.get_distinct_cams_by_car_id() """
    instantiated_class = compcar[0]
    TEST_CAMS = []
    for test_cam, cam in zip(TEST_CAMS, sorted(instantiated_class.get_all_cam_ids())):
        assert test_cam == cam


def test_get_all_car_ids(compcar):
    """ Test CompcarDatset.get_distinct_cams_by_car_id() """
    instantiated_class = compcar[0]
    TEST_CARS = [1, 2, 3, 4]
    for test_car, car in zip (TEST_CARS, sorted(instantiated_class.get_all_car_ids())):
        assert test_car == car


def test_compcar_iter(compcar):
    """ Test CompcarDatset.__iter__() """
    instantiated_class = compcar[0]
    FILE_NAMES = compcar[1]
    chip_ids = tuple(i for i, _, _, _, _ in FILE_NAMES)

    for chip in instantiated_class:
        assert chip.filepath in chip_ids
