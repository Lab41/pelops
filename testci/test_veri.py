import pytest
import datetime
import os

import pelops.utils as utils
from pelops.datasets.veri import VeriDataset

@pytest.fixture
def veri(tmpdir):
    """ Set up some test files and an instance of VeriDataset() """
    # Write a file to read back
    FILE_NAMES = (
        # filepath, car_id, cam_id, time, misc
        ("0001_c001_00027065_0.jpg", 1, 1, datetime.datetime.fromtimestamp(int("00027065")), {"binary": 0}),
        ("0001_c002_00028680_0.jpg", 1, 2, datetime.datetime.fromtimestamp(int("00028680")), {"binary": 0}),
        ("0001_c003_00029105_0.jpg", 1, 3, datetime.datetime.fromtimestamp(int("00029105")), {"binary": 0}),
        ("0002_c001_00060920_1.jpg", 2, 1, datetime.datetime.fromtimestamp(int("00060920")), {"binary": 1}),
        ("0002_c002_00060935_1.jpg", 2, 2, datetime.datetime.fromtimestamp(int("00060935")), {"binary": 1}),
        ("0002_c003_00061525_1.jpg", 2, 3, datetime.datetime.fromtimestamp(int("00061525")), {"binary": 1}),
    )
    # The contents of the files do not matter, the name is enough
    name_query = tmpdir.join("name_query.txt")
    name_query.write("TEST")
    name_test = tmpdir.join("name_test.txt")
    name_train = tmpdir.join("name_train.txt")
    name_train.write("TEST")
    tmpdir.mkdir("image_query")
    image_test = tmpdir.mkdir("image_test")
    tmpdir.mkdir("image_train")
    gt_image = tmpdir.join("gt_image.txt")
    gt_image.write("TEST")
    jk_image = tmpdir.join("jk_image.txt")
    jk_image.write("TEST")
    train_label = tmpdir.join("train_label.txt")
    train_label.write("TEST")
    list_color = tmpdir.join("list_color.txt")
    list_color.write("TEST")
    list_type = tmpdir.join("list_type.txt")
    list_type.write("TEST")
    names = ""
    for name, _, _, _, _ in FILE_NAMES:
        names += name + "\n"
    name_test.write(names)

    # Setup the class
    instantiated_class = VeriDataset(name_test.dirname, utils.SetType.TEST.value)

    # Rename filepath
    FILE_NAMES = (
        # filepath, car_id, cam_id, time, misc
        (name_test.dirname + "/image_test/" + "0001_c001_00027065_0.jpg", 1, 1, datetime.datetime.fromtimestamp(int("00027065")), {"binary": 0}),
        (name_test.dirname + "/image_test/" + "0001_c002_00028680_0.jpg", 1, 2, datetime.datetime.fromtimestamp(int("00028680")), {"binary": 0}),
        (name_test.dirname + "/image_test/" + "0001_c003_00029105_0.jpg", 1, 3, datetime.datetime.fromtimestamp(int("00029105")), {"binary": 0}),
        (name_test.dirname + "/image_test/" + "0002_c001_00060920_1.jpg", 2, 1, datetime.datetime.fromtimestamp(int("00060920")), {"binary": 1}),
        (name_test.dirname + "/image_test/" + "0002_c002_00060935_1.jpg", 2, 2, datetime.datetime.fromtimestamp(int("00060935")), {"binary": 1}),
        (name_test.dirname + "/image_test/" + "0002_c003_00061525_1.jpg", 2, 3, datetime.datetime.fromtimestamp(int("00061525")), {"binary": 1}),
    )
    return (instantiated_class, FILE_NAMES)


def test_veri_chips_len(veri):
    """ Test that VeriDataset.chips is the correct length """
    instantiated_class = veri[0]
    FILE_NAMES = veri[1]
    # check that self.chips has been created, is not empty, and has the right
    # number of entries
    assert len(FILE_NAMES)
    assert len(FILE_NAMES) == len(instantiated_class.chips)


def test_veri_chips_vals(veri):
    """ Test that VeriDataset chips have the correct values. """
    instantiated_class = veri[0]
    FILE_NAMES = veri[1]

    # Check that the correct chips exist
    for filepath, car_id, cam_id, time, misc in FILE_NAMES:
        chip = instantiated_class.chips[filepath]
        assert car_id == chip.car_id
        assert cam_id == chip.cam_id
        assert time == chip.time
        # No misc data
        assert misc["binary"] == chip.misc["binary"]
        # Filepath should be filled
        assert chip.filepath


def test_get_all_chips_by_car_id(veri):
    """ Test VeriDataset.get_all_chips_by_car_id() """
    instantiated_class = veri[0]
    FILE_NAMES = veri[1]

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


def test_get_all_chips_by_cam_id(veri):
    """ Test VeriDataset.get_all_chips_by_cam_id() """
    instantiated_class = veri[0]
    FILE_NAMES = veri[1]

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


def test_get_distinct_cams_by_car_id(veri):
    """ Test VeriDataset.get_distinct_cams_by_car_id() and get_distinct_cams_per_car """
    instantiated_class = veri[0]
    CAR_ID = 1
    TEST_CAMS = [1, 2, 3]
    for test_cam, cam in zip(TEST_CAMS, sorted(instantiated_class.get_distinct_cams_by_car_id(CAR_ID))):
        assert test_cam == cam


def test_get_all_cam_ids(veri):    
    """ Test VeriDataset.get_distinct_cams_by_car_id() """
    instantiated_class = veri[0]
    TEST_CAMS = [1, 2, 3]
    for test_cam, cam in zip(TEST_CAMS, sorted(instantiated_class.get_all_cam_ids())):
        assert test_cam == cam


def test_get_all_car_ids(veri):
    """ Test VeriDataset.get_distinct_cams_by_car_id() """
    instantiated_class = veri[0]
    TEST_CARS = [1, 2]
    for test_car, car in zip (TEST_CARS, sorted(instantiated_class.get_all_car_ids())):
        assert test_car == car


def test_veri_iter(veri):
    """ Test VeriDataset.__iter__() """
    instantiated_class = veri[0]
    FILE_NAMES = veri[1]
    chip_ids = tuple(i for i, _, _, _, _ in FILE_NAMES)

    for chip in instantiated_class:
        assert chip.filepath in chip_ids