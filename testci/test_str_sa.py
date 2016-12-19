import pytest

from pelops.datasets.str_sa import get_sa_cam_id
from pelops.datasets.str_sa import get_sa_car_id
from pelops.datasets.str_sa import int_from_string
from pelops.datasets.str_sa import STR_SA


def test_str_sa_construction_raise():
    """ Test that STR_SA() raises an error when not given a directory. """
    with pytest.raises(ValueError):
        STR_SA()


def test_int_from_string():
    """ Test int_from_string() """
    TEST_STRINGS = (
        # String, Args, Answer
        ("test_010_test", ("test_", 3), 10),
        ("test_010_test", ("FAIL_", 3), None),
        ("test_010",      ("test_", 3), 10),
        ("test_11_test",  ("test_", 2), 11),
        ("010_test",      ("",      3), 10),
    )

    for test_string, args, answer in TEST_STRINGS:
        assert answer == int_from_string(test_string, args[0], args[1])


def test_get_sa_cam_id():
    """ Test get_sa_cam_id() """
    TEST_STRINGS = (
        # String, Answer
        ("match00001_cam02.png",      2),
        ("match00001_cam01_mask.png", 1),
        ("match00010_cam01.png",      1),
        ("match00011_cam02_mask.png", 2),
    )

    for test_string, answer in TEST_STRINGS:
        assert answer == get_sa_cam_id(test_string)


def test_get_sa_car_id():
    """ Test get_sa_car_id() """
    TEST_STRINGS = (
        # String, Answer
        ("match00001_cam02.png",      1),
        ("match00001_cam01_mask.png", 1),
        ("match00010_cam01.png",      10),
        ("match00011_cam02_mask.png", 11),
    )

    for test_string, answer in TEST_STRINGS:
        assert answer == get_sa_car_id(test_string)
