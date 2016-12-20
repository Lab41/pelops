import pytest
import os

from pelops.datasets.str import get_sa_cam_id
from pelops.datasets.str import get_sa_car_id
from pelops.datasets.str import int_from_string
from pelops.datasets.str import StrDataset


@pytest.fixture
def str_sa(tmpdir):
    """ Set up some test files and an instance of StrDataset(). """
    # Write a file to read back
    FILE_NAMES = (
        # filepath, car_id, cam_id, time, misc
        ("match00001_cam02.png", 1, 2, 2, None),
        ("match00001_cam01_mask.png", None, None, None, None),
        ("match00010_cam01.png", 10, 1, 1, None),
        ("match00011_cam02_mask.png", None, None, None, None)
    )
    # The contents of the files do not matter, the name is enough
    internal_dir = tmpdir.mkdir("crossCameraMatches")
    for name, _, _, _, _ in FILE_NAMES:
        out_file = internal_dir.join(name)
        out_file.write("TEST")

    # Setup the class
    instantiated_class = StrDataset(os.path.dirname(out_file.dirname))

    # Rename filepath
    FILE_NAMES = (
        (out_file.dirname + "/" + "match00001_cam02.png", 1, 2, 2, None),
        (out_file.dirname + "/" + "match00001_cam01_mask.png", None, None, None, None),
        (out_file.dirname + "/" + "match00010_cam01.png", 10, 1, 1, None),
        (out_file.dirname + "/" + "match00011_cam02_mask.png", None, None, None, None)
    )

    # Filter out the files that were not read
    RET_FILE_NAMES = tuple(t for t in FILE_NAMES if t[1] is not None)
    return (instantiated_class, RET_FILE_NAMES)


def test_str_sa_chips_len(str_sa):
    """ Test that StrDataset.chips is the correct length """
    instantiated_class = str_sa[0]
    FILE_NAMES = str_sa[1]
    # check that self.chips has been created, is not empty, and has the right
    # number of entries
    assert len(FILE_NAMES)
    assert len(FILE_NAMES) == len(instantiated_class.chips)


def test_str_sa_chips_vals(str_sa):
    """ Test that StrDataset chips have the correct values. """
    instantiated_class = str_sa[0]
    FILE_NAMES = str_sa[1]

    # Check that the correct chips exist
    for filepath, car_id, cam_id, time, misc in FILE_NAMES:
        chip = instantiated_class.chips[filepath]
        assert car_id == chip.car_id
        assert cam_id == chip.cam_id
        # The time is just the camera id for the STR SA data
        assert cam_id == chip.time
        # No misc data
        assert chip.misc is None
        # Filepath should be filled
        assert chip.filepath


def test_get_all_chips_by_car_id(str_sa):
    """ Test StrDataset.get_all_chips_by_car_id() """
    instantiated_class = str_sa[0]
    FILE_NAMES = str_sa[1]

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


def test_get_all_chips_by_cam_id(str_sa):
    """ Test StrDataset.get_all_chips_by_cam_id() """
    instantiated_class = str_sa[0]
    FILE_NAMES = str_sa[1]

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


def test_str_sa_iter(str_sa):
    """ Test StrDataset.get_chip_image_path() """
    instantiated_class = str_sa[0]
    FILE_NAMES = str_sa[1]
    chip_ids = tuple(i for i, _, _, _, _ in FILE_NAMES)

    for chip in instantiated_class:
        assert chip.filepath in chip_ids


def test_int_from_string():
    """ Test int_from_string() """
    TEST_STRINGS = (
        # String, Args, Answer
        ("test_010_test",                     ("test_", 3), 10),
        ("test_010_test",                     ("FAIL_", 3), None),
        ("test_010",                          ("test_", 3), 10),
        ("test_11_test",                      ("test_", 2), 11),
        ("010_test",                          ("",      3), 10),
        ("/foo/bar/bass/test_/test_010_test", ("test_", 3), 10),
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
