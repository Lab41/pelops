import pytest

from pelops.datasets.str_sa import get_sa_cam_id
from pelops.datasets.str_sa import get_sa_car_id
from pelops.datasets.str_sa import int_from_string
from pelops.datasets.str_sa import STR_SA


@pytest.fixture
def str_sa(tmpdir):
    """ Set up some test files and an instance of STR_SA(). """
    # Write a file to read back
    FILE_NAMES = (
        # Filename, car_id, cam_id, chip_id
        ("match00001_cam02.png", 1, 2, "1_2_2"),
        ("match00001_cam01_mask.png", None, None, None),
        ("match00010_cam01.png", 10, 1, "10_1_1"),
        ("match00011_cam02_mask.png", None, None, None)
    )
    # The contents of the files do not matter, the name is enough
    for name, _, _, _ in FILE_NAMES:
        out_file = tmpdir.join(name)
        out_file.write("TEST")

    # Setup the class
    instantiated_class = STR_SA(directory=out_file.dirname)

    # Filter out the files that were not read
    RET_FILE_NAMES = tuple(t for t in FILE_NAMES if t[3] is not None)
    return (instantiated_class, RET_FILE_NAMES)


def test_str_sa_chips_len(str_sa):
    """ Test that STR_SA.chips is the correct length """
    instantiated_class = str_sa[0]
    FILE_NAMES = str_sa[1]
    # check that self.chips has been created, is not empty, and has the right
    # number of entries
    assert len(FILE_NAMES) == len(instantiated_class.chips)


def test_str_sa_chips_vals(str_sa):
    """ Test that STR_SA chips have the correct values. """
    instantiated_class = str_sa[0]
    FILE_NAMES = str_sa[1]

    # Check that the correct chips exist
    for _, car_id, cam_id, chip_id in FILE_NAMES:
        chip = instantiated_class.chips[chip_id]
        print(chip)
        assert car_id == chip.car_id
        assert cam_id == chip.cam_id
        # The time is just the camera id for the STR SA data
        assert cam_id == chip.time
        # No misc data
        assert chip.misc is None
        # File name should be filled
        assert chip.filename


def test_get_all_chips_by_carid(str_sa):
    """ Test STR_SA.get_all_chips_by_carid() """
    instantiated_class = str_sa[0]
    FILE_NAMES = str_sa[1]

    seen_ids = []
    for _, car_id, cam_id, chip_id in FILE_NAMES:
        # Generate all the chips by hand, and compare
        if car_id in seen_ids:
            continue
        seen_ids.append(car_id)
        chips = []
        for key, val in instantiated_class.chips.items():
            if val.car_id == car_id:
                chips.append(val)

        chips.sort()
        test_chips = sorted(instantiated_class.get_all_chips_by_carid(car_id))
        assert chips == test_chips


def test_get_all_chips_by_cam_id(str_sa):
    """ Test STR_SA.get_all_chips_by_camid() """
    instantiated_class = str_sa[0]
    FILE_NAMES = str_sa[1]

    seen_ids = []
    for _, car_id, cam_id, chip_id in FILE_NAMES:
        # Generate all the chips by hand, and compare
        if cam_id in seen_ids:
            continue
        seen_ids.append(cam_id)
        chips = []
        for key, val in instantiated_class.chips.items():
            if val.cam_id == cam_id:
                chips.append(val)

        chips.sort()
        test_chips = sorted(instantiated_class.get_all_chips_by_camid(cam_id))
        assert chips == test_chips


def test_get_chip_image_path(str_sa):
    """ Test STR_SA.get_chip_image_path() """
    instantiated_class = str_sa[0]
    FILE_NAMES = str_sa[1]
    chip_ids = (i for _, _, _, i in FILE_NAMES)

    for chip_id in chip_ids:
        path = instantiated_class.get_chip_image_path(chip_id)
        assert path


def test_str_sa_iter(str_sa):
    """ Test STR_SA.get_chip_image_path() """
    instantiated_class = str_sa[0]
    FILE_NAMES = str_sa[1]
    chip_ids = tuple(i for _, _, _, i in FILE_NAMES)

    for chip in instantiated_class:
        assert chip.chip_id in chip_ids


def test_str_sa_construction_raise():
    """ Test that STR_SA() raises an error when not given a directory. """
    with pytest.raises(ValueError):
        STR_SA()


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
