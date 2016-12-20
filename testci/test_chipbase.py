import pytest

from pelops.datasets.chipbase import Chip
from pelops.datasets.chipbase import ChipBase


@pytest.fixture
def chips():
    CHIPS = (
        # car_id, cam_id, time, filename, misc
        (1, 1, 100, "car1_cam1.png", None),
        (1, 2, 105, "car1_cam2.png", None),
        (1, 3, 110, "car1_cam3.png", None),
        (2, 1, 100, "car1_cam1.png", None),
        (2, 1, 102, "car1_cam2.png", None),
        (2, 1, 104, "car1_cam3.png", None),
    )

    chips = {}
    for car_id, cam_id, time, filename, misc in CHIPS:
        chip_id = '_'.join((str(car_id), str(cam_id), str(time)))
        chip = Chip(chip_id, car_id, cam_id, time, filename, misc)
        chips[chip_id] = chip

    return chips


@pytest.fixture
def chipbase(chips):
    """ Set up a instance of ChipBase(). """
    # Setup the class
    instantiated_class = ChipBase(dataset_name="Test")

    # Monkey Patch in a fake chips dictionary
    instantiated_class.chips = chips

    return instantiated_class


def test_chips_len(chipbase, chips):
    """ Test that ChipBase.chips is the correct length """
    assert len(chips) == len(chipbase)


def get_all_function_tester(in_chips, in_chipbase, index, test_function):
    """ Check that a chip getting function gets all the correct chips.

    This function tests a chip getting function, such as
    `get_all_chips_by_carid()` by creating a list of every correct chip from
    the true list of chips, and comparing it to the list returned by the
    function.

    Args:
        in_chips: The output of chips()
        in_chipbase: The output of chipbase()
        index: The location of the id in the chips object to use to compare. 0
            is the chip_id, 1 is the car_id, 2 is the cam_id.
        test_function: The function to test, it should return a list of chips
            selected by some id value.

    Returns:
        None
    """
    seen_ids = []
    for tup in in_chips.values():
        test_id = tup[index]
        # Generate all the chips by hand, and compare
        if test_id in seen_ids:
            continue
        seen_ids.append(test_id)
        chips_list = []
        for _, val in in_chipbase.chips.items():
            if val[index] == test_id:
                chips_list.append(val)

        chips_list.sort()
        test_chips = sorted(test_function(test_id))
        assert chips_list == test_chips


def test_get_all_chips_by_car_id(chipbase, chips):
    """ Test ChipBase.get_all_chips_by_carid() """
    CAR_ID_INDEX = 1
    get_all_function_tester(chips, chipbase, CAR_ID_INDEX,
                            chipbase.get_all_chips_by_carid)


def test_get_all_chips_by_cam_id(chipbase, chips):
    """ Test ChipBase.get_all_chips_by_camid() """
    CAM_ID_INDEX = 2
    get_all_function_tester(chips, chipbase, CAM_ID_INDEX,
                            chipbase.get_all_chips_by_camid)


def test_get_chip_image_path(chipbase, chips):
    """ Test ChipBase.get_chip_image_path() """
    for chip_id, chip in chips.items():
        assert chip.filename == chipbase.get_chip_image_path(chip_id)


def test_chipbase_iter(chipbase, chips):
    """ Test iteration over ChipBase() """
    for chip in chipbase:
        assert chip in chips.values()
