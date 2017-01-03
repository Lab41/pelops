import pytest

from pelops.datasets.chip import ChipDataset, Chip


@pytest.fixture
def chips():
    CHIPS = (
        # filepath, car_id, cam_id, time, misc
        ("car1_cam1.png", 1, 1, 100, None),
        ("car1_cam2.png", 1, 2, 105, None),
        ("car1_cam3.png", 1, 3, 110, None),
        ("car2_cam1.png", 2, 1, 100, None),
        ("car2_cam2.png", 2, 1, 102, None),
        ("car2_cam3.png", 2, 1, 104, None),
    )

    chips = {}
    for filepath, car_id, cam_id, time, misc in CHIPS:
        chip = Chip(filepath, car_id, cam_id, time, misc)
        chips[filepath] = chip

    return chips


@pytest.fixture
def chip_dataset(chips):
    """ Set up a instance of ChipDataset(). """
    # Setup the class
    instantiated_class = ChipDataset(dataset_path="Test")

    # Monkey Patch in a fake chips dictionary
    instantiated_class.chips = chips

    return instantiated_class


def test_chips_len(chip_dataset, chips):
    """ Test that ChipDataset.chips is the correct length """
    assert len(chips) == len(chip_dataset)


def get_all_function_tester(in_chips, in_chipbase, index, test_function):
    """ Check that a chip getting function gets all the correct chips.

    This function tests a chip getting function, such as
    `get_all_chips_by_carid()` by creating a list of every correct chip from
    the true list of chips, and comparing it to the list returned by the
    function.

    Args:
        in_chips: The output of chips()
        in_chipbase: The output of chipbase()
        index: The location of the id in the chips object to use to compare. 
            0 is the filepath (aka chip_id), 1 is the car_id, 2 is the cam_id.
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


def test_get_all_chips_by_car_id(chip_dataset, chips):
    """ Test ChipDataset.get_all_chips_by_carid() """
    CAR_ID_INDEX = 1
    get_all_function_tester(chips, chip_dataset, CAR_ID_INDEX,
                            chip_dataset.get_all_chips_by_car_id)


def test_get_all_chips_by_cam_id(chip_dataset, chips):
    """ Test ChipDataset.get_all_chips_by_camid() """
    CAM_ID_INDEX = 2
    get_all_function_tester(chips, chip_dataset, CAM_ID_INDEX,
                            chip_dataset.get_all_chips_by_cam_id)


def test_get_distinct_cams_by_car_id(chip_dataset):
    """ Test ChipDataset.get_distinct_cams_by_car_id() and get_distinct_cams_per_car() """
    CAR_ID = 1
    TEST_CAMS = [1, 2, 3]
    for test_cam, cam in zip(TEST_CAMS, sorted(chip_dataset.get_distinct_cams_by_car_id(CAR_ID))):
        assert test_cam == cam


def test_get_all_cam_ids(chip_dataset):
    """ Test ChipDataset.get_all_cam_ids() """
    TEST_CAMS = [1, 2, 3]
    for test_cam, cam in zip(TEST_CAMS, sorted(chip_dataset.get_all_cam_ids())):
        assert test_cam == cam


def test_get_all_car_ids(chip_dataset):
    TEST_CARS = [1, 2]
    for test_car, car in zip (TEST_CARS, sorted(chip_dataset.get_all_car_ids())):
        assert test_car == car


def test_chipdataset_iter(chip_dataset, chips):
    """ Test iteration over ChipDataset() """
    for chip in chip_dataset:
        assert chip in chips.values()
