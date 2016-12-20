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


def test_get_all_chips_by_car_id(chipbase, chips):
    """ Test ChipBase.get_all_chips_by_carid() """
    seen_ids = []
    for chip_id, car_id, cam_id, _, _, _ in chips.values():
        # Generate all the chips by hand, and compare
        if car_id in seen_ids:
            continue
        seen_ids.append(car_id)
        chips = []
        for key, val in chipbase.chips.items():
            if val.car_id == car_id:
                chips.append(val)

        chips.sort()
        test_chips = sorted(chipbase.get_all_chips_by_carid(car_id))
        assert chips == test_chips


def test_get_all_chips_by_cam_id(chipbase, chips):
    """ Test ChipBase.get_all_chips_by_camid() """
    seen_ids = []
    for chip_id, car_id, cam_id, _, _, _ in chips.values():
        # Generate all the chips by hand, and compare
        if cam_id in seen_ids:
            continue
        seen_ids.append(cam_id)
        chips = []
        for key, val in chipbase.chips.items():
            if val.cam_id == cam_id:
                chips.append(val)

        chips.sort()
        test_chips = sorted(chipbase.get_all_chips_by_camid(cam_id))
        assert chips == test_chips


def test_get_chip_image_path(chipbase, chips):
    """ Test ChipBase.get_chip_image_path() """
    for chip_id, chip in chips.items():
        assert chip.filename == chipbase.get_chip_image_path(chip_id)


def test_chipbase_iter(chipbase, chips):
    """ Test iteration over ChipBase() """
    for chip in chipbase:
        assert chip in chips.values()
