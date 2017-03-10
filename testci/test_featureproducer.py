import collections
import datetime
import pytest
import numpy as np
from PIL import Image

from pelops.features.feature_producer import FeatureProducer


@pytest.fixture
def img_data():
    DATA = [[[  0,   0,   0],
             [255, 255, 255],
             [  0,   0,   0]],
            [[255, 255, 255],
             [  0,   0,   0],
             [255, 255, 255]],
            [[  0,   0,   0],
             [255, 255, 255],
             [  0,   0,   0]]]
    return np.array(DATA, dtype=np.uint8)


@pytest.fixture
def chip_producer(img_data):
    Chip = collections.namedtuple("Chip", ["filepath", "car_id", "cam_id", "time", "img_data", "misc"])
    ChipProducer = collections.namedtuple("ChipProducer", ["chips"])
    CHIPS = (
        # filepath, car_id, cam_id, time, img_data, misc
        ("car1_cam1.png", 1, 1, datetime.datetime(2016, 10, 1, 0, 1, 2, microsecond=100), img_data, {}),
    )

    chip_producer = ChipProducer({})
    for filepath, car_id, cam_id, time, img_data, misc in CHIPS:
        print(img_data.shape)
        chip = Chip(filepath, car_id, cam_id, time, img_data, misc)
        chip_producer.chips[filepath] = chip

    return chip_producer


@pytest.fixture
def monkey_feature_producer(chip_producer):
    # Monkey patch the __init__() function so that it will succeed
    def new_init(self, chip_producer):
        self.chip_producer = chip_producer
        self.feat_size = 1

    FeatureProducer.__init__ = new_init

    return FeatureProducer(chip_producer)


def test_set_variables_raises():
    with pytest.raises(NotImplementedError):
        fp = FeatureProducer(None)


def test_produce_features_raises(monkey_feature_producer):
    with pytest.raises(NotImplementedError):
        monkey_feature_producer.produce_features(None)


def test_get_image_img_data(monkey_feature_producer, chip_producer, img_data):
    for key, chip in chip_producer.chips.items():
        image = monkey_feature_producer.get_image(chip)
        image_array = np.array(image)
        assert np.array_equal(img_data, np.array(image))


def test_return_features_raises(monkey_feature_producer):
    with pytest.raises(NotImplementedError):
        monkey_feature_producer.return_features()
