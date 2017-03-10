from PIL import Image
import collections
import datetime
import numpy as np
import pytest

from pelops.features.resnet50 import ResNet50FeatureProducer


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
    CHIPS = (
        # filepath, car_id, cam_id, time, img_data, misc
        ("car1_cam1.png", 1, 1, datetime.datetime(2016, 10, 1, 0, 1, 2, microsecond=100), img_data, {}),
    )

    chip_producer = {"chips": {}}
    for filepath, car_id, cam_id, time, img_data, misc in CHIPS:
        chip = Chip(filepath, car_id, cam_id, time, img_data, misc)
        chip_producer["chips"][filepath] = chip

    return chip_producer


@pytest.fixture
def feature_producer(chip_producer):
    res = ResNet50FeatureProducer(chip_producer)
    return res


def test_features(feature_producer, chip_producer):
    for _, chip in chip_producer["chips"].items():
        features = feature_producer.produce_features(chip)
        assert features.shape == (1, 2048)
        assert np.sum(features) != 0


def test_preprocess_image(feature_producer, img_data):
    img = Image.fromarray(img_data)
    img_resized = feature_producer.preprocess_image(img, 224, 224)
    assert img_resized.shape == (1, 224, 224, 3)
