from PIL import Image
from skimage import color
from skimage.feature import hog
import collections
import datetime
import numpy as np
import pytest
from itertools import product

from pelops.features.hog import HOGFeatureProducer


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
def hog_features(img_data):
    img = Image.fromarray(img_data)
    img = img.convert("RGB")
    img = img.resize((256, 256), Image.BICUBIC)
    img = color.rgb2gray(np.array(img))
    features = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(16, 16))

    return features


@pytest.fixture
def chip_producer(img_data):
    Chip = collections.namedtuple("Chip", ["filepath", "car_id", "cam_id", "time", "img_data", "misc"])
    CHIPS = (
        # filepath, car_id, cam_id, time, img_data, misc
        ("car1_cam1.png", 1, 1, datetime.datetime(2016, 10, 1, 0, 1, 2, microsecond=100), img_data, {}),
        (b"car1_cam1.png", 1, 1, datetime.datetime(2016, 10, 1, 0, 1, 2, microsecond=100), img_data, {}),
    )

    chip_producer = {"chips": {}}
    for filepath, car_id, cam_id, time, img_data, misc in CHIPS:
        chip = Chip(filepath, car_id, cam_id, time, img_data, misc)
        chip_producer["chips"][filepath] = chip

    return chip_producer


@pytest.fixture
def feature_producer(chip_producer):
    hog = HOGFeatureProducer(chip_producer)

    return hog


def test_features(feature_producer, chip_producer, hog_features):
    fp = feature_producer
    for _, chip in chip_producer["chips"].items():
        features = feature_producer.produce_features(chip)
        assert len(features) == fp.cells[0] * fp.cells[1] * fp.orientations
        assert np.array_equal(features, hog_features)


def test_inputs(chip_producer):
    pix_sizes = (32, 64, 126, 256, 512)
    cell_counts = (1, 2, 4, 16)
    orientation_counts = (2, 4, 8, 16)
    for pix, cell, orientation in product(pix_sizes, cell_counts, orientation_counts):
        hog = HOGFeatureProducer(
            chip_producer,
            image_size=(pix, pix),
            cells=(cell, cell),
            orientations=orientation,
        )
        for _, chip in chip_producer["chips"].items():
            features = hog.produce_features(chip)
            assert len(features) == (cell**2) * orientation
