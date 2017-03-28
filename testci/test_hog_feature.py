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
def pil_image(img_data):
    img = Image.fromarray(img_data)
    img = img.convert("RGB")
    img = img.resize((256, 256), Image.BICUBIC)
    return img


@pytest.fixture
def hog_features(pil_image):
    img = color.rgb2gray(np.array(pil_image))
    features = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(16, 16))

    return features


@pytest.fixture
def color_features(pil_image):
    hists = []
    for channel in pil_image.split():
        channel_array = np.array(channel)
        values, _ = np.histogram(channel_array.flat, bins=256)
        hists.append(values)

    return np.concatenate(hists)


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
    hog = HOGFeatureProducer(chip_producer)

    return hog


def test_features(feature_producer, chip_producer, hog_features, color_features):
    fp = feature_producer
    hog_len = len(hog_features)
    hist_len = len(color_features)

    for _, chip in chip_producer["chips"].items():
        features = feature_producer.produce_features(chip)
        assert len(features) == hog_len + hist_len

        total_features = np.concatenate((hog_features, color_features))
        assert np.array_equal(features, total_features)


def test_inputs(chip_producer):
    pix_sizes = (32, 64, 128, 256, 512)
    cell_counts = (1, 2, 4, 16)
    orientation_counts = (2, 4, 8, 16)
    histogram_bins = (32, 64, 128, 256)
    for pix, cell, orientation, histogram_bin in product(pix_sizes, cell_counts, orientation_counts, histogram_bins):
        hog = HOGFeatureProducer(
            chip_producer,
            image_size=(pix, pix),
            cells=(cell, cell),
            orientations=orientation,
            histogram_bins_per_channel=histogram_bin,
        )
        for _, chip in chip_producer["chips"].items():
            features = hog.produce_features(chip)
            assert len(features) == ((cell**2) * orientation) + (3 * histogram_bin)
