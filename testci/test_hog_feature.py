from PIL import Image
from skimage import color
from skimage.feature import hog
import collections
import datetime
import numpy as np
import pytest
from itertools import product

from pelops.features.hog import HOGFeatureProducer

def hog_features(img):
    img = color.rgb2gray(np.array(img))
    features = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(16, 16))
    return features


def hist_features(img):
        MAX_CHANNELS = 3
        BINS = 256

        channels = img.split()

        # Remove alpha channels
        if len(channels) > MAX_CHANNELS:
            channels = channel[:MAX_CHANNELS]

        # Calculate features
        hist_features = np.zeros(MAX_CHANNELS * BINS)
        for i, channel in enumerate(channels):
            channel_array = np.array(channel)
            values, _ = np.histogram(channel_array.flat, bins=BINS)
            start = i * BINS
            end = (i+1) * BINS
            hist_features[start:end] = values

        return hist_features


@pytest.fixture(scope="module")
def img_data():
    data = {
        "DATA_1":{},
        "DATA_3":{},
        "DATA_4":{},
    }

    # Raw data
    data["DATA_1"]["array"] = np.array([
        [[  0,   0,   0],
         [255, 255, 255],
         [  0,   0,   0]],
    ], dtype=np.uint8)

    data["DATA_3"]["array"] = np.array([
        [[  0,   0,   0],
         [255, 255, 255],
         [  0,   0,   0]],
        [[255, 255, 255],
         [  0,   0,   0],
         [255, 255, 255]],
        [[  0,   0,   0],
         [255, 255, 255],
         [  0,   0,   0]],
    ], dtype=np.uint8)

    data["DATA_4"]["array"] = np.array([
        [[  0,   0,   0],
         [255, 255, 255],
         [  0,   0,   0]],
        [[255, 255, 255],
         [  0,   0,   0],
         [255, 255, 255]],
        [[  0,   0,   0],
         [255, 255, 255],
         [  0,   0,   0]],
        [[  0,   0,   0],
         [  0,   0,   0],
         [  0,   0,   0]],
    ], dtype=np.uint8)

    # PIL images
    for data_id in data:
        arr = data[data_id]["array"]
        img = Image.fromarray(arr)
        img = img.convert("RGB")
        img = img.resize((256, 256), Image.BICUBIC)
        data[data_id]["image"] = img

    # Calculate HOG features
    for data_id in data:
        img = data[data_id]["image"]
        hog = hog_features(img)
        data[data_id]["hog_features"] = hog

    # Calculate Histogram features
    for data_id in data:
        img = data[data_id]["image"]
        hist = hist_features(img)
        data[data_id]["hist_features"] = hist

    return data


@pytest.fixture
def chip_producer(img_data):
    Chip = collections.namedtuple("Chip", ["filepath", "car_id", "cam_id", "time", "img_data", "misc"])
    CHIPS = []
    for i, data_id in enumerate(img_data):
        data = img_data[data_id]
        arr = data["array"]
        # We use the data_id as the filepath since we do not actually open the
        # file and it only needs to be unique
        #
        # filepath, car_id, cam_id, time, img_data, misc
        chip = (data_id, i, 1, datetime.datetime(2016, 10, 1, 0, 1, 2, microsecond=100+i), arr, {})
        CHIPS.append(chip)

    chip_producer = {"chips": {}}
    for filepath, car_id, cam_id, time, data, misc in CHIPS:
        chip = Chip(filepath, car_id, cam_id, time, data, misc)
        chip_producer["chips"][filepath] = chip

    return chip_producer


@pytest.fixture
def feature_producer(chip_producer):
    hog = HOGFeatureProducer(chip_producer)

    return hog


def test_features(feature_producer, chip_producer, img_data):
    fp = feature_producer

    for _, chip in chip_producer["chips"].items():
        data_id = chip.filepath
        data = img_data[data_id]
        hog_features = data["hog_features"]
        hist_features = data["hist_features"]
        hog_len = len(hog_features)
        hist_len = len(hist_features)

        features = feature_producer.produce_features(chip)
        assert len(features) == hog_len + hist_len

        total_features = np.concatenate((hog_features, hist_features))
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
