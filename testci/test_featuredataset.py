import datetime
import pytest

import numpy as np
from pelops.datasets.chip import ChipDataset, Chip
from pelops.datasets.featuredataset import FeatureDataset

FEAT_LENGTH = 2048

@pytest.fixture
def chips():
    CHIPS = (
        # filepath, car_id, cam_id, time, misc
        ("car1_cam1.png", 1, 1, datetime.datetime(2016, 10,1, 0, 1, 2, microsecond=100), {}),
        ("car1_cam2.png", 1, 2, datetime.datetime(2016, 10,1, 0, 1, 2, microsecond=105), {}),
        ("car1_cam3.png", 1, 3, datetime.datetime(2016, 10,1, 0, 1, 2, microsecond=110), {}),
        ("car2_cam1.png", 2, 1, datetime.datetime(2016, 10,1, 0, 1, 2, microsecond=100), {}),
        ("car2_cam2.png", 2, 1, datetime.datetime(2016, 10,1, 0, 1, 2, microsecond=102), {}),
        ("car2_cam3.png", 2, 1, datetime.datetime(2016, 10,1, 0, 1, 2, microsecond=104), {}),
    )

    chips = {}
    for filepath, car_id, cam_id, time, misc in CHIPS:
        chip = Chip(filepath, car_id, cam_id, time, misc)
        chips[filepath] = chip

    return chips

@pytest.fixture
def feature_dataset(chips):
    OUTPUT_FNAME = '/tmp/test_featre_dataset.hdf5'
    feat_data = np.random.random((len(chips), FEAT_LENGTH))
    FeatureDataset.save(OUTPUT_FNAME, list(chips.keys()), list(chips.values()), feat_data)
    return FeatureDataset(OUTPUT_FNAME)

def test_get_feats(chips, feature_dataset):
    chip_key = next(iter(chips))
    chip = chips[chip_key]
    assert len(feature_dataset.get_feats_for_chip(chip)) == FEAT_LENGTH

def test_load_save(chips, feature_dataset):
    chip_key = next(iter(chips))
    assert feature_dataset.chips[chip_key] == chips[chip_key]
