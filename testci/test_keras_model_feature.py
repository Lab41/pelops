from PIL import Image
import collections
import datetime
import numpy as np
import pytest

from pelops.features.keras_model import KerasModelFeatureProducer


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


def test_preprocess_image(img_data):
    img = Image.fromarray(img_data)
    img_resized = KerasModelFeatureProducer.preprocess_image(img, 224, 224)
    assert img_resized.shape == (1, 224, 224, 3)
