import pytest

from pelops.features.keras_model import KerasModelFeatureProducer


def test_load_model_workaround():
    model_filename = './small.json'
    weight_filename = './small.hdf5'
    model = KerasModelFeatureProducer.load_model_workaround(
        model_filename, weight_filename)
    assert model.layers[0].name == 'dense_1'
