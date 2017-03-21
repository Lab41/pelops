import os

import pytest

from pelops.features.keras_model import KerasModelFeatureProducer


def test_load_model_workaround():
    # @TODO get some environment variable set when in CI environment
    # test to see, modify path...
    if os.getenv('CIRCLECI', 'false').lower() == 'true':
        model_filename = '/home/ubuntu/pelops/testci/small.json'
        weight_filename = '/home/ubuntu/pelops/testci/small.hdf5'
    if int(os.getenv('INDOCKERCONTAINER', 0)) == 1:
        model_filename = '/pelops_root/testci/small.json'
        weight_filename = '/pelops_root/testci/small.hdf5'

    model = KerasModelFeatureProducer.load_model_workaround(
        model_filename, weight_filename)
    assert model.layers[0].name == 'dense_8'
