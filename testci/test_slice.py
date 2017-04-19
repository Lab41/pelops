import os
import csv
import shutil
import pytest
import tempfile
import pelops.datasets.slice as slice


@pytest.fixture
def slice_env():
    work_dir = tempfile.mkdtemp('_pelops_testing')
    truth = [
        ['% obSetIdx', ' chipIdx', ' targetID'],
        ['1', ' 1', '0'],
        ['1', ' 2', '1'],
        ['1', ' 3', '0']
    ]
    truth_file = os.path.join(work_dir, 'truth.txt')
    with open(truth_file, 'w', newline='') as truth_hdl:
        csv.writer(truth_hdl).writerows(truth)

    chip_dir = os.path.join(work_dir, 'ObSet001_1492560663_TestDir')
    os.makedirs(chip_dir)
    for chipId in range(1, 4):
        img_dir = os.path.join(chip_dir, 'images')
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)
        img_file = os.path.join(img_dir, 'ObSet001-%.3d.png' % chipId)
        with open(img_file, 'w') as img_hdl:
            pass

    yield work_dir
    shutil.rmtree(work_dir)


def test_slice_chip_load(slice_env):
    slice_dataset = slice.SliceDataset(slice_env)
    assert len(slice_dataset.chips) == 3

def test_slice_chip_car_id(slice_env):
    slice_dataset = slice.SliceDataset(slice_env)
    assert 'tgt-000000001' in [chip.car_id for chip in slice_dataset.chips.values()]
