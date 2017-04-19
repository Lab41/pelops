import os
import csv
import shutil
import pytest
import tempfile
import pelops.datasets.slice as slice


@pytest.fixture
def slice_env():
    """Setup mock STR SLiCE dataset"""
    work_dir = tempfile.mkdtemp('_pelops_testing')
    truth = [
        ['% obSetIdx', ' chipIdx', ' targetID'],
        ['1', ' 1', '0'],
        ['1', ' 2', '1'],
        ['1', ' 3', '0'],
        ['2', ' 1', '1']
    ]
    truth_file = os.path.join(work_dir, 'truth.txt')
    with open(truth_file, 'w', newline='') as truth_hdl:
        csv.writer(truth_hdl).writerows(truth)

    for obset, chipid in {(row[0], row[1].strip()) for row in truth[1:]}:
        chip_dir = os.path.join(work_dir, 'ObSet00%s_1492560663_TestDir' % obset)
        if not os.path.isdir(chip_dir):
            os.makedirs(chip_dir)
        img_dir = os.path.join(chip_dir, 'images')
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)
        img_file = os.path.join(img_dir, 'ObSet001-00%s.png' % chipid)
        with open(img_file, 'w') as img_hdl:
            pass

    yield work_dir

    try:
        shutil.rmtree(work_dir)
    except IOError:
        pass


def test_slice_chip_load(slice_env):
    """Test that SLiCE chips load without error"""
    slice_dataset = slice.SliceDataset(slice_env)
    assert len(slice_dataset.chips) == 4


def test_slice_chip_tgt_car_id(slice_env):
    """Test that SLiCE chips for target vehicles are processed properly."""
    slice_dataset = slice.SliceDataset(slice_env)
    target_ids = [chip.car_id for chip in slice_dataset.chips.values() if chip.car_id.startswith('tgt-')]
    assert 'tgt-000000001' in target_ids
    assert len(target_ids) == 2
    assert len(set(target_ids)) == 1


def test_slice_chip_unk_car_id(slice_env):
    """Test that SLiCE chips for non-target vehicles are processed properly."""
    slice_dataset = slice.SliceDataset(slice_env)
    unk_ids = [chip.car_id for chip in slice_dataset.chips.values() if chip.car_id.startswith('unk-')]
    assert 'unk-000000001' in unk_ids
    assert len(unk_ids) == 2


def test_slice_chip_dtg(slice_env):
    """Test that date/times encoded in filenames are processed properly."""
    slice_dataset = slice.SliceDataset(slice_env)
    dtgs = {chip.time[:10] for chip in slice_dataset.chips.values()}
    assert len(dtgs) == 1
    assert '2017-04-18' in dtgs
