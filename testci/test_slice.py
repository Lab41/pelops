import io
import csv
import pytest
import pelops.datasets.slice as slice


@pytest.fixture
def slice_env(tmpdir):
    """Setup mock STR SLiCE dataset"""
    work_dir = tmpdir.mkdir('pelops_testing')
    truth = [
        ['% obSetIdx', ' chipIdx', ' targetID'],
        ['1', ' 1', '0'],
        ['1', ' 2', '1'],
        ['1', ' 3', '0'],
        ['2', ' 1', '1']
    ]

    truth_file = work_dir.join('truth.txt')
    with io.StringIO(newline='') as truth_hdl:
        csv.writer(truth_hdl).writerows(truth)
        truth_hdl.seek(0)
        truth_file.write(truth_hdl.read())

    for obset, chipid in {(row[0], row[1].strip()) for row in truth[1:]}:
        obset_dir = work_dir.join('ObSet00{}_1492560663_TestDir'.format(obset))
        obset_dir.ensure(dir=True)
        img_dir = obset_dir.join('images')
        img_dir.ensure(dir=True)
        img_file = img_dir.join('ObSet001-00{}.png'.format(chipid))
        img_file.ensure(dir=False)

    yield work_dir.strpath


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
    dtgs = {chip.time[:7] for chip in slice_dataset.chips.values()}
    assert len(dtgs) == 1
    assert '2017-04' in dtgs
