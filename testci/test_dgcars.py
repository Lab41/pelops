import pytest

import os.path
import json

import pelops.utils as utils
from pelops.datasets.dgcars import DGCarsDataset
from pelops.datasets.chip import Chip
from pelops.utils import SetType


@pytest.fixture
def dgcars(tmpdir):
    # Define some test and training data, all will be the sum
    TRAIN = [
        {"url": "http://example.com/img.jpg", "hash": "2a8cedfa145b4345aed3fd9e82796c3e", "resnet50": "minivan", "model": "ZX2", "filename": "black/Ford/2a8cedfa145b4345aed3fd9e82796c3e.jpg", "make": "Ford", "color": "black"},
        {"url": "http://example.com/img.jpg", "hash": "8241daf452ace679162c69386f26ddc7", "resnet50": "sports_car", "model": "Mazda6 Sport", "filename": "red/Mazda/8241daf452ace679162c69386f26ddc7.jpg", "make": "Mazda", "color": "red"},
        {"url": "http://example.com/img.jpg", "hash": "e8dc3fb78206b14fe3568c1b28e5e5a1", "resnet50": "cab", "model": "XJ Series", "filename": "yellow/Jaguar/e8dc3fb78206b14fe3568c1b28e5e5a1.jpg", "make": "Jaguar", "color": "yellow"},
    ]
    TEST = [
        {"url": "http://example.com/img.jpg", "hash": "8881e7b561393f1d778a70dd449433e9", "resnet50": "racer", "model": "IS F", "filename": "yellow/Lexus/8881e7b561393f1d778a70dd449433e9.jpg", "make": "Lexus", "color": "yellow"},
        {"url": "http://example.com/img.jpg", "hash": "38e857d5235afda4315676c0b7756832", "resnet50": "pickup", "model": "Mark VII", "filename": "silver/Lincoln/38e857d5235afda4315676c0b7756832.jpg", "make": "Lincoln", "color": "silver"},
        {"url": "http://example.com/img.jpg", "hash": "6eb2b407cc398e70604bfd336bb2efad", "resnet50": "pickup", "model": "Lightning", "filename": "orange/Ford/6eb2b407cc398e70604bfd336bb2efad.jpg", "make": "Ford", "color": "orange"},
        {"url": "http://example.com/img.jpg", "hash": "eb3811772ec012545c8952d88906d355", "resnet50": "racer", "model": "Rockette", "filename": "green/Fairthorpe/eb3811772ec012545c8952d88906d355.jpg", "make": "Fairthorpe", "color": "green"},
        {"url": "http://example.com/img.jpg", "hash": "8dbbc1d930c7f2e4558efcc596728945", "resnet50": "minivan", "model": "S70", "filename": "white/Volvo/8dbbc1d930c7f2e4558efcc596728945.jpg", "make": "Volvo", "color": "white"},
        {"url": "http://example.com/img.jpg", "hash": "ed45784812d1281bcb61f217f4422ab5", "resnet50": "convertible", "model": "A8", "filename": "green/Audi/ed45784812d1281bcb61f217f4422ab5.jpg", "make": "Audi", "color": "green"},
        {"url": "http://example.com/img.jpg", "hash": "763ca4abbbb9b042b21f19fd80986179", "resnet50": "pickup", "model": "W126", "filename": "green/Mercedes-Benz/763ca4abbbb9b042b21f19fd80986179.jpg", "make": "Mercedes-Benz", "color": "green"},
    ]

    WRITE_LIST = (
        # filename, data list, settype
        ("allFiles", TRAIN + TEST, SetType.ALL),
        ("training", TRAIN, SetType.TRAIN),
        ("testing", TEST, SetType.TEST),
    )

    output_chips = {
        SetType.ALL: [],
        SetType.TRAIN: [],
        SetType.TEST: [],
    }
    for filename, data_list, settype in WRITE_LIST:
        fn = tmpdir.join(filename)
        with open(fn.strpath, "w") as f:
            for d in data_list:
                # Write the data list files
                line = json.dumps(d)
                f.write(line + "\n")

                # Make a chip
                fp = os.path.join(tmpdir.strpath, d["filename"])
                chip = Chip(fp, None, None, None, d)
                output_chips[settype].append(chip)

    # Instantiate a DGCarsDataset() class
    output_classes = {
        SetType.ALL: DGCarsDataset(tmpdir.strpath, SetType.ALL.value),
        SetType.TRAIN: DGCarsDataset(tmpdir.strpath, SetType.TRAIN.value),
        SetType.TEST: DGCarsDataset(tmpdir.strpath, SetType.TEST.value),
    }

    return (output_classes, output_chips)


def test_dgcars_chips_len(dgcars):
    classes = dgcars[0]
    answer_chips = dgcars[1]
    # check that self.chips has been created, is not empty, and has the right
    # number of etries
    for key, cls in classes.items():
        ans = answer_chips[key]
        assert len(cls.chips) == len(ans)

def test_dgcars_chips_vals(dgcars):
    classes = dgcars[0]
    answer_chips = dgcars[1]

    for key, cls in classes.items():
        ans = answer_chips[key]
        for chip in cls:
            # The chip must match one of our hand built chips
            assert chip in ans
            # Various values are None
            assert chip.car_id is None
            assert chip.cam_id is None
            assert chip.time is None
            # Misc and filepath should exist
            assert chip.filepath
            assert chip.misc
            # Misc is a dictionary like object
            assert hasattr(chip.misc, "get")


def test_get_all_chips_by_car_id(dgcars):
    classes = dgcars[0]
    answer_chips = dgcars[1]

    for key, cls in classes.items():
        ans = answer_chips[key]

        # All car_id values are None in DG Cars
        all_chips = sorted(cls.get_all_chips_by_car_id(None))
        assert all_chips == sorted(ans)


def test_get_all_chips_by_cam_id(dgcars):
    classes = dgcars[0]
    answer_chips = dgcars[1]

    for key, cls in classes.items():
        ans = answer_chips[key]

        # All cam_id values are None in DG Cars
        all_chips = sorted(cls.get_all_chips_by_cam_id(None))
        assert all_chips == sorted(ans)


def test_get_distinct_cams_by_car_id(dgcars):
    classes = dgcars[0]
    answer_chips = dgcars[1]

    for key, cls in classes.items():
        ans = answer_chips[key]

        # All car_id values are None in DG Cars
        assert cls.get_distinct_cams_by_car_id(None) == {None}


def test_get_all_cam_ids(dgcars):
    classes = dgcars[0]
    answer_chips = dgcars[1]

    for key, cls in classes.items():
        ans = answer_chips[key]

        # All cam_id values are None in DG Cars
        assert cls.get_all_cam_ids() == [None]


def test_get_all_car_ids(dgcars):
    classes = dgcars[0]
    answer_chips = dgcars[1]

    for key, cls in classes.items():
        ans = answer_chips[key]

        # All car_id values are None in DG Cars
        assert cls.get_all_car_ids() == [None]


def test_dgcars_iter(dgcars):
    classes = dgcars[0]
    answer_chips = dgcars[1]

    for key, cls in classes.items():
        ans = answer_chips[key]

        # Ensure that we can iterate and get all of the items
        for chip in cls:
            assert chip in ans

        # Ensure list can access the iterator, and that there are no extra
        # chips
        cls_chips = list(cls)
        for chip in ans:
            assert chip in cls_chips
