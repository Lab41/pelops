import pytest

from collections import namedtuple
from itertools import product, combinations_with_replacement
from pelops.datasets.chip import Chip
from pelops.utils import SetType
import json
import os.path
import pelops.training.utils as utils


@pytest.fixture(scope="session")
def make_model_color_classes():
    MAKES = ("Honda", "Toyota", None,)
    MODELS = ("Civic", "Corolla", None,)
    COLORS = ("Red", "Blue", None,)

    # Make all possible sets of three "chips", including with replacement, so
    # for example we'll get:
    #
    # (Honda, Honda, Honda)
    # (Honda, Honda, Toyota)
    # (Honda, Toyota, None)
    #
    # And others
    NUMBER_OF_CHIPS = 3
    MAKE_P = combinations_with_replacement(MAKES, NUMBER_OF_CHIPS)
    MODEL_P = combinations_with_replacement(MODELS, NUMBER_OF_CHIPS)
    COLOR_P = combinations_with_replacement(COLORS, NUMBER_OF_CHIPS)

    answer = namedtuple("answer", ["make", "model", "color"])

    chips_and_answers = []
    for makes, models, colors in product(MAKE_P, MODEL_P, COLOR_P):
        chips = []
        ans = []
        for make, model, color in zip(makes, models, colors):
            chips.append(Chip(None, None, None, None, {"make": make, "model": model, "color": color, "other_key": "DO NOT SELECT THIS"}))
            ans.append(answer(make, model, color))
        chips_and_answers.append((tuple(chips), tuple(ans)))

    return chips_and_answers


def test_attributes_to_classes(make_model_color_classes):
    chips_and_answers = make_model_color_classes

    # The keyr function, and the indices from the answer namedtuple
    # corresponding to what is returned by the function
    function_and_indices = (
        (utils.key_make_model, [0, 1]),
        (utils.key_color, [2]),
        (utils.key_make_model_color, [0, 1, 2]),
    )

    for chips, answers in chips_and_answers:
        for func, indices in function_and_indices:
            class_to_index = utils.attributes_to_classes(chips, func)

            # Make the answer set by selecting the correct entries out of the
            # answer namedtuple
            answer = []
            for a in answers:
                temp_tup = []
                for i in indices:
                    temp_tup.append(str(a[i]))

                ans_str = "_".join(temp_tup)
                answer.append(ans_str)

            answer = set(answer)

            # Test that the keys are correct
            keys = set(class_to_index.keys())
            assert keys == answer

            # Test that the indices are correct
            indices = set(class_to_index.values())
            assert indices == {i for i in range(len(answer))}


@pytest.fixture(scope="session")
def chips_and_answers():
    MAKES = ("Honda", None,)
    MODELS = ("Civic", None,)
    COLORS = ("Red", None,)

    answer = namedtuple("answer", ["make", "model", "color"])

    chips_and_answers = []
    for make, model, color in product(MAKES, MODELS, COLORS):
        chip = Chip(None, None, None, None, {"make": make, "model": model, "color": color, "other_key": "DO NOT SELECT THIS"})
        ans = answer(make, model, color)
        chips_and_answers.append((chip, ans))

    # If misc is missing, it should also work
    chip = Chip(None, None, None, None, None)
    ans = answer(None, None, None)
    chips_and_answers.append((chip, ans))

    # If keys are missing, it should also work
    chip = Chip(None, None, None, None, {"other_key": "DO NOT SELECT THIS"})
    ans = answer(None, None, None)
    chips_and_answers.append((chip, ans))

    # If misc is missing, it should still work
    fake_chip = namedtuple("fakechip", ["not_misc"])
    chip = fake_chip(None)
    ans = answer(None, None, None)
    chips_and_answers.append((chip, ans))

    return chips_and_answers


def test_key_make_model(chips_and_answers):
    for chip, answer in chips_and_answers:
        ans = (str(answer.make), str(answer.model))
        real_answer = "_".join(ans)
        assert utils.key_make_model(chip) == real_answer


def test_key_color(chips_and_answers):
    for chip, answer in chips_and_answers:
        real_answer = str(answer.color)
        assert utils.key_color(chip) == real_answer


def test_key_make_model_color(chips_and_answers):
    for chip, answer in chips_and_answers:
        ans = (str(answer.make), str(answer.model), str(answer.color))
        real_answer = "_".join(ans)
        assert utils.key_make_model_color(chip) == real_answer


# A fake ChipDataset
class FakeChipDataset(object):
    def __init__(self, chips, set_type):
        self.chips = chips
        self.set_type = set_type

    def __iter__(self):
        for chip in self.chips.values():
            yield chip
        raise StopIteration()


@pytest.fixture()
def fake_dataset(tmpdir):
    MAKE_MODELS = (
        ("honda", "civic"),
        ("toyota", "corolla"),
    )

    # File names must start at 0 and increment by 1. They must be assigned to
    # the tuples after the tuples have been sorted alphabetically.
    chips = {}
    for i, (make, model) in enumerate(sorted(MAKE_MODELS)):
        name = str(i) + ".jpg"
        fn = tmpdir.join(name)
        fn.write("FAKE IMAGE")
        chip = Chip(str(fn), None, None, None, {"make": make, "model": model})
        chips[i] = chip

    return FakeChipDataset(chips, None)


def test_KerasDirectory_write_links(tmpdir, fake_dataset):
    # Link the files into the tmp directory
    out_dir = tmpdir.mkdir("output")
    kd = utils.KerasDirectory(fake_dataset, utils.key_make_model)
    kd.write_links(output_directory=out_dir.strpath)

    # Because we always read through the chips dictionary in the same order,
    # the output files are deterministic. We now check that they exist.
    sorted_keys = sorted(fake_dataset.chips.keys())
    for key in sorted_keys:
        chip = fake_dataset.chips[key]
        file_basename = os.path.basename(chip.filepath)
        test_file = os.path.join(out_dir.strpath, "all", str(key), file_basename)
        is_file = os.path.isfile(test_file)
        assert is_file

    # We also write out a key -> index map
    map_file = os.path.join(out_dir.strpath, "all", "class_to_index_map.json")
    is_file = os.path.isfile(map_file)
    assert is_file


def test_KerasDirectory_write_map(tmpdir, fake_dataset):
    # Write a map file
    out_dir = tmpdir.mkdir("json")
    kd = utils.KerasDirectory(fake_dataset, utils.key_make_model)
    kd.write_map(output_directory=out_dir.strpath)

    # Check that it is a file
    map_file = os.path.join(out_dir.strpath, "class_to_index_map.json")
    is_file = os.path.isfile(map_file)
    assert is_file

    # Check that the contents are correct
    CONT = {"toyota_corolla": 1, "honda_civic": 0}
    with open(map_file, 'r') as json_file:
        reloaded_string = json.load(json_file)

    for key, val in reloaded_string.items():
        assert CONT[key] == val


def test_KerasDirectory_set_root():
    TYPES = (
        # Normal cases
        (SetType.ALL, "all"),
        (SetType.QUERY, "query"),
        (SetType.TEST, "test"),
        (SetType.TRAIN, "train"),
        # Not SetTypes, should return "all"
        (None, "all"),
    )

    for set_type, answer in TYPES:
        fcd = FakeChipDataset({}, set_type)
        kd = utils.KerasDirectory(fcd, None)
        assert kd.root == answer

    # Lists of chips should be ok too, but they return "all"
    kd = utils.KerasDirectory([], None)
    assert kd.root == "all"
