import pytest

from collections import namedtuple
from itertools import product, combinations_with_replacement
from pelops.datasets.chip import Chip
from pelops.utils import SetType
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

    # The tuplizer function, and the indices from the answer namedtuple
    # corresponding to what is returned by the function
    function_and_indices = (
        (utils.tuplize_make_model, [0, 1]),
        (utils.tuplize_color, [2]),
        (utils.tuplize_make_model_color, [0, 1, 2]),
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
                    temp_tup.append(a[i])

                answer.append(tuple(temp_tup))

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


def test_tuplize_make_model(chips_and_answers):
    for chip, answer in chips_and_answers:
        real_answer = (answer.make, answer.model)
        assert utils.tuplize_make_model(chip) == real_answer


def test_tuplize_color(chips_and_answers):
    for chip, answer in chips_and_answers:
        real_answer = (answer.color,)
        assert utils.tuplize_color(chip) == real_answer


def test_tuplize_make_model_color(chips_and_answers):
    for chip, answer in chips_and_answers:
        assert utils.tuplize_make_model_color(chip) == answer


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
        ("1.jpg", "honda", "civic"),
        ("2.jpg", "toyota", "corolla"),
    )

    chips = {}
    for name, make, model in MAKE_MODELS:
        fn = tmpdir.join(name)
        fn.write("")
        chip = Chip(str(fn), None, None, None, {"make": make, "model": model})
        chips[name] = chip

    return FakeChipDataset(chips, None)


def test_KerasDirectory_write_links(tmpdir, fake_dataset):
    # Link the files into the tmp directory
    out_dir = tmpdir.mkdir("output")
    kd = utils.KerasDirectory(fake_dataset, utils.tuplize_make_model)
    kd.write_links(output_directory=str(out_dir))

    # Because we always read through the chips dictionary in the same order,
    # the output files are deterministic. We now check that they exist.
    for i, chip in enumerate(fake_dataset.chips.values()):
        f = str(out_dir) + "/all/" + str(i) + "/" + os.path.basename(chip.filepath)
        is_file = os.path.isfile(f)
        assert is_file


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
