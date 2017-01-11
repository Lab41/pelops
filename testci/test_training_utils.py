import pytest

from collections import namedtuple
from itertools import product, combinations_with_replacement
from pelops.datasets.chip import Chip
import pelops.training.utils as utils


@pytest.fixture(scope="module")
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
        (utils.make_model, [0, 1]),
        (utils.color, [2]),
        (utils.make_model_color, [0, 1, 2]),
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


@pytest.fixture(scope="module")
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


def test_make_model(chips_and_answers):
    for chip, answer in chips_and_answers:
        real_answer = (answer.make, answer.model)
        assert utils.make_model(chip) == real_answer


def test_color(chips_and_answers):
    for chip, answer in chips_and_answers:
        real_answer = (answer.color,)
        assert utils.color(chip) == real_answer


def test_make_model_color(chips_and_answers):
    for chip, answer in chips_and_answers:
        assert utils.make_model_color(chip) == answer
