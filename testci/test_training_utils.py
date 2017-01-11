import pytest

from collections import namedtuple
from itertools import product, permutations
from pelops.datasets.chip import Chip
import pelops.training.utils as utils


@pytest.fixture(scope="module")
def make_model_color_classes():
    MAKES = ("Honda", "Toyota", None,)
    MODELS = ("Civic", "Corolla", None,)
    COLORS = ("Red", "Blue", None,)
    MAKE_P = permutations(MAKES, 2)
    MODEL_P = permutations(MODELS, 2)
    COLOR_P = permutations(COLORS, 2)

    answer = namedtuple("answer", ["make", "model", "color"])

    chips_and_answers = []
    for makes, models, colors in product(MAKE_P, MODEL_P, COLOR_P):
        chip0 = Chip(None, None, None, None, {"make": makes[0], "model": models[0], "color": colors[0], "other_key": "DO NOT SELECT THIS"})
        chip1 = Chip(None, None, None, None, {"make": makes[1], "model": models[1], "color": colors[1], "other_key": "DO NOT SELECT THIS"})
        ans0 = answer(makes[0], models[0], colors[0])
        ans1 = answer(makes[1], models[1], colors[1])
        ans = (ans0, ans1)
        chips = (chip0, chip1)
        chips_and_answers.append((chips, ans))

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
