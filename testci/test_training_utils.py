import pytest

from collections import namedtuple
from itertools import product
from pelops.datasets.chip import Chip
import pelops.training.utils as utils


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
