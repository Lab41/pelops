import datetime as dt

import pelops.experiment_api.utils as utils


def test_ImageType():
    vals = utils.ImageType.__members__
    assert 'ALL' in vals
    assert 'QUERY' in vals
    assert 'TEST' in vals
    assert 'TRAIN' in vals


def test_get_index_of_tuple():
    TEST_LIST = [
        (0, 'Who', 'John'),
        (1, 'What', 'Pizza'),
        (2, 'Where', 'Little Caesar'),
        (3, 'When', 'Noon'),
        (4, 'How', 'Eat'),
        (5, None, None),
    ]

    # Test that we can find ints, strings, and Nones
    assert 1 == utils.get_index_of_tuple(TEST_LIST, 0, 0)
    assert 2 == utils.get_index_of_tuple(TEST_LIST, 1, 'What')
    assert 6 == utils.get_index_of_tuple(TEST_LIST, 1, None)

    # Test that we report the last position if we don't find an answer
    assert len(TEST_LIST) == utils.get_index_of_tuple(
        TEST_LIST, 0, 'NOT THERE')


def test_get_numeric():
    TEST_STRINGS = (
        ('c002.jpg', '002'),
        ('_012_', '012'),
    )

    for test_input, answer in TEST_STRINGS:
        assert answer == utils.get_numeric(test_input)


def test_should_drop():
    # Never drop
    assert utils.should_drop(1.) is True
    # Always drop
    assert utils.should_drop(0.) is False
