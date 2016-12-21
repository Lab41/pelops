import datetime

import pelops.utils as utils


def test_SetType():
    vals = utils.SetType.__members__
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

def test_get_index_of_pairs():
    TEST_LIST = [
        (0, 0, 'Mozart'),
        (1, 'Twinkle', 'Twinkle'),
        (2, 'Where', 'Little Caesar'),
        (3, 'When', 'Noon'),
        (4, 'How', 'Eat'),
        (5, None, None),
    ]

    # Test that we can find ints, strings, and Nones
    assert 1 == utils.get_index_of_pairs(TEST_LIST, 0, 1, 0)
    assert 2 == utils.get_index_of_pairs(TEST_LIST, 1, 2, 'Twinkle')
    assert 6 == utils.get_index_of_pairs(TEST_LIST, 1, 2, None)

    # Test that we report the last position if we don't find an answer
    assert len(TEST_LIST) == utils.get_index_of_pairs(
        TEST_LIST, 0, 1, 'NOT THERE')

def test_get_basename():
    TEST_FILEPATHS = (
        ("/path/to/file/hello.py", "hello.py"),
        ("hello.py", "hello.py")
    ) 

    for test_input, answer in TEST_FILEPATHS:
        assert answer == utils.get_basename(test_input)


def test_get_numeric():
    TEST_STRINGS = (
        ('c002.jpg', '002'),
        ('_012_', '012'),
    )

    for test_input, answer in TEST_STRINGS:
        assert answer == utils.get_numeric(test_input)


def test_get_timestamp():
    assert "2012-09-16 12:03:04" == str(utils.get_timestamp(datetime.datetime(2012, 9, 16, 12, 3, 4)))
    assert 1 == utils.get_timestamp(1)
    assert "Saturday" == utils.get_timestamp("Saturday")


def test_should_drop():
    # Never drop
    assert utils.should_drop(1.) is True
    # Always drop
    assert utils.should_drop(0.) is False
