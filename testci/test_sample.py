import pytest

def func(x):
    return x+1


""" test func """
def test_func():
    assert func(2) == 3