import pytest

import datetime as dt

# OpenCV is *VERY* hard to install in CircleCI, so if we don't have it, skip these tests
try:
    cv2 = pytest.importorskip("cv2")
    from pelops.datasets.chipper import FrameProducer
except ImportError:
    pass


@pytest.fixture
def frame_time_fp(tmpdir):
    # Define a FrameProducer with just enough information to run __get_frame_time()
    ifp = FrameProducer(
        file_list = [],
    )
    ifp.vid_metadata = {"fps": 30}

    return ifp


@pytest.fixture
def frame_time_fp_data(tmpdir):
    # Data to test __get_frame_time()
    DATA = (
        # (filename, frame number), (answer)
        (("/foo/bar/baz_20000101T000000-00000-006000.mp4", 0), dt.datetime(2000, 1, 1)),
        (("/foo/bar/baz_20000101T000000-00600-012000.mp4", 0), dt.datetime(2000, 1, 1, 0, 10)),
        (("/foo/bar/baz_20000101T000000-00000-006000.mp4", 1), dt.datetime(2000, 1, 1, 0, 0, 0, 33333)),
        (("/foo/bar/baz_20000101T000000-00600-012000.mp4", 10), dt.datetime(2000, 1, 1, 0, 10, 0, 333333)),
    )
    return DATA


def test_get_frame_time(frame_time_fp, frame_time_fp_data):
    for input, answer in frame_time_fp_data:
         output = frame_time_fp._FrameProducer__get_frame_time(input[0], input[1])
         assert output == answer
