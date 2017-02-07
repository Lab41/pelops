from collections import namedtuple, deque
from enum import Enum
import logging
import os.path
import datetime as dt

import numpy as np
import imageio
import cv2

Frame = namedtuple('Frame', ['filename', 'frame_number', 'img_data', 'timestamp'])
ExtractedChip = namedtuple('ExtractedChip', ['filename', 'frame_number', 'x', 'y', 'w', 'h', 'img_data', 'timestamp'])
logger = logging.getLogger('Chipper')


class Methods(Enum):
    OPENCV=1
    BACKGROUND_SUB=2


class FrameProducer(object):
    def __init__(self,
                 file_list,
                 open_func=open,
                 decoder='ffmpeg',
                 desired_framerate=2):
        self.current_file = 0
        self.file_list = file_list
        self.open_func = open_func
        self.decoder = decoder
        self.desired_framerate = desired_framerate

    def __iter__(self):
        for filename in self.file_list:
            logger.info('Staring file: {}'.format(filename))
            self.vid = imageio.get_reader(self.open_func(filename), self.decoder)
            self.vid_metadata = self.vid.get_meta_data()
            self.step_size = int(self.vid_metadata['fps']/self.desired_framerate)
            for frame_number in range(0, self.vid.get_length(), self.step_size):
                timestamp = self.__get_frame_time(filename, frame_number)
                yield Frame(filename, frame_number, self.vid.get_data(frame_number), timestamp)
        raise StopIteration()

    def __get_frame_time(self, filename, frame_number):
        # Get the number of seconds relative to the start of the video
        fps = self.vid_metadata['fps']
        seconds_from_start = frame_number / fps
        frame_delta = dt.timedelta(seconds=seconds_from_start)

        # Get the time from the file name, and the offset from that time as well
        # File names look like:
        #
        # /foo/bar/baz_20151001T223412-00600-01200.mp4
        #
        # Where '20151001T223412' is the date and time, and '00600' is the
        # offset from that time in seconds. The first video starts at the
        # correct time and has an offset of '00000'. Other videos after that
        # have offsets (normally in multiples of 10 minutes).
        base = os.path.basename(filename)
        tmp_str = base.split('_')[-1]
        time_str, start_second, _ = tmp_str.split('-')
        time = dt.datetime.strptime(time_str, "%Y%m%dT%H%M%S")
        time_delta = dt.timedelta(seconds=int(start_second))

        return time + time_delta + frame_delta


class Chipper(object):
    def __init__(self,
                 frame_producer,
                 mask_modifier=None,
                 box_expander=None,
                 kernel_size=(7, 7),
                 threshold=30,
                 chipping_method=Methods.BACKGROUND_SUB):
        self.frame_producer = frame_producer
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.mask_modifier = mask_modifier
        self.box_expander = box_expander
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.chipping_method = chipping_method

    def __iter__(self):
        if self.chipping_method == Methods.BACKGROUND_SUB:
            last_N_frames = deque()
            N = 10

        for frame in self.frame_producer:
            extracted_chips = []
            img_data = frame.img_data
            original_img_data = np.copy(frame.img_data)

            if self.chipping_method == Methods.OPENCV:
                fg_mask = self.fgbg.apply(img_data)
                if self.mask_modifier:
                    fg_mask = self.mask_modifier(fg_mask)
                img_data = cv2.bitwise_and(img_data, img_data, mask=fg_mask)
                difference_image = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
                if self.mask_modifier:
                    gray = self.mask_modifier(gray)

                if len(last_N_frames) < N:
                    last_N_frames.append(gray)
                    continue
                else:
                    background_image = np.median(last_N_frames, axis=0)
                    background_image = np.array(background_image, dtype=np.uint8)
                    difference_image = cv2.absdiff(background_image, gray)

                    _ = last_N_frames.popleft()
                    last_N_frames.append(gray)

            blurred_diff_image = cv2.GaussianBlur(difference_image, self.kernel_size, 0)
            _, th1 = cv2.threshold(blurred_diff_image, self.threshold, 255, cv2.THRESH_BINARY)
            _, contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 125:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                if self.box_expander:
                    x, y, w, h = self.box_expander(x, y, w, h)
                ec = ExtractedChip(filename=frame.filename,
                                   frame_number=frame.frame_number,
                                   x=x,
                                   y=y,
                                   w=w,
                                   h=h,
                                   img_data=np.copy(original_img_data[y:y+h, x:x+w]),
                                   timestamp=frame.timestamp,
                                   )
                extracted_chips.append(ec)
            yield extracted_chips
        raise StopIteration()


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(prog='chipper.py', description='Extract trips for a video')
    parser.add_argument("dataset_path", default="dataset_path", action="store", type=str,
                        help="Path to the dataset in hdfs.")

    parser.add_argument('string_to_match', type=str, help='string to match <str> in filename')
    args = parser.parse_args()

    from hdfs3 import HDFileSystem
    hdfs = HDFileSystem(host='namenode', port=8020)
    filenames = hdfs.glob(args.dataset_path)

    def get_info(filename):
        bname = os.path.basename(filename)
        return bname.split('-')[0], int(bname.split('-')[1])
    filenames = sorted(filenames, key=get_info)
    filenames_filtered = [filename for filename in filenames if args.string_to_match in filename]

    fp = FrameProducer(filenames_filtered, hdfs.open)

    chipper = Chipper(fp)
    count = 0
    for _ in chipper:
        count += 1
    logger.warn('Total Chips: {}'.format(count))

if __name__ == '__main__':
    main()
