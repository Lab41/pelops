import imageio
import cv2
from collections import namedtuple

Frame = namedtuple('Frame', ['filename', 'frame_number', 'img_data'])
ExtractedChip = namedtuple('ExtractedChip', ['filename', 'frame_number', 'x', 'y', 'w', 'h', 'img_data'])


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
            self.vid = imageio.get_reader(self.open_func(filename), self.decoder)
            self.vid_metadata = self.vid.get_meta_data()
            self.step_size = int(self.vid_metadata['fps']/self.desired_framerate)
            for frame_number in range(0, self.vid.get_length(), self.step_size):
                yield Frame(filename, frame_number, self.vid.get_data(frame_number))
        raise StopIteration()


class Chipper(object):
    def __init__(self, frame_producer):
        self.frame_producer = frame_producer
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    def __iter__(self):
        for frame in self.frame_producer:
            extracted_chips = []
            img_data = frame.img_data
            original_img_data = np.copy(frame.img_data)
            fg_mask = self.fgbg.apply(img_data)
            img_data = cv2.bitwise_and(img_data, img_data, mask=fg_mask)

            gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            ret, th1 = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            _, contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 50:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                ec = ExtractedChip(filename=frame.filename,
                                   frame=frame.fram_number,
                                   x=x,
                                   y=y,
                                   w=w,
                                   h=h,
                                   img_data=original_img_data[y:y+h, x:x+w])
                extracted_chips.append(ec)
            yield extracted_chips
        raise StopIteration()

