import cv2
import os


class VideoReader:
    def __init__(self, filename):
        self.filename = filename
        self.cap = cv2.VideoCapture(self.filename)
        self.is_ok = self.cap.isOpened()

        if not self.is_ok:
            raise RuntimeError("Error opening video stream or file")

    def read(self):
        if not self.is_ok:
            raise RuntimeError("Error opening video stream or file")

        self.is_ok, frame = self.cap.read()
        return self.is_ok, frame

    def release(self):
        self.cap.release()


class VideoType:
    MP4 = "mp4"
    AVI = "avi"

class VideoWriter:
    FORMAT = {
        VideoType.MP4 : 'XVID',
        VideoType.AVI : 'MJPG',
    }

    def __init__(self, filename, file_type=VideoType.AVI):
        self.filename = filename + "." + file_type
        self.file_type = file_type
        self.width, self.height = None, None
        self.out = None

    def write(self, image):
        h, w, _ = image.shape
        if self.out is None:
            self.width, self.height = w, h
            self.out = cv2.VideoWriter(
                self.filename,
                cv2.VideoWriter_fourcc(*VideoWriter.FORMAT[self.file_type]),
                24, (self.width, self.height))
        else:
            assert self.width == w and self.height == h

        self.out.write(image)

    def release(self):
        self.out.release()


if __name__ == "__main__":
    vr = VideoReader("/data/wlt/videos/office-1.mp4")
    frame = vr.read()
    print (type(frame), frame.shape)
