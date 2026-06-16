import cv2

from .types import Frame, Resolution


class CameraDevice:
    def __init__(self, source=0):
        self.source = source
        self.capture = None

    def open(self):
        self.capture = cv2.VideoCapture(self.source)
        return self.capture

    def close(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = None

    def read(self):
        if self.capture is None:
            return None
        ok, frame = self.capture.read()
        return frame if ok else None


class VideoAcquisition:
    def __init__(
        self,
        device: CameraDevice | None = None,
        resolution: Resolution | None = None,
        frameRate: int = 30,
    ):
        self.device = device or CameraDevice()
        self.resolution = resolution or Resolution(640, 480)
        self.frameRate = frameRate
        self._frameId = 0

    def start(self) -> None:
        self.device.open()

    def stop(self) -> None:
        self.device.close()

    def getFrame(self) -> Frame:
        frame = self.device.read()
        self._frameId += 1
        if frame is not None:
            frame = cv2.resize(frame, (self.resolution.width, self.resolution.height))
        return Frame(data=frame, frameId=self._frameId)
