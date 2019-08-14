import logging
import time
import cv2

from queue import Queue

from .strategy_detector_barcode import StrategyDetectorBarcode
from .strategy_detector_digits import StrategyDetectorDigits

from enola_opencv_utils.rw_frames import VideoCaptureThread, ImageWriter


class ManagerDetector():

    def __init__(self, config, queue):
        self.detections = queue
        self.config = config
        self.show_video = True if self.config.get('DEFAULT', 'SHOW') == '1' else False

        settings = [
            [cv2.CAP_PROP_FPS, int(self.config.get('CAMERA', 'FPS'))],
            [cv2.CAP_PROP_FRAME_WIDTH, int(self.config.get('CAMERA', 'WIDTH'))],
            [cv2.CAP_PROP_FRAME_HEIGHT, int(self.config.get('CAMERA', 'HEIGHT'))]
        ]

        self.q = Queue()
        self.video = VideoCaptureThread(int(self.config.get('CAMERA', 'PATH')), max_queue=100, settings=settings)        
        self.video.start(self.q)
        # video = VideoCapture('http://192.168.0.101:8080/stream/video/mjpeg?resolution=HD&&Username=admin&&Password=ZWR1YXJkb19u&&tempid=0.20093701226258998')
        # video = VideoCaptureThread('http://192.168.0.101:8080/stream/video/mjpeg?resolution=HD&&Username=admin&&Password=ZWR1YXJkb19u&&tempid=0.20093701226258998', max_queue=10)
        # video = VideoCaptureThread('udp://@192.168.10.1:11111')  # Comandos: 'command', 'streamon'

        # video = VideoCapture(0)
        # image_writer = ImageWriter('data_tests/errors/5/img_', 'png')
        # video_writer = VideoWriter('saves/drone_1')

        # initial = time.time()
        # img = cv2.imread("data_tests/dia_20190707/img_20190707-182011_29.png")

        logging.debug(self.video.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        logging.debug(self.video.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.debug(self.video.cap.get(cv2.CAP_PROP_FPS))

    def run(self):
        # while True:
        while(self.video.is_opened()):
            # frame = img
            # timestamp = time.time()

            # ret, frame = video.read()
            # print (q.qsize())
            data = self.q.get()
            frame = data['frame']
            timestamp = data['time']

            self.load_strategy()
            self.strategy.analize(frame, timestamp)

            #if self.show_video:
            # cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Stop Thread
        self.video.stop()
        if self.show_video:
            # Release everything if job is finished
            cv2.destroyAllWindows()

    def load_strategy(self):
        self.strategy = self._get_strategy()

    def _get_strategy(self):
        if self.config.get('DEFAULT', 'STRATEGY') == 'BARCODE':
            return StrategyDetectorBarcode(self)
        if self.config.get('DEFAULT', 'STRATEGY') == 'DIGITS':
            return StrategyDetectorDigits(self)
