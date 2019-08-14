import configparser
import logging
import logging.config

from multiprocessing import Process, Queue

from detector.detector import ManagerDetector
from detector.ws_client import ClientWebSocket

logging.config.fileConfig('config/logging.ini')
logger = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read('config/config.ini')

logger.debug('Starting App')

# Launch process than run barcode detector, pass Queue
# Launch because TORNADO dont support multithreading
def launch_detector_process(config, queue):
    manager = ManagerDetector(config, queue)
    manager.run()
cars_detected = Queue()
p = Process(target=launch_detector_process, args=(config, cars_detected))
p.start()

# Lanza thread que analiza cola y manda al web socket
clientWs = ClientWebSocket(config.get('WS_SERVER', 'HOST'), int(config.get('DEFAULT', 'CAMERA')), cars_detected)
clientWs.ioloop.start()
