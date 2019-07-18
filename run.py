import configparser
import logging
import logging.config

from multiprocessing import Process, Queue

from detector.barcode_detector import run_barcode_detector
from detector.ws_client import ClientWebSocket

logging.config.fileConfig('config/logging.ini')
logger = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read('config/config.ini')

logger.debug('Starting App')

# Launch process than run barcode detector, pass Queue
# Launch because TORNADO dont support multithreading
cars_detected = Queue()
p = Process(target=run_barcode_detector, args=(cars_detected, config))
p.start()
# run_barcode_detector(cars_detected, config)

# Lanza thread que analiza cola y manda al web socket
clientWs = ClientWebSocket(config.get('WS_SERVER', 'HOST'), int(config.get('DEFAULT', 'CAMERA')), cars_detected)
clientWs.ioloop.start()
