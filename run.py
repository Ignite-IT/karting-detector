from multiprocessing import Process, Queue

from detector.barcode_detector import run_barcode_detector
from detector.ws_client import ClientWebSocket

# Launch process than run barcode detector, pass Queue
# Launch because TORNADO dont support multithreading
cars_detected = Queue()
p = Process(target=run_barcode_detector, args=(cars_detected,))
p.start()

# Lanza thread que analiza cola y manda al web socket
clientWs = ClientWebSocket("ws://localhost:8888/camera", 1, cars_detected)
clientWs.ioloop.start()
