import time

import cv2

import numpy as np
from queue import Queue

from .barcode import check_barcodes, bytes_to_number
from .digits import check_digits

from utils.shape_detector import ShapeDetector
from enola_opencv_utils.rw_frames import VideoCaptureThread, ImageWriter

CANT_BYTES = 4


def countours_analize(frame, timestamp, cars_detected, image_writer=None):
    #
    # HACER CONFIGURABLE TODO LO QUE VA DEFINIENDO QUE COUNTOURS NOS QUEDAMOS Y CUALES NO
    #

    sd = ShapeDetector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # Aca puedo usar Canny o Thresold -> VER QUE CONVIENE
    edged = cv2.Canny(blurred, 50, 150)

    cv2.imshow('gray', gray)
    cv2.imshow('blurred', blurred)
    cv2.imshow('edged', edged)

    # find contours in the edge map
    cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts) -> Other way

    index = -1
    # loop over the contours
    for c in cnts:
        index += 1
        # AVERIGUAR BIEN QUE RETORNA moments
        m = cv2.moments(c)
        # Aseguro que m00 != 0
        if m["m00"] != 0:
            # center = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))

            shape = 'unidentified'
            # Consigo la forma en base
            shape, approx = sd.detect(c, 0.1)
            if shape not in ['rectangle', 'square']:  #, 'pentagon']:
                # Ignore this contour
                continue

            # Buscar hijos y que haya la misma cantidad que CANT_BYTES
            hierarchy_w = hierarchy[0]
            childs_index = [i for i in range(len(hierarchy_w)) if (hierarchy_w[i][3] == index)]
            if len(childs_index) != 2 and len(childs_index) != 3:
                continue
            # if len(childs_index) != CANT_BYTES:
            #     continue
            # childs_info = [hierarchy_w[i] for i in range(len(hierarchy_w)) if (hierarchy_w[i][3] == index)]
            childs_cnts = [cnts[i] for i in range(len(cnts)) if hierarchy_w[i][3] == index]
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            # https://docs.opencv.org/trunk/d1/d32/tutorial_py_contour_properties.html

            # Conseguimos la recta rotada para calcular el aspect ratio correcto.
            # x, y, w, h = cv2.boundingRect(approx) -> OLD
            # aspect_ratio = float(w) / h -> OLD
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            w = rect[1][0]
            h = rect[1][1]
            aspect_ratio_w = float(w) / h
            aspect_ratio_h = float(h) / w
            # CALCULAR LOS ASPECT_RATIO DIVIDIENDO W/H Y H/W YA QUE SEGUN COMO LES EN UN MOMENTO SE ROTA
            # ESTO HAY QUE AJUSTARLO SEGUN LA POSICION DEL KARTING
            if (aspect_ratio_w < 1.2 or aspect_ratio_w > 2.4) and (aspect_ratio_h < 1.2 or aspect_ratio_h > 2.4):
                continue

            # print (aspect_ratio_w)
            # print (aspect_ratio_h)

            # Relación entre el área de contorno y el área del rectángulo delimitador
            # Ver bien cual es el valor correcto y contourArea conviene con 'c' o 'approx'
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            rect_area = w * h
            extent = float(area) / rect_area
            if extent < 0.5 or extent > 1:
                continue

            # aspect_ratio = aspect_ratio_h if aspect_ratio_h > aspect_ratio_w else aspect_ratio_w
            width = w if w > h else h
            height = h if h < w else w

            if False:
                number = int(check_digits(frame, gray, approx, childs_cnts))
            if True:
                positions = check_barcodes(frame, childs_cnts, width, height)
                number = bytes_to_number(positions)
                if len(positions) != CANT_BYTES:
                    # image_writer.save_frame(frame)
                    continue
            print ('NUMERO: %d, %f' % (number, timestamp))
            cars_detected.put({'timestamp': timestamp, 'number': number})

            # crop_img = frame[y:y+h, x:x+w]
            # cv2.imshow("cropped", crop_img)

            # https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html
            # Dibujo el contour aproximado, no el real
            # cv2.drawContours(frame, [box], -1, (0, 0, 255), 2)
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

    # draw the status text on the frame
    # cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('frame', frame)


def run_barcode_detector(cars_detected, config):
    settings = [
        [cv2.CAP_PROP_FPS, int(config.get('CAMERA', 'FPS'))],
        [cv2.CAP_PROP_FRAME_WIDTH, int(config.get('CAMERA', 'WIDTH'))],
        [cv2.CAP_PROP_FRAME_HEIGHT, int(config.get('CAMERA', 'HEIGHT'))]
    ]
    # HOW USE
    # video = VideoCapture('http://192.168.0.101:8080/stream/video/mjpeg?resolution=HD&&Username=admin&&Password=ZWR1YXJkb19u&&tempid=0.20093701226258998')
    q = Queue()
    # video = VideoCaptureThread('http://192.168.0.101:8080/stream/video/mjpeg?resolution=HD&&Username=admin&&Password=ZWR1YXJkb19u&&tempid=0.20093701226258998', max_queue=10)
    video = VideoCaptureThread(int(config.get('CAMERA', 'PATH')), max_queue=100, settings=settings)
    # video = VideoCaptureThread('udp://@192.168.10.1:11111')  # Comandos: 'command', 'streamon'
    video.start(q)
    # video = VideoCapture(0)
    # image_writer = ImageWriter('data_tests/errors/5/img_', 'png')
    # video_writer = VideoWriter('saves/drone_1')

    # initial = time.time()
    # img = cv2.imread("data_tests/dia_20190707/img_20190707-182011_29.png")

    print (video.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print (video.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print (video.cap.get(cv2.CAP_PROP_FPS))

    # while True:
    while(video.is_opened()):
        # frame = img
        # timestamp = time.time()

        # ret, frame = video.read()
        # print (q.qsize())
        data = q.get()
        frame = data['frame']
        timestamp = data['time']

        # image_writer.save_frame(frame)
        # video_writer.save_frame(frame)

        countours_analize(frame, timestamp, cars_detected)

        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Stop Thread
    video.stop()
    # Release everything if job is finished
    cv2.destroyAllWindows()
