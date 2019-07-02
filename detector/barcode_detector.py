import time

import cv2

import numpy as np
from queue import Queue

# import imutils  # Esta libreria sirve para manejar las diferentes versiones de OpenCv.

from enola_opencv_utils.rw_frames import VideoCapture, VideoWriter, VideoCaptureThread, ImageWriter

CANT_BYTES = 3


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c, epsilon=0.04):
        # initialize the shape name and approximate the contour
        shape = "unidentified"

        # Aca lo que hacemos es en base a una imagen que puede estar sucia jugamos con el 2do parametro
        # de approxPolyDP el cual une puntos y nos ayuda a formar determinada forma.
        # https://docs.opencv.org/4.0.0/dd/d49/tutorial_py_contour_features.html
        peri = cv2.arcLength(c, True)  # Con True le decimos que es una imagen rellena y no solo una curva
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

        # return the name of the shape
        return shape, approx

def check_barcodes(cnts):
    positions = []
    sd = ShapeDetector()
    sorted_cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])
    for c in sorted_cnts:
        M = cv2.moments(c)
        # Aseguro que m00 != 0
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            shape = 'unidentified'
            # Consigo la forma en base
            shape, approx = sd.detect(c)

            if False:  # shape not in ['rectangle', 'square']:
                # Ignore this contour
                continue

            x, y, w, h = cv2.boundingRect(approx)
            # rect = (x, y), (x + w, y + h)
            aspect_ratio = float(w) / h

            # Extent: Entender bien
            # rect_area = w * h
            # area = cv2.contourArea(approx)
            # extent = float(area) / rect_area
            # if extent < 0.7:
            #     continue

            #
            # AGREGAR MAS CONDICIONES PARA QUE SEA 1 o 0
            #
            print (aspect_ratio)
            if aspect_ratio >= 0.27 and aspect_ratio <= 0.6:
                positions.append(0)

            if aspect_ratio >= 0.61 and aspect_ratio <= 0.81:
                positions.append(1)

            cv2.drawContours(frame, [approx], -1, (0, 0, 255), 2)

    return positions

def bytes_to_number(positions):
    number = 0
    for i in range(len(positions)):
        if i == 0:
            number += positions[i] * 1
        else:
            number += positions[i] * (i * 2)
    return number

def countours_analize(frame, timestamp):
    sd = ShapeDetector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # Aca puedo usar Canny o Thresold -> Ver q conviene
    edged = cv2.Canny(blurred, 50, 150)

    cv2.imshow('gray', gray)
    cv2.imshow('blurred', blurred)
    cv2.imshow('edged', edged)

    # find contours in the edge map
    cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)

    status = "Nada"

    approxs = []
    index = -1
    # loop over the contours
    for c in cnts:
        index += 1
        M = cv2.moments(c)
        # Aseguro que m00 != 0
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            shape = 'unidentified'
            # Consigo la forma en base
            shape, approx = sd.detect(c)

            if shape not in ['rectangle', 'square', 'pentagon']:
                # Ignore this contour
                continue

            x, y, w, h = cv2.boundingRect(approx)
            rect = (x, y), (x + w, y + h)

            aspect_ratio = float(w) / h
            if aspect_ratio < 1 or aspect_ratio > 2:
                continue

            rect_area = w * h
            area = cv2.contourArea(approx)
            extent = float(area) / rect_area
            if extent < 0.7:
                continue

            print ("Childs " + str(index))
            hierarchy_w = hierarchy[0]
            childs_index = [i for i in range(len(hierarchy_w)) if (hierarchy_w[i][3] == index)]
            if len(childs_index) != CANT_BYTES:
                continue
            # childs_info = [hierarchy_w[i] for i in range(len(hierarchy_w)) if (hierarchy_w[i][3] == index)]
            childs_cnts = [cnts[i] for i in range(len(cnts)) if hierarchy_w[i][3] == index]

            positions = check_barcodes(childs_cnts)
            number = bytes_to_number(positions)
            print (positions)
            print (number)
            cars_detected.put({'time': time.time(), 'number': number})

            # crop_img = frame[y:y+h, x:x+w]
            # cv2.imshow("cropped", crop_img)

            # https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html
            # Dibujo el contour aproximado, no el real
            cv2.drawContours(frame, [approx], -1, (0, 0, 255), 2)
            # cv2.drawContours(frame, childs_cnts, -1, (0, 255, 0), 2)
            # approxs.append(approx)

    # draw the status text on the frame
    cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # cv2.drawContours(frame, approxs, -1, (0, 0, 255), 4)
    cv2.imshow('frame', frame)


# HOW USE
# video = VideoCapture('http://192.168.0.101:8080/stream/video/mjpeg?resolution=HD&&Username=admin&&Password=ZWR1YXJkb19u&&tempid=0.20093701226258998')
q = Queue()
# video = VideoCaptureThread('http://192.168.0.101:8080/stream/video/mjpeg?resolution=HD&&Username=admin&&Password=ZWR1YXJkb19u&&tempid=0.20093701226258998', max_queue=10)
video = VideoCaptureThread(0, max_queue=10)
# video = VideoCaptureThread('udp://@192.168.10.1:11111')  # Comandos: 'command', 'streamon'
video.start(q)
# video = VideoCapture(0)
# image_writer = ImageWriter('ip_1/', 'png')
# video_writer = VideoWriter('saves/drone_1')

# initial = time.time()

img = cv2.imread("../data_tests/dia_20190629/20190629-114707_96.png")
img = cv2.imread("../data_tests/dia_20190629/20190629-114707_97.png")
img = cv2.imread("../data_tests/dia_20190629/20190629-114707_98.png")
img = cv2.imread("../data_tests/dia_20190629/20190629-114707_102.png")

# img = cv2.imread("../data_tests/dia_20190629/20190629-114708_125.png")

cars_detected = Queue()
#
# CREAR THREAD QUE ANALICE LOS AUTOS DETECTADOS Y LOS ENVIE A DONDE CORRESPONDA
#
# while True:
while(video.is_opened()):
    # frame = img
    # timestamp = time.time()

    ret, frame = video.read()

    data = q.get()
    frame = data['frame']
    timestamp = data['time']

    # image_writer.save_frame(frame)
    # video_writer.save_frame(frame)

    countours_analize(frame, timestamp)

    # cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop Thread
video.stop()
# Release everything if job is finished
cv2.destroyAllWindows()
