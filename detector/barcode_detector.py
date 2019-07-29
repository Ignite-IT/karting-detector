import time

import cv2

import numpy as np
from queue import Queue

from scipy.spatial import distance as dist
import imutils  # Esta libreria sirve para manejar las diferentes versiones de OpenCv.
from imutils import perspective
from imutils import contours
from imutils.perspective import four_point_transform

from enola_opencv_utils.rw_frames import VideoCaptureThread, ImageWriter

CANT_BYTES = 4

DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (1, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 1, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c, epsilon=0.04, print_len=False):
        # initialize the shape name and approximate the contour
        shape = "unidentified"

        # Aca lo que hacemos es en base a una imagen que puede estar sucia jugamos con el 2do parametro
        # de approxPolyDP el cual une puntos y nos ayuda a formar determinada forma.
        # https://docs.opencv.org/4.0.0/dd/d49/tutorial_py_contour_features.html
        peri = cv2.arcLength(c, True)  # Con True le decimos que es una imagen rellena y no solo una curva
        approx = cv2.approxPolyDP(c, epsilon * peri, True)
        if print_len:
            print (len(approx))

        # if the shape is a rect, it will have 2 vertices
        if len(approx) == 2:
            shape = "rect"

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

        elif len(approx) == 6:
            shape = "hexagon"

        elif len(approx) == 7:
            shape = "heptagon"

        elif len(approx) == 8:
            shape = "octagon"

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

        # return the name of the shape
        return shape, approx

def check_digits(frame, gray, cnt, cnts):
    # extract the thermostat display, apply a perspective transform
    # to it
    warped = four_point_transform(gray, cnt.reshape(4, 2))
    output = four_point_transform(frame, cnt.reshape(4, 2))

    # threshold the warped image, then apply a series of morphological
    # operations to cleanup the thresholded image
    thresh = cv2.threshold(warped, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # find contours in the thresholded image, then initialize the
    # digit contours lists
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    digitCnts = []

    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # if the contour is sufficiently large, it must be a digit
        if w < 10 or h < 10:
            continue
        digitCnts.append(c)
        cv2.drawContours(output, [c], -1, (0, 255, 255), 2)

    if len(digitCnts) == 0:
        return "0"

    # sort the contours from left-to-right, then initialize the
    # actual digits themselves
    digitCnts = contours.sort_contours(digitCnts,
                                       method="left-to-right")[0]
    digits = []

    try:
        # loop over each of the digits
        for c in digitCnts:
            # extract the digit ROI
            (x, y, w, h) = cv2.boundingRect(c)
            roi = thresh[y:y + h, x:x + w]

            # compute the width and height of each of the 7 segments
            # we are going to examine
            (roiH, roiW) = roi.shape
            (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
            dHC = int(roiH * 0.05)

            # define the set of 7 segments
            segments = [
                ((0, 0), (w, dH)),  # top
                ((0, 0), (dW, h // 2)), # top-left
                ((w - dW, 0), (w, h // 2)), # top-right
                ((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
                ((0, h // 2), (dW, h)), # bottom-left
                ((w - dW, h // 2), (w, h)), # bottom-right
                ((0, h - dH), (w, h))   # bottom
            ]
            on = [0] * len(segments)

            # loop over the segments
            for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                # extract the segment ROI, count the total number of
                # thresholded pixels in the segment, and then compute
                # the area of the segment
                segROI = roi[yA:yB, xA:xB]
                total = cv2.countNonZero(segROI)
                area = (xB - xA) * (yB - yA)

                # if the total number of non-zero pixels is greater than
                # 50% of the area, mark the segment as "on"
                if total / float(area) > 0.5:
                    on[i] = 1
                cv2.imshow("roi" + str(i), segROI)

            # lookup the digit and draw it on the image
            try:
                digit = DIGITS_LOOKUP[tuple(on)]
                digits.append(digit)
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(output, str(digit), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            except:
                print ("error")
                print (on)
                print ("")
                pass
    except:
        return "0"

    if len(digits) == 0:
        return "0"
    number = "".join(str(d) for d in digits)
    cv2.imshow("Output", output)
    cv2.imshow("Tresh", thresh)
    return number

def check_barcodes(frame, cnts, w, h):
    positions = []
    sd = ShapeDetector()
    sorted_cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0], reverse=False)
    pixelsPerMetric = 1  # aspect_ratio  # None
    for c in sorted_cnts:
        m = cv2.moments(c)
        # Aseguro que m00 != 0
        if m["m00"] != 0:
            # center = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))

            shape, approx = sd.detect(c)
            if shape == 'circle':
                # Ignore this contour
                break

            # CALCULANDO CON ASPECT RATIO
            # rect = cv2.minAreaRect(approx)
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # w = rect[1][0]
            # h = rect[1][1]
            # aspect_ratio_w = float(w) / h
            # aspect_ratio_h = float(h) / w
            #
            # AGREGAR MAS CONDICIONES PARA QUE SEA 1 o 0
            #
            # if (aspect_ratio_w >= 0.16 and aspect_ratio_w <= 0.24) or (aspect_ratio_h >= 0.16 and aspect_ratio_h <= 0.24):
            #     positions.append(0)
            # if (aspect_ratio_w >= 0.28 and aspect_ratio_w <= 0.36) or (aspect_ratio_h >= 0.28 and aspect_ratio_h <= 0.36):
            #     positions.append(1)
            # END - CALCULANDO CON ASPECT RATIO

            # CALCULANDO CON TAMANOS
            # https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype="int")

            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 0), 2)

            # loop over the original points and draw them
            for (x, y) in box:
                # cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

                # unpack the ordered bounding box, then compute the midpoint
                # between the top-left and top-right coordinates, followed by
                # the midpoint between bottom-left and bottom-right coordinates
                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)

                # compute the midpoint between the top-left and top-right points,
                # followed by the midpoint between the top-righ and bottom-right
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)

                # draw the midpoints on the image
                # cv2.circle(frame, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                # cv2.circle(frame, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                # cv2.circle(frame, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                # cv2.circle(frame, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

                # draw lines between the midpoints
                # cv2.line(frame, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                #          (255, 0, 255), 2)
                # cv2.line(frame, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                #          (255, 0, 255), 2)

                # compute the Euclidean distance between the midpoints
                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

                # NOT USE
                # if the pixels per metric has not been initialized, then
                # compute it as the ratio of pixels to supplied metric
                # (in this case, inches)
                # if pixelsPerMetric is None:
                #     pixelsPerMetric = dB / 0.9995  # args["width"]
                # compute the size of the object
                dimA = dA  # / pixelsPerMetric  # height
                dimB = dB  # / pixelsPerMetric  # width

                width = dB if dB < dA else dA
                percent_width = ((width * 100) / w)
                if percent_width > 7.8 and percent_width < 10.8:
                    positions.append(0)
                if percent_width > 11.6 and percent_width < 15.4:
                    positions.append(1)

                # draw the object sizes on the image
                cv2.putText(frame, "{:.1f}%".format(percent_width),
                            (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 255, 255), 2)
                # cv2.putText(frame, "{:.1f}in".format(dimA),
                #             (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.65, (255, 255, 255), 2)
                # cv2.putText(frame, "{:.1f}in".format(dimB),
                #             (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.65, (255, 255, 255), 2)

                # Only iterate once because always calculate the same
                break
            # END - CALCULANDO CON TAMANOS

            # cv2.drawContours(frame, [approx], -1, (0, 0, 255), 2)
            # break

    return positions


def bytes_to_number(positions):
    number = 0
    for i in range(len(positions)):
        if i == 0:
            number += positions[i] * 1
        else:
            number += positions[i] * (2**i)
    return number


def countours_analize(frame, timestamp, cars_detected, image_writer = None):
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
            shape, approx = sd.detect(c, 0.02)
            if shape not in ['rectangle', 'square']:  #, 'pentagon']:
                # Ignore this contour
                continue

            # Buscar hijos y que haya la misma cantidad que CANT_BYTES
            hierarchy_w = hierarchy[0]
            childs_index = [i for i in range(len(hierarchy_w)) if (hierarchy_w[i][3] == index)]
            if len(childs_index) != 2:
                continue
            # if len(childs_index) != CANT_BYTES:
            #     continue
            # childs_info = [hierarchy_w[i] for i in range(len(hierarchy_w)) if (hierarchy_w[i][3] == index)]
            childs_cnts = [cnts[i] for i in range(len(cnts)) if hierarchy_w[i][3] == index]

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

            if True:
                number = int(check_digits(frame, gray, approx, childs_cnts))
            if False:
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
        [cv2.CAP_PROP_FRAME_WIDTH, int(config.get('CAMERA', 'WIDTH'))],
        [cv2.CAP_PROP_FRAME_HEIGHT, int(config.get('CAMERA', 'HEIGHT'))],
        [cv2.CAP_PROP_FPS, int(config.get('CAMERA', 'FPS'))]
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
