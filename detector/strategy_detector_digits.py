import logging
import cv2

import numpy as np

import imutils  # Esta libreria sirve para manejar las diferentes versiones de OpenCv.
from imutils import contours
from imutils.perspective import four_point_transform

from .strategy_detector import StrategyDetector

from utils.shape_detector import ShapeDetector


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
    (1, 1, 1, 1, 0, 1, 0): 9
}


class StrategyDetectorDigits(StrategyDetector):

    def __init__(self, manager):
        self.manager = manager
        self.cant_digits = int(self.manager.config.get('DIGITS', 'CANT_DIGITS'))

    def analize(self, frame, timestamp):
        edged, blurred, gray = self.treat_canny_image(frame)
        cnts, hierarchy = self.get_contours(edged)

        index = -1
        # loop over the contours
        for c in cnts:
            index += 1

            if not self.is_contour_valid(index, c, cnts, hierarchy, frame):
                continue

            # hierarchy_w = hierarchy[0]
            # childs_info = [hierarchy_w[i] for i in range(len(hierarchy_w)) if (hierarchy_w[i][3] == index)]
            # childs_cnts = [cnts[i] for i in range(len(cnts)) if hierarchy_w[i][3] == index]
            if self.manager.show_video:
                # cv2.drawContours(frame, childs_cnts, -1, (0, 255, 255), 2)
                pass

            sd = ShapeDetector()
            shape, approx = sd.detect(c, 0.1)
            number = int(self.check_digits(frame, gray, approx))

            logging.debug('NUMERO: %d, %f' % (number, timestamp))
            self.manager.detections.put({'timestamp': timestamp, 'number': number})

            if self.manager.show_video:
                # crop_img = frame[y:y+h, x:x+w]
                # cv2.imshow("cropped", crop_img)

                # https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html
                # Dibujo el contour aproximado, no el real
                # cv2.drawContours(frame, [box], -1, (0, 0, 255), 2)
                # cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                pass

        if self.manager.show_video:
            # draw the status text on the frame
            # cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow('frame', frame)

    def is_contour_valid(self, index, c, cnts, hierarchy, frame):
        sd = ShapeDetector()

        m = cv2.moments(c)
        # Aseguro que m00 != 0
        if m["m00"] == 0:
            return False

        # center = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
        # Consigo la forma en base
        shape, approx = sd.detect(c, 0.1)
        if shape not in ['rectangle', 'square']:
            # Ignore other shpaes because after I need only 4 points
            return False

        # Buscar hijos y que haya la misma cantidad que CANT_DIGITS
        hierarchy_w = hierarchy[0]
        childs_index = [i for i in range(len(hierarchy_w)) if (hierarchy_w[i][3] == index)]
        if len(childs_index) < self.cant_digits:
            return False

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
        # CALCULAR LOS ASPECT_RATIO DIVIDIENDO W/H Y H/W YA QUE SEGUN COMO LES DE EN UN MOMENTO SE ROTA
        # ESTO HAY QUE AJUSTARLO SEGUN LA POSICION DEL KARTING
        #
        # DEFINIR........................!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #
        if (aspect_ratio_w < 1.2 or aspect_ratio_w > 2.4) and (aspect_ratio_h < 1.2 or aspect_ratio_h > 2.4):
            return False

        # Relación entre el área de contorno y el área del rectángulo delimitador
        # Ver bien cual es el valor correcto y contourArea conviene con 'c' o 'approx'
        # Lo que hacemos aca es calcular las areas. boundingRect no toma la rotacion, entonces cuanto mas lejos del 1
        # este extent mas rotado en Y va a estar. Entonces cuando llega a 0.53 esta demasiado rotado para nosotros.
        area = cv2.contourArea(approx)
        x, y, w, h = cv2.boundingRect(approx)
        rect_area = w * h
        extent = float(area) / rect_area
        if extent < 0.53 or extent > 1:
            return False

        # print (shape)
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
        # cv2.drawContours(frame, [c], -1, (0, 0, 255), 2)

        return True

    def check_digits(self, frame, gray, cnt):
        # extract the thermostat display, apply a perspective transform to it
        warped = four_point_transform(gray, cnt.reshape(4, 2))
        output = four_point_transform(frame, cnt.reshape(4, 2))

        # threshold the warped image, then apply a series of morphological
        # operations to cleanup the thresholded image
        thresh = cv2.threshold(warped, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        #
        # ACA PUEDO VER DE AGRANDAR LA IMAGEN SI ES MUY CHICA
        # VER QUE SE CONSIDERA CHICO
        # height, width = thresh.shape[:2]
        # thresh = cv2.resize(thresh, None, fx=4, fy=4)

        # find contours in the thresholded image, then initialize the
        # digit contours lists
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        digit_cnts = []

        # loop over the digit area candidates
        for c in cnts:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            # if the contour is sufficiently large, it must be a digit
            #
            # CUIDADO: Este numero se va achicando cuando esta muy lejos la imagen
            #
            if w < 10 or h < 10:
                continue
            digit_cnts.append(c)
            cv2.drawContours(output, [c], -1, (0, 255, 255), 2)

        if len(digit_cnts) == 0:
            return "0"

        # sort the contours from left-to-right, then initialize the
        # actual digits themselves
        digit_cnts = contours.sort_contours(digit_cnts,
                                            method="left-to-right")[0]
        digits = []

        try:
            # loop over each of the digits
            for c in digit_cnts:
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
                    #
                    # CON ESTE NUMERO SE PUEDE JUGAR
                    # DEFECTO: 0.5
                    if total / float(area) > 0.4:
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
        if self.manager.show_video:
            cv2.imshow("Output", output)
            cv2.imshow("Tresh", thresh)

        return number
