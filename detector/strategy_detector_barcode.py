import logging
import cv2

import numpy as np

from scipy.spatial import distance as dist
from imutils import perspective

from .strategy_detector import StrategyDetector

from utils.shape_detector import ShapeDetector


class StrategyDetectorBarcode(StrategyDetector):

    def __init__(self, manager):
        self.manager = manager
        self.cant_bytes = int(self.manager.config.get('BARCODE', 'CANT_BYTES'))

    def analize(self, frame, timestamp):
        image = self.treat_canny_image(frame)
        cnts, hierarchy = self.get_contours(image)

        index = -1
        # loop over the contours
        for c in cnts:
            index += 1

            if not self.is_contour_valid(index, c, cnts, hierarchy):
                continue

            hierarchy_w = hierarchy[0]
            # childs_info = [hierarchy_w[i] for i in range(len(hierarchy_w)) if (hierarchy_w[i][3] == index)]
            childs_cnts = [cnts[i] for i in range(len(cnts)) if hierarchy_w[i][3] == index]
            if self.manager.show_video:
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

            # Relación entre el área de contorno y el área del rectángulo delimitador
            # Ver bien cual es el valor correcto y contourArea conviene con 'c' o 'approx'
            x, y, w, h = cv2.boundingRect(c)
            # aspect_ratio = aspect_ratio_h if aspect_ratio_h > aspect_ratio_w else aspect_ratio_w
            width = w if w > h else h
            height = h if h < w else w

            positions = self.check_barcodes(frame, childs_cnts, width, height)
            number = self.bytes_to_number(positions)
            if len(positions) != self.cant_bytes:
                # image_writer.save_frame(frame)
                continue

            logging.debug('NUMERO: %d, %f' % (number, timestamp))
            self.manager.detections.put({'timestamp': timestamp, 'number': number})

            if self.manager.show_video:
                # crop_img = frame[y:y+h, x:x+w]
                # cv2.imshow("cropped", crop_img)

                # https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html
                # Dibujo el contour aproximado, no el real
                # cv2.drawContours(frame, [box], -1, (0, 0, 255), 2)
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

        if self.manager.show_video:
            # draw the status text on the frame
            # cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow('frame', frame)

    def is_contour_valid(self, index, c, cnts, hierarchy):
        sd = ShapeDetector()

        m = cv2.moments(c)
        # Aseguro que m00 != 0
        if m["m00"] == 0:
            return False

        # center = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
        # Consigo la forma en base
        shape, approx = sd.detect(c, 0.1)
        if shape not in ['rectangle', 'square', 'pentagon']:
            # Ignore this contour
            return False

        # Buscar hijos y que haya la misma cantidad que CANT_BYTES
        hierarchy_w = hierarchy[0]
        childs_index = [i for i in range(len(hierarchy_w)) if (hierarchy_w[i][3] == index)]
        if len(childs_index) != self.cant_bytes:
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
        if (aspect_ratio_w < 1.4 or aspect_ratio_w > 1.8) and (aspect_ratio_h < 1.4 or aspect_ratio_h > 1.8):
            return False

        # print (aspect_ratio_w)
        # print (aspect_ratio_h)

        # Relación entre el área de contorno y el área del rectángulo delimitador
        # Ver bien cual es el valor correcto y contourArea conviene con 'c' o 'approx'
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        rect_area = w * h
        extent = float(area) / rect_area
        if extent < 0.6 or extent > 1:
            return False

        return True

    def midpoint(self, pt_a, pt_b):
        return ((pt_a[0] + pt_b[0]) * 0.5, (pt_a[1] + pt_b[1]) * 0.5)

    def check_barcodes(self, frame, cnts, w, h):
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
                    (tltrX, tltrY) = self.midpoint(tl, tr)
                    (blbrX, blbrY) = self.midpoint(bl, br)

                    # compute the midpoint between the top-left and top-right points,
                    # followed by the midpoint between the top-righ and bottom-right
                    (tlblX, tlblY) = self.midpoint(tl, bl)
                    (trbrX, trbrY) = self.midpoint(tr, br)

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

    def bytes_to_number(self, positions):
        number = 0
        for i in range(len(positions)):
            if i == 0:
                number += positions[i] * 1
            else:
                number += positions[i] * (2**i)
        return number
