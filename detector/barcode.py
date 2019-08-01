import cv2
import numpy as np

from scipy.spatial import distance as dist
from imutils import perspective

from utils.shape_detector import ShapeDetector


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


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
