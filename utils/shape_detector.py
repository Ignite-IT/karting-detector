import cv2


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
