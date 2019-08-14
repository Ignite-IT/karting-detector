import cv2


class StrategyDetector():

    def __init__(self, manager):
        self.manager = manager

    def analize(self, frame, timestamp):
        raise Exception("Not Implementer")

    def treat_canny_image(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        # Aca puedo usar Canny o Thresold -> VER QUE CONVIENE
        edged = cv2.Canny(blurred, 50, 150)

        if self.manager.show_video:
            cv2.imshow('gray', gray)
            cv2.imshow('blurred', blurred)
            cv2.imshow('edged', edged)

        return edged, blurred, gray

    def get_contours(self, image):
        # find contours in the edge map
        cnts, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(cnts) -> Other way
        return cnts, hierarchy
