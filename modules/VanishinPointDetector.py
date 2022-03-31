# Contribution
# https://github.com/KEDIARAHUL135/VanishingPoint
import cv2
import math
import numpy as np


class VanishingPointDetector():
    def __init__(self, img):
        self.img = img
        self.lines = None
        self.vp = None

        # Threshold by which lines will be rejected wrt the horizontal
        self.REJECT_DEGREE_TH = 4.0

    def visualize(self):
        if not self.lines:
            print("Supporting lines not stored in the instance. use getVanishingPoint method first.")
            exit(0)
        img = self.img.copy()
        for line in self.lines:
            cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 8)
            cv2.circle(img, (int(self.vp[0]), int(self.vp[1])), 20, (0, 0, 255), -1)
        cv2.imshow('Vanishing point and its supporting lines', img)
        # cv2.imwrite('karuizawa_module1.jpg', img)
        cv2.waitKey(0)

    def filterLines(self, lines):
        result_lines = []

        for line in lines:
            [[x1, y1, x2, y2]] = line

            if x1 != x2:
                m = (y2 - y1) / (x2 - x1)
            else:
                m = 100000000
            b = y2 - m * x2
            # theta will contain values between -90 -> +90.
            theta = math.degrees(math.atan(m))

            # Rejecting lines of slope near to 0 degree or 90 degree and storing others
            if self.REJECT_DEGREE_TH <= abs(theta) <= (90 - self.REJECT_DEGREE_TH):
                ll = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)    # length of the line
                result_lines.append([x1, y1, x2, y2, m, b, ll])

        if len(result_lines) > 15:
            result_lines = sorted(result_lines, key=lambda x: x[-1], reverse=True)
            result_lines = result_lines[:15]

        return result_lines

    def getLines(self):
        grayscale_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(grayscale_img, (5, 5), 1)
        edge_img = cv2.Canny(blurred_img, 40, 255)
        lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180, 50, 10, 15)

        if lines is None:
            print("Not enough lines found in the image for Vanishing Point detection.")
            exit(0)

        lines = self.filterLines(lines)
        self.lines = lines

        return lines

    def getVanishingPoint(self):
        vp = None
        min_error = 100000000000

        lines = self.getLines()

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                m1, b1 = lines[i][4], lines[i][5]
                m2, b2 = lines[j][4], lines[j][5]

                if m1 != m2:
                    x0 = (b1 - b2) / (m2 - m1)
                    y0 = m1 * x0 + b1

                    err = 0
                    for k in range(len(lines)):
                        m, b = lines[k][4], lines[k][5]
                        m_ = (-1 / m)
                        b_ = y0 - m_ * x0

                        x_ = (b - b_) / (m_ - m)
                        y_ = m_ * x_ + b_

                        ll = math.sqrt((y_ - y0)**2 + (x_ - x0)**2)

                        err += ll**2

                    err = math.sqrt(err)

                    if min_error > err:
                        min_error = err
                        vp = [x0, y0]

        self.vp = vp
        return vp
