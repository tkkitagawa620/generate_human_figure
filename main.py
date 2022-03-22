import os
import cv2
import math
import numpy as np
import modules.VanishinPointDetector as vp

if __name__ == "__main__":
    # imgs = vp.ReadImage("imgs_input/karuizawa.jpg")
    Image = cv2.imread('imgs_input/karuizawa.jpg')
    Lines = vp.GetLines(Image)  # Getting the lines form the image
    VanishingPoint = vp.GetVanishingPoint(Lines)  # Get vanishing point

    # Checking if vanishing point found
    if VanishingPoint is None:
        print("Vanishing Point not found. Possible reason is that not enough lines are found in the image for determination of vanishing point.")

    # Draw lines and vanishing point
    for Line in Lines:
        cv2.line(Image, (Line[0], Line[1]), (Line[2], Line[3]), (0, 255, 0), 2)
    cv2.circle(Image, (int(VanishingPoint[0]), int(VanishingPoint[1])), 10, (0, 0, 255), -1)
    cv2.imwrite('imgs_output/karuizawa_module1.jpg', Image)
