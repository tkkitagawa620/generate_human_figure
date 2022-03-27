import cv2
from modules.VanishinPointDetector import VanishingPointDetector
# from modules.ScaleDetector import ScaleDetector

if __name__ == "__main__":
    img = cv2.imread('imgs_input/karuizawa.jpg')
    vpd = VanishingPointDetector(img)
    vp = vpd.getVanishingPoint()
    vpd.visualize()
