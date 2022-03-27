
import random
import cv2
from modules.VanishinPointDetector import VanishingPointDetector
from modules.ScaleDetector import ScaleDetector

if __name__ == "__main__":
    img = cv2.imread('imgs_input/karuizawa.jpg', cv2.IMREAD_COLOR)
    mask_img = cv2.imread('imgs_input/karuizawa_masked.jpg')
    th, mask_img = cv2.threshold(mask_img, 128, 255, cv2.THRESH_BINARY)
    h, w = img.shape[:2]

    vpd = VanishingPointDetector(img)
    vp = vpd.getVanishingPoint()
    # vpd.visualize()

    positions = []
    while len(positions) < 3:
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        if mask_img[y][x][0] == 255:
            positions.append((x, y))
            cv2.circle(img, (x, y), 10, (255, 255, 0), 2)

    rp = (420, 900)
    rh = 150
    sd = ScaleDetector(img, vp, rp, rh)
    sd.drawHumanFigure()
    sd.drawVanishingPoint()
    for pos in positions:
        sd.drawTargetPoint(pos)
        sd.drawScaledFigure(pos)
    cv2.imshow('result', sd.img)
    cv2.waitKey(0)
