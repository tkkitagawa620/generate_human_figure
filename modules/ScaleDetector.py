import cv2


class ScaleDetector():
    def __init__(self, img, vanishingPoint, referencePoint, referenceHeight):
        # Vanishing point
        self.vp = vanishingPoint
        # Y and X cordinates of the reference figure origin
        self.rp = referencePoint
        self.rp_x = rp[0]
        self.rp_y = rp[1]
        self.rp_h = referenceHeight
        # Load image
        self.img = img

    def drawHumanFigure(self, x=None, y=None, h=None, color=None):
        x = x or self.rp_x
        y = y or self.rp_y
        h = h or self.rp_h
        color = color or (0, 255, 0)
        thickness = 4
        fig_w = 40

        top_y = y - h
        top = (x, top_y)
        rp = (x, y)
        _neckline_y = int((y - top_y) * 0.1429 + top_y)

        cv2.line(img, rp, top, color, thickness)
        cv2.line(img, (x - fig_w, top_y), (x + fig_w, top_y), color, thickness)
        cv2.line(img, (x - fig_w, y), (x + fig_w, y), color, thickness)
        cv2.line(img, (x - fig_w, _neckline_y), (x + fig_w, _neckline_y), color, thickness)
        return None

    def drawVanishingPoint(self):
        cv2.circle(self.img, tuple(map(int, self.vp)), 5, (0, 255, 0), 4)
        return None

    def drawTargetPoint(self, tp):
        cv2.circle(img, tuple(map(int, tp)), 5, (0, 0, 255), 4)

    def getExtendedLine(self, img, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        w = img.shape[0]
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        y_0 = b
        y_w = m * w + b
        # print('m: ', m)
        # print('b: ', b)
        # print('y_0: ', y_0)
        # print('y_w: ', y_w)
        # print('extended line cordinates: ', ((0, y_0), (w, y_w)))
        return ((0, y_0), (w, y_w), m, b)

    def findIntersection(self, l1, l2):
        (a, b), (c, d) = l1
        (e, f), (g, h) = l2
        # print(a, b, c, d, e, f, g, h)
        x_deno = (d - b) * (g - e) - (c - a) * (h - f)
        x_nume = (f * g - e * h) * (c - a) - (b * c - a * d) * (g - e)
        y_deno = (d - b) * (g - e) - (c - a) * (h - f)
        y_nume = (f * g - e * h) * (d - b) - (b * c - a * d) * (h - f)
        return (x_nume / x_deno, y_nume / y_deno)

    def calculateFigureHeight(self, tp):
        # Init
        tp_x = tp[0]
        tp_y = tp[1]
        tp_h = None
        v_img = self.img.copy()

        # Step 1
        print(self.rp)
        sp1 = self.findIntersection((self.rp, (0, self.rp_y)), (self.vp, tp))
        hl = self.getExtendedLine(v_img, self.rp, (0, self.rp_y))
        cv2.line(v_img, tuple(map(int, hl[0])), tuple(map(int, hl[1])), (255, 0, 0), 2)
        cv2.circle(v_img, tuple(map(int, sp1)), 4, (0, 255, 0), 4)

        # Step 2
        sp2 = (sp1[0], sp1[1] - self.rp_h)
        cv2.circle(v_img, tuple(map(int, sp2)), 4, (0, 255, 0), 4)

        # Step 3
        sl1 = self.getExtendedLine(v_img, self.vp, sp1)
        sl2 = self.getExtendedLine(v_img, self.vp, sp2)
        cv2.line(v_img, tuple(map(int, sl1[0])), tuple(map(int, sl1[1])), (255, 0, 0), 2)
        cv2.line(v_img, tuple(map(int, sl2[0])), tuple(map(int, sl2[1])), (255, 0, 0), 2)

        # Step 4
        tp_top_y = int(sl2[2] * tp_x + sl2[3])
        tp_h = tp_y - tp_top_y
        cv2.circle(v_img, (tp_x, tp_top_y), 4, (0, 255, 0), 4)

        # Visualize the result
        self.drawHumanFigure(tp_x, tp_y, tp_h, (255, 0, 255))
        cv2.imshow('Human Scale', v_img)
        cv2.waitKey(0)

        return tp_h


vp = (597, 741)
rp = (420, 900)
tp = (620, 1250)
reference_height = 150

img_path = "imgs_input/karuizawa.jpg"
img = cv2.imread(img_path)
sd = ScaleDetector(img, vp, rp, reference_height)
sd.drawHumanFigure()
sd.drawVanishingPoint()
sd.drawTargetPoint(tp)
sd.calculateFigureHeight(tp)
cv2.imshow('Human Scale', img)
cv2.waitKey(0)


# positions = []
# while len(positions) < 3:
#     y = random.randint(0, height)
#     x = random.randint(0, width)
#     if msk[y][x] == 255:
#         positions.append((y, x))
#         cv2.circle(img, (x, y), 10, (255, 255, 0), 2)
#
# cv2.imshow('Randomly generated location', img)
# cv2.waitKey(0)
#
# src_pts = np.array([[0.0, 0.0], [1108.0, 0.0], [0, 1477], ], dtype=np.float32)
# dst_pts = np.array([[30.0, 30.0], [1078.0, 30.0], [30.0, 1447.0]], dtype=np.float32)
#
# img_mark = img.copy()
# for pt in src_pts:
#     print(pt)
#     cv2.drawMarker(img_mark, tuple(map(int, pt)), (0, 255, 0), thickness=4)
# cv2.imshow('Mark affine anker', img)
# cv2.waitKey(0)
#
# mat = cv2.getAffineTransform(src_pts, dst_pts)
# affine_img = cv2.warpAffine(img, mat, (w, h))
# cv2.imshow('Affine transfrom', affine_img)
# cv2.waitKey(0)
