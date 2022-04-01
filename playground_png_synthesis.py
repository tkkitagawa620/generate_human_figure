import cv2
from datetime import datetime

from cv2 import imshow

BG_PATH = "./imgs_input/karuizawa.jpg"   # 背景画像
ALPHA_PATH = "./imgs_figures/walking_1.png"  # 合成アルファ画像
ALPHA_SCALE = 1.0

# メイン関数


def main():
    add_img = load_alphaImage(ALPHA_PATH, ALPHA_SCALE)
    bg_img = cv2.imread(BG_PATH)

    bg_img = merge_images(bg_img, add_img, 0, 0)  # 座標を指定してアルファ画像を合成
    save_image(bg_img)  # 画像を保存

# アルファ画像を読み込む関数


def load_alphaImage(path, scale):
    add_img = cv2.imread(path, -1)  # アルファチャンネルで読み込み
    add_img = img_resize(add_img, scale)  # リサイズ
    return add_img

# 画像をリサイズする関数


def img_resize(img, scale):
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

# 画像を保存


def save_image(img):
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = "./result/" + date + ".png"
    cv2.imwrite(path, img)  # ファイル保存
    cv2.imshow('result', img)
    cv2.waitKey(0)

# 画像を合成する関数(s_xは画像を貼り付けるx座標、s_yは画像を貼り付けるy座標)


def merge_images(bg, fg_alpha, s_x, s_y):
    print(fg_alpha[0])
    alpha = fg_alpha[:, :, 3]  # アルファチャンネルだけ抜き出す(要は2値のマスク画像)

    alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)  # grayをBGRに
    alpha = alpha / 255.0    # 0.0〜1.0の値に変換
    fg = fg_alpha[:, :, :3]

    f_h, f_w, _ = fg.shape  # アルファ画像の高さと幅を取得
    b_h, b_w, _ = bg.shape  # 背景画像の高さを幅を取得

    # 画像の大きさと開始座標を表示
    print("f_w:{} f_h:{} b_w:{} b_h:{} s({}, {})".format(f_w, f_h, b_w, b_h, s_x, s_y))

    bg[s_y:f_h + s_y, s_x:f_w + s_x] = (bg[s_y:f_h + s_y, s_x:f_w + s_x] * (1.0 - alpha)).astype('uint8')  # アルファ以外の部分を黒で合成
    bg[s_y:f_h + s_y, s_x:f_w + s_x] = (bg[s_y:f_h + s_y, s_x:f_w + s_x] + (fg * alpha)).astype('uint8')  # 合成

    return bg


if __name__ == '__main__':
    main()
