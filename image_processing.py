import argparse

import cv2
import numpy as np


def size(udlr):
    u, d, l, r = udlr
    return (d-u)*(r-l)


def cos_sim(v1, v2):
    v1 = v1.reshape(-1).astype(np.float128)
    v2 = v2.reshape(-1).astype(np.float128)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def main(img_path):
    img = cv2.imread(img_path)  # ファイル読み込み
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([30, 4, 130])-30  # np.array(color)    # 抽出する色の下限(bgr)
    upper = np.array([30, 4, 255])+30    # 抽出する色の上限(bgr)
    img_mask = cv2.inRange(img_hsv, lower, upper)  # bgrからマスクを作成
    extract = cv2.bitwise_and(img, img, mask=img_mask)  # 元画像とマスクを合成
    img_gray = cv2.cvtColor(extract, cv2.COLOR_BGR2GRAY)
    n_labels, labels = cv2.connectedComponents(img_gray)
    udlrs = []
    Y, X = np.meshgrid(range(labels.shape[1]), range(labels.shape[0]))
    for i in range(n_labels):
        flag = labels == i
        x = X[flag]
        y = Y[flag]
        udlrs.append((x.min(), x.max(), y.min(), y.max()))
    ids = list(range(n_labels))
    ids.sort(key=lambda i: size(udlrs[i]), reverse=True)

    pais = ["E", "S", "N", "rd", "gd", "wd"]\
        + [f"{i+1}m" for i in range(9)]\
        + [f"{i+1}p" for i in range(9)]\
        + [f"{i+1}s" for i in range(9)]
    pai_imgs = []
    for pai in pais:
        pai_imgs.append(cv2.imread(f'./data/{pai}.png'))

    ids = ids[1:14]
    ids.sort(key=lambda i: udlrs[i][2])
    tehai = []
    for i in ids:
        u, d, l, r = udlrs[i]
        target_img = img[u:d, l:r]
        if target_img.shape[0] < target_img.shape[1]:
            target_img = target_img.transpose([1, 0, 2])
        similarity = []
        for test_img in pai_imgs:
            test_img = cv2.resize(test_img, dsize=(
                target_img.shape[1], target_img.shape[0]))
            similarity.append(cos_sim(target_img, test_img))
        tehai.append(pais[np.argmax(similarity)])
    print(tehai)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extract tehai from image')
    parser.add_argument('img_path', help='path of image')
    args = parser.parse_args()
    main(args.img_path)
