#! env python
# coding: utf-8
# 功能：对图像进行预处理，将文字部分单独提取出来
# 并存放到ocr目录下
# 文件名为原验证码文件的文件名
import os

import cv2
import numpy as np


def pretreat(path):
    imgs = []
    for img in os.listdir(path):
        img = os.path.join(path, img)
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        # 得到图像中的文本部分
        img = img[3:22, 120:177]
        imgs.append(img)
    return imgs


def load_data(path='data.npy'):
    if not os.path.isfile(path):
        imgs = pretreat('imgs')
        np.save(path, imgs)
    return np.load(path)


if __name__ == '__main__':
    imgs = load_data()
    print(imgs.shape)
    cv2.imwrite('temp.jpg', imgs[0])
