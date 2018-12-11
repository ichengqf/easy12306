#! env python
# coding: utf-8
# 功能：对图像进行预处理，将文字部分单独提取出来
# 并存放到ocr目录下
# 文件名为原验证码文件的文件名
import os

import cv2
import numpy as np


def get_text(img):
    # 得到图像中的文本部分
    return img[3:22, 120:177]


def get_imgs(img):
    interval = 5
    length = 67
    imgs = []
    for x in range(40, img.shape[0] - length, interval + length):
        for y in range(interval, img.shape[1] - length, interval + length):
            imgs.append(cv2.resize(img[x:x + length, y:y + length], (28, 28)))
    return imgs


def pretreat(path):
    texts, imgs = [], []
    for img in os.listdir(path):
        img = os.path.join(path, img)
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        texts.append(get_text(img))
        imgs.append(get_imgs(img))
    return texts, imgs


def load_data(path='data.npz'):
    if not os.path.isfile(path):
        texts, imgs = pretreat('imgs')
        np.savez(path, texts=texts, imgs=imgs)
    f = np.load(path)
    return f['texts'], f['imgs']


if __name__ == '__main__':
    texts, imgs = load_data()
    print(texts.shape)
    print(imgs.shape)
