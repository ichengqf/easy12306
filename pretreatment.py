#! env python
# coding: utf-8
# 功能：对图像进行预处理，将文字部分单独提取出来
# 并存放到ocr目录下
# 文件名为原验证码文件的文件名
import hashlib
import os
import pathlib

import cv2
import numpy as np
import requests

import base64


PATH = 'imgs'


def download_image():
    # 抓取验证码
    # 存放到指定path下
    # 文件名为图像的MD5
    url = 'https://kyfw.12306.cn/passport/captcha/captcha-image64?login_site=E&module=login&rand=sjrand&1547799423572&callback=jQuery191016122564965500885_1547799417155&_=1547799417157'
    r = requests.get(url)
    fn = hashlib.md5(r.content).hexdigest()
    print(r.content)
    with open(f'{PATH}/{fn}.jpg', 'wb') as fp:
        fp.write(base64.b64decode(r.imge))


def download_images():
    pathlib.Path(PATH).mkdir(exist_ok=True)
    for idx in range(40000):
        download_image()
        print(idx)


def get_text(img):
    # 得到图像中的文本部分
    return img[3:22, 120:177]


def avhash(im):
    im = cv2.resize(im, (8, 8))
    avg = im.mean()
    _, im = cv2.threshold(im, avg, 1, cv2.THRESH_BINARY)
    im = im.reshape(-1)
    im = np.packbits(im)
    return im


def get_imgs(img):
    interval = 5
    length = 67
    imgs = []
    for x in range(40, img.shape[0] - length, interval + length):
        for y in range(interval, img.shape[1] - length, interval + length):
            imgs.append(avhash(img[x:x + length, y:y + length]))
    return imgs


def pretreat():
    if not os.path.isdir(PATH):
        download_images()
    texts, imgs = [], []
    for img in os.listdir(PATH):
        img = os.path.join(PATH, img)
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        texts.append(get_text(img))
        imgs.append(get_imgs(img))
    return texts, imgs


def load_data(path='data.npz'):
    if not os.path.isfile(path):
        texts, imgs = pretreat()
        np.savez(path, texts=texts, images=imgs)
    f = np.load(path)
    return f['texts'], f['images']


if __name__ == '__main__':
    texts, imgs = load_data()
    print(texts.shape)
    print(imgs.shape)
    imgs = imgs.reshape(-1, 8)
    print(np.unique(imgs, axis=0).shape)
