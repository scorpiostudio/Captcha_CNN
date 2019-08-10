# -*- coding=utf-8 -*-
import numpy as np
import random
from PIL import Image
import config
import os


def convert2gray(img):
    """
    将图像转为灰度图
    :param img: 输入参数，图像的numpy.array对象
    :return:
    """
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        return gray
    else:
        return img


def get_name_and_image(data_path):
    """
    随机获取路径下的图像名称和图像数据
    :param data_path:
    :return:
    """
    all_image = os.listdir(data_path)
    random_file = random.randint(0, len(all_image) - 1)
    name = all_image[random_file][0:config.MAX_CAPTCHA]
    image = Image.open(os.path.join(data_path, all_image[random_file]))
    image = np.array(image)
    image = convert2gray(image)
    return name, image


def text2vec(text):
    """
    将图像的名称转化为一阶张量
    :param text:
    :return:
    """
    text_len = len(text)
    if text_len > config.MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')
    vector = np.zeros(config.MAX_CAPTCHA * config.CHAR_SET_LEN)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * config.CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


def vec2text(vec):
    """
    将向量转化为对应的文本
    :param vec:
    :return:
    """
    char_pos = vec[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % config.CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


def get_batch(data_path, size=config.batch_size):
    """
    获取一批数据
    :param data_path: 获取数据的路径，通常为训练集、验证集目录
    :param size: 每个批次处理的样本数量
    :return:
    """
    batch_x = np.zeros([size, config.IMAGE_HEIGHT * config.IMAGE_WIDTH])
    batch_y = np.zeros([size, config.MAX_CAPTCHA * config.CHAR_SET_LEN])

    for i in range(size):
        name, image = get_name_and_image(data_path)
        batch_x[i, :] = 1 * (image.flatten())
        batch_y[i, :] = text2vec(name)
    return batch_x, batch_y
