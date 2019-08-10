# -*- coding=utf-8 -*-
import shutil
import os
import config
import random


def build_data_set():
    """
    根据数据集划分训练集、验证集、测试集
    验证集、测试集、训练集占比为1：1：8
    :return:
    """
    # 检测测试集路径
    if not os.path.exists(os.path.join(config.test_data_path)):
        os.mkdir(config.test_data_path, 'w+')
    # 检测验证集路径
    if not os.path.exists(os.path.join(config.validation_data_path)):
        os.mkdir(config.test_data_path, 'w+')
    # 检测训练集路径
    if not os.path.exists(os.path.join(config.train_data_path)):
        os.mkdir(config.test_data_path, 'w+')

    files = os.listdir(config.data_path)
    n = int(len(files) / 10)
    # 建立验证集
    for i in range(n):
        f = random.choice(files)
        shutil.move(os.path.join(config.data_path, f), os.path.join(config.validation_data_path, f))
        files = os.listdir(config.data_path)

    # 建立测试集
    files = os.listdir(config.data_path)
    for i in range(n):
        f = random.choice(files)
        shutil.move(os.path.join(config.data_path, f), os.path.join(config.test_data_path, f))
        files = os.listdir(config.data_path)

    # 建立训练集
    files = os.listdir(config.data_path)
    for f in files:
        shutil.move(os.path.join(config.data_path, f), os.path.join(config.train_data_path, f))


if __name__ == "__main__":
    build_data_set()
