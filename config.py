# -*- coding:utf-8 -*-
import os


work_path = os.getcwd()
data_path = work_path + '/data'  #数据集
train_data_path = work_path + '/train_data'  #训练集
test_data_path = work_path + '/test_data'    #测试集
validation_data_path = work_path + '/validation_data'    #测试集
model_path = work_path + '/model/crack_captcha_model' #模型

# 数字字符集
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# 小写字母字符集
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# 大写字母字符集
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# 字符集
CHAR_SET = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于MAX_CAPTCHA, '_'用来补齐
# 字符集长度
CHAR_SET_LEN = len(CHAR_SET)

# 图像高度，像素值
IMAGE_HEIGHT = 26
# 图像宽度，像素值
IMAGE_WIDTH = 80
# 验证码的最长字符数
MAX_CAPTCHA = 4

# 每批次训练的数量
batch_size = 32  # size of batch
# 学习率
learning_rate = 0.001   # learning rate for adam
