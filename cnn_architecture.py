# -*- coding=utf-8 -*-
import tensorflow as tf
import config
import math

X = tf.placeholder(tf.float32, [None, config.IMAGE_HEIGHT * config.IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, config.MAX_CAPTCHA * config.CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  # dropout


# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    """
    构建CNN卷积神经网络架构
    :param w_alpha:
    :param b_alpha:
    :return:
    """
    x = tf.reshape(X, shape=[-1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1])

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    # 卷积 + Relu激活函数
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    # 池化
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # dropout 防止过拟合
    conv1 = tf.nn.dropout(conv1, rate=1 - keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    # 卷积 + Relu激活函数
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    # 池化
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # dropout 防止过拟合
    conv2 = tf.nn.dropout(conv2, rate=1 - keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    # 卷积 + Relu激活函数
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    # 池化
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # dropout 防止过拟合
    conv3 = tf.nn.dropout(conv3, rate=1 - keep_prob)

    # Fully connected layer
    height = int(math.ceil(config.IMAGE_HEIGHT / 8))
    width = int(math.ceil(config.IMAGE_WIDTH / 8))
    w_d = tf.Variable(w_alpha * tf.random_normal([height * width * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    # 全连接 + Relu
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, rate=1 - keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, config.MAX_CAPTCHA * config.CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([config.MAX_CAPTCHA * config.CHAR_SET_LEN]))
    # 全连接
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out
