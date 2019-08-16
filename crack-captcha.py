# -*- coding=utf-8 -*-
import tensorflow as tf
import os
from PIL import Image
import numpy as np
import utils
import cnn_architecture
import time
import config


def predict_captcha():
    """
    识别测试集路径的验证码
    :return:
    """
    output = cnn_architecture.crack_captcha_cnn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 获取训练后参数路径
        checkpoint = tf.train.get_checkpoint_state("model")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find CNN network Model.")
        n = 0
        test_image_files = os.listdir(config.test_data_path)
        for f in test_image_files:
            image = Image.open(os.path.join(config.test_data_path, f))
            image = np.array(image)
            image = utils.convert2gray(image)
            image = image.flatten()
            time1 = time.time()
            predict = tf.argmax(tf.reshape(output, [-1, config.MAX_CAPTCHA, config.CHAR_SET_LEN]), 2)
            text_list = sess.run(predict, feed_dict={cnn_architecture.X: [image], cnn_architecture.keep_prob: 1})
            predict_text = utils.vec2text(text_list)
            time2 = time.time()
            elapsed = time2 - time1
            print("{} predict:{} elapsed time: {} ms".format(f, predict_text, format(elapsed * 1000, '0.2f')))
            index = f.rfind(".")
            if predict_text == f[index - config.MAX_CAPTCHA:index]:
                n += 1
        print("ACC {}".format(n / (len(test_image_files) if len(test_image_files) > 0 else 1)))
        

def crack_captcha():
    """
    识别测试集路径的验证码
    :return:
    """
    test_image_files = os.listdir(config.test_data_path)
    n = 0
    for f in test_image_files:
        time1 = time.time()
        code = get_captcha(os.path.join(config.test_data_path, f))
        time2 = time.time()
        elapsed = time2 - time1
        print("{} predict:{} elapsed time: {} ms".format(f, code, format(elapsed * 1000, '0.2f')))
        if code == f[0:4]:
            n += 1
    print("ACC {}".format(n / (len(test_image_files) if len(test_image_files) > 0 else 1)))


def get_captcha(image_path):
    """
    识别测试集路径的验证码
    :return:
    """
    tf.reset_default_graph()
    image = Image.open(os.path.join(config.test_data_path, image_path))
    image = np.array(image)
    image = utils.convert2gray(image)
    image = image.flatten()
    image = np.reshape(image, (-1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 获取训练后参数路径
        checkpoint = tf.train.get_checkpoint_state("model")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:",checkpoint.model_checkpoint_path)
            graph = tf.get_default_graph()  # 加载默认图
            input_x = graph.get_tensor_by_name("input_x:0")
            out_y = graph.get_tensor_by_name("out_y:0")
            keep_prob = graph.get_tensor_by_name("keep_prob:0")
            predict = tf.argmax(tf.reshape(out_y, [-1, config.MAX_CAPTCHA, config.CHAR_SET_LEN]), 2)
            text_list = sess.run(predict, feed_dict={input_x: image, keep_prob: 1})
            predict_text = utils.vec2text(text_list)
            return predict_text
        else:
            print("Could not find CNN network Model.")
            return None


if __name__ == "__main__":
    crack_captcha()
