# -*- coding=utf-8 -*-
import tensorflow as tf
import config
import utils
import cnn_architecture


# 训练
def train_crack_captcha_cnn():
    """
    使用训练集路径的验证码训练CNN卷积神经网络
    :return:
    """
    output = cnn_architecture.crack_captcha_cnn()
    # 计算损失
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=cnn_architecture.Y))
    # 计算梯度
    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss)
    # 目标预测
    predict = tf.reshape(output, [-1, config.MAX_CAPTCHA, config.CHAR_SET_LEN])
    # 目标预测最大值
    max_idx_p = tf.argmax(predict, 2)
    # 真实标签最大值
    max_idx_l = tf.argmax(tf.reshape(cnn_architecture.Y, [-1, config.MAX_CAPTCHA, config.CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    # 准确率
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = utils.get_batch(config.train_data_path)
            _, loss_ = sess.run([optimizer, loss], feed_dict={cnn_architecture.X: batch_x, cnn_architecture.Y: batch_y,
                                                              cnn_architecture.keep_prob: 0.75})
            print("step {} loss = {}".format(step, loss_))

            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = utils.get_batch(config.train_data_path, 100)
                acc = sess.run(accuracy, feed_dict={cnn_architecture.X: batch_x_test, cnn_architecture.Y: batch_y_test,
                                                    cnn_architecture.keep_prob: 1.})
                print("step {} Train ACC = {}".format(step, acc))

                batch_x_test, batch_y_test = utils.get_batch(config.validation_data_path, 100)
                acc = sess.run(accuracy, feed_dict={cnn_architecture.X: batch_x_test, cnn_architecture.Y: batch_y_test,
                                                    cnn_architecture.keep_prob: 1.})
                print("step {} Validation ACC = {}".format(step, acc))
                # 如果准确率大于80%，每隔1000步保存一次模型
                if acc > 0.80 and step % 1000 == 0:
                    saver.save(sess, config.model_path, global_step=step)
                # 如果准确率大于99.99%,保存模型,完成训练
                if acc > 0.995:
                    saver.save(sess, config.model_path, global_step=step)
                    print("model path is ", config.model_path)
                    break
            # 步进
            step += 1


if __name__ == "__main__":
    train_crack_captcha_cnn()
