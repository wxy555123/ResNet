# coding: utf-8
from __future__ import division, print_function, absolute_import
import pickle
import numpy as np
import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" #切换CPU运行

import tensorflow as tf
import resnet_model
import random
from PIL import Image
from nets import nets_factory
import matplotlib.pyplot as plt
from tensorflow.contrib.slim import nets

slim = tf.contrib.slim

TFRECORD_PATH = "./images1/train.tfrecord"
MOD_DIR = "./saved_model/"
MOD_NAME = "res_net.ckpt"

_num_classes = 5
_batch_size = 64
_train_epochs = 10000
_global_step = 0
_print_per_epochs = 100
_save_per_epochs = 1000
_is_descent_lr = False
_resnet_size = 32
_data_format = "channels_last"
_height = 224
_width = 224
_depth = 3


def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [224, 224], 4)
    return batch


def per_image_standardization(image):
    image = image.astype('float32')
    image[:, :, 0] = (image[:, :, 0] - np.mean(image[:, :, 0])) / np.std(image[:, :, 0])
    image[:, :, 1] = (image[:, :, 1] - np.mean(image[:, :, 1])) / np.std(image[:, :, 1])
    image[:, :, 2] = (image[:, :, 2] - np.mean(image[:, :, 2])) / np.std(image[:, :, 2])
    return image


# 读取tfrecord
def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [224, 224, 3])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return image, label


# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.float32, [None])

# 学习率
lr = tf.Variable(0.1 * _batch_size / 128, dtype=tf.float32)

# 获取图片数据和标签
image, label = read_and_decode(TFRECORD_PATH)

# 使用shuffle_batch可以随机打乱 next_batch挨着往下取
# shuffle_batch才能实现[img,label]的同步,也即特征和label的同步,不然可能输入的特征和label不匹配
# 比如只有这样使用,才能使img和label一一对应,每次提取一个image和对应的label
# shuffle_batch返回的值就是RandomShuffleQueue.dequeue_many()的结果
# Shuffle_batch构建了一个RandomShuffleQueue，并不断地把单个的[img,label],送入队列中
image_batch, label_batch = tf.train.shuffle_batch(
    [image, label],
    batch_size=_batch_size,
    capacity=_batch_size * 4,
    allow_smaller_final_batch=True,
    min_after_dequeue=_batch_size * 2,
    num_threads=2)

# 定义网络结构
train_network_fn = nets_factory.get_network_fn(
    "resnet_v2_50",
    num_classes=_num_classes,
    weight_decay=0.0005,
    is_training=True)

with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    # inputs: a tensor of size [batch_size, height, width, channels]
    X = tf.reshape(x, [_batch_size, 224, 224, 3])
    # 数据输入网络得到输出值
    logits, end_points = train_network_fn(X)
    # 把标签转成one_hot形式
    one_hot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=_num_classes)
    # 计算loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels))
    # 优化器
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([tf.group(*update_ops)]):
        # train_op = slim.learning.create_train_op(total_loss, optimizer, global_step)
        train_op = optimizer.minimize(loss)

    # 计算准确率
    correct_prediction = tf.equal(tf.argmax(one_hot_labels, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.global_variables_initializer().run()
    #     tf.local_variables_initializer().run()
    print("LearningRate(before restore model): ", lr.eval())

    # 载入先前训练的模型
    saver = tf.train.Saver()
    if (os.path.exists(MOD_DIR + "checkpoint")):
        print("restore model from dir", MOD_DIR)
        ckpt = tf.train.get_checkpoint_state(MOD_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            _global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %d' % _global_step)
        else:
            print('No checkpoint file found')

        print("LearningRate(after restore model): ", lr.eval())

    # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    # 启动Queue Runners，此时文件名队列已经进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(_train_epochs + 1):

        # 获取一个批次是数据和标签
        b_image, b_label = sess.run([image_batch, label_batch])
        # b_image = data_augmentation(b_image)

        # 优化模型
        sess.run(train_op, feed_dict={x: b_image, y: b_label})

        if _is_descent_lr:
            # 每迭代四分之_train_epochs次降低一下学习率
            if i % int(_train_epochs / 5) == 0 and i != 0:
                sess.run(tf.assign(lr, lr * 0.5))

        # step 计数+1
        _global_step += 1
        # 每迭代10 次打印一下loss 和 accuracy
        if i % _print_per_epochs == 0:
            '''
            _logits = sess.run(logits, feed_dict={x: b_image})
            print(b_label,"---------",_logits)
            '''
            acc_, loss_ = sess.run([accuracy, loss], feed_dict={x: b_image, y: b_label})
            learning_rate_ = sess.run(lr)
            print("Iter:%d GlobalStep:%d Loss:%.3f Accuracy:%.2f LearningRate:%.4f" % (
                i, _global_step, loss_, acc_, learning_rate_))

        # 每迭代100 次保存一次模型
        if i % _save_per_epochs == 0 and i != 0:
            # 先删除旧的model
            for file in os.listdir(MOD_DIR):
                if (file.startswith("checkp") or file.startswith(MOD_NAME)):  # 找到那4个文件
                    filePath = os.path.join(MOD_DIR, file)
                    os.remove(filePath)
            # 保存新的model
            saver.save(sess, MOD_DIR + MOD_NAME, global_step=_global_step)
            print("GlobalStep: %d, save model." % (_global_step))
            # 满足设置条件，就停止训练保存模型
            if i == _train_epochs:
                break
    print("Training finished.")

    # 通知其他线程关闭
    coord.request_stop()
    # 其他所有线程关闭后，这个函数才能返回
    coord.join(threads)

tf.reset_default_graph()  # 清除当前默认图中堆栈，重置默认图，实现模型参数的多次读取