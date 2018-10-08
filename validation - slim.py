import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" #切换CPU运行

import tensorflow as tf
import numpy as np
import resnet_model
from PIL import Image
from nets import nets_factory
import matplotlib.pyplot as plt

MOD_DIR = "./saved_model/"
IMAGE_PATH = './test_image/3.jpg'
MOD_NAME = "res_net.ckpt"

_batch_size = 1
_num_classes = 5
_resnet_size = 32
_data_format = "channels_last"
_height = 224
_width = 224
_depth = 3

TFRECORD_PATH = "./images1/train.tfrecord"


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


# 获取图片数据和标签
image, label = read_and_decode(TFRECORD_PATH)

image_batch, label_batch = tf.train.shuffle_batch(
    [image, label],
    batch_size=_batch_size,
    capacity=400,
    allow_smaller_final_batch=True,
    min_after_dequeue=1 * 32,
    num_threads=1)

# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224, 3])

# 定义网络结构
test_network_fn = nets_factory.get_network_fn(
    "resnet_v2_50",
    num_classes=_num_classes,
    weight_decay=0.0005,
    is_training=False)

with tf.Session() as sess:
    X = tf.reshape(x, [_batch_size, 224, 224, 3])

    logits, endpoints = test_network_fn(X)
    # 预测值
    predict = tf.reshape(logits, [-1, 5])
    predict = tf.argmax(predict, 1)

    # 初始化
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    # saver = tf.train.import_meta_graph('resnet_model/model.ckpt-27527.meta')

    # 载入先前训练的模型
    if (os.path.exists(MOD_DIR + "checkpoint")):
        print("restore model from dir", MOD_DIR)
        ckpt = tf.train.get_checkpoint_state(MOD_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            _global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %d' % _global_step)
        else:
            print('No checkpoint file found')

    # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    # 启动队列
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(3):
        b_image, b_label = sess.run([image_batch, label_batch])
        #         # 显示图片
        #         img = Image.fromarray((b_image+0.5)*255).astype('unit8')
        #         plt.imshow(np.array(img))
        #         plt.show()

        # 打印标签
        # print('label:',b_label)
        # 预测
        # _logits = sess.run([logits], feed_dict={x:b_image})
        # print
        # print('predict:',_logits)
        # print(b_image)
        _logits, _ = sess.run([logits, endpoints], feed_dict={x: b_image})
        print(b_label, "---------")
        print(_logits)

    #         one_hot_labels = tf.one_hot(indices=tf.cast(label_batch, tf.int32), depth=5)
    #         loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels))
    #         _loss = sess.run(loss, feed_dict={x:b_image})
    #         print(_loss)

    # 通知其他线程关闭
    coord.request_stop()
    # 其他所有线程关闭之后，这一函数才能返回
    coord.join(threads)