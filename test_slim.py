import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" #切换CPU运行

import tensorflow as tf
import numpy as np
import resnet_model
from PIL import Image
from nets import nets_factory
import matplotlib.pyplot as plt

MOD_DIR = "./saved_model/"
IMAGE_PATH = './test_image/5.jpg'
MOD_NAME = "res_net.ckpt"

_num_classes = 5
_resnet_size = 32
_data_format = "channels_last"
_height = 224
_width = 224
_depth = 3


# {"rock":0, "urban":1, "water":2, "wetland":3, "wood":4}
def IdToClassName(id):
    classDict = {0: "bus", 1: "dinosaur", 2: "elephant", 3: "flower", 4: "horse"}
    return classDict[id]


# 图片转矩阵函数
def ImageToArray(imagePath, width, height):
    image = Image.open(imagePath, "r")  # 读取图片文件
    out = image.resize((width, height), Image.ANTIALIAS)
    return np.array(out)  # 将图片以数组的形式读入变量


image_arr = ImageToArray(IMAGE_PATH, 224, 224).astype("float32")
image_arr = image_arr.astype('float32') * (1. / 255) - 0.5  # 归一化处理
image_X = image_arr.reshape([1, 224, 224, 3])

# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224, 3])

# 定义网络结构
test_network_fn = nets_factory.get_network_fn(
    "resnet_v2_50",
    num_classes=_num_classes,
    weight_decay=0.0005,
    is_training=False)

# saver = tf.train.Saver()
# tf.reset_default_graph()

with tf.Session() as sess:
    X = tf.reshape(x, [1, 224, 224, 3])

    logits, endpoints = test_network_fn(X)
    # 预测值
    predict = tf.reshape(logits, [-1, 5])
    predict = tf.argmax(predict, 1)

    # 初始化
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

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

    prob = sess.run(logits, feed_dict={x: image_X})
    print(prob)
    dim_1_prob = np.squeeze(prob)
    sm_prob = tf.nn.softmax(dim_1_prob)
    predictions = sess.run(sm_prob)

    # 显示图片
    img = Image.open(IMAGE_PATH)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    # 显示得分排序
    top_k = predictions.argsort()[::-1]  # prob从大到小排列的的arg list
    for id in top_k:
        # 获取分类名称
        className = IdToClassName(id)
        # 获取该分类的概率
        score = predictions[id]
        print("%s (score = %.5f)" % (className, score))
