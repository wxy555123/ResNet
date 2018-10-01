import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #切换CPU运行

import tensorflow as tf
import numpy as np
import resnet_model
from PIL import Image
import matplotlib.pyplot as plt


MOD_DIR = "./cifar10_model/"
IMAGE_PATH = 'test_image/3.jpg'
MOD_NAME = "res_net.ckpt"

_num_classes = 10
_resnet_size = 32
_data_format = "channels_last"
_height = 32
_width = 32
_depth = 3


def per_image_standardization(image):
    image = image.astype('float32')
    image[:, :, :, 0] = (image[:, :, :, 0] - np.mean(image[:, :, :, 0])) / np.std(image[:, :, :, 0])
    return  image

def IdToClassName(id):
    classDict = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer",
               5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}
    return classDict[id]

#图片转矩阵函数
def ImageToArray(imagePath, width, height):
    image = Image.open(imagePath, "r")
    out = image.resize((width, height),Image.ANTIALIAS)
    return np.array(out)   #将图片以数组的形式读入变量

image_arr = ImageToArray(IMAGE_PATH, 32, 32).astype("float32") #转矩阵
image_X = image_arr.reshape([1, 32, 32, 3]) #加一维
#image_X = per_image_standardization(image_X) #标准化


# placeholder
x = tf.placeholder(tf.float32,[None,32,32,3])

#定义网络
network = resnet_model.cifar10_resnet_v2_generator(
    _resnet_size, _num_classes, _data_format)



with tf.Session() as sess:
    X = tf.reshape(x,[1,32,32,3])

    logits = network(X, False)

    # 初始化
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    # 载入先前训练的模型
    if(os.path.exists(MOD_DIR + "checkpoint")):
        print("restore model from dir", MOD_DIR)
        ckpt = tf.train.get_checkpoint_state(MOD_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            _global_step = int( ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1] )
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %d' % _global_step)
        else:
            print('No checkpoint file found')

    # 打印logits
    prob = sess.run(logits, feed_dict={x:image_X})
    print("\n logits:\n", prob, "\n")

    dim_1_prob = np.squeeze(prob)
    sm_prob = tf.nn.softmax(dim_1_prob)
    predictions = sess.run(sm_prob)
    '''
    # 显示图片
    img = Image.open(IMAGE_PATH)
    plt.imshow(img)
    plt.show()
    '''

    # 显示得分排序
    top_k = predictions.argsort()[::-1]  # prob从大到小排列的的arg list
    for id in top_k:
        # 获取分类名称
        className = IdToClassName(id)
        # 获取该分类的概率
        score = predictions[id]
        print("%s (score = %.5f)" % (className, score))


