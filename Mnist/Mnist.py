# fileName: mnist_utils
# author: xiaofu.qin
# create at 2017/11/30
# description: 读取MNIST数据集的图片和label,
# 代码来自http://blog.csdn.net/u014046170/article/details/47445919

import numpy as np
import struct


class Mnist(object):

    def __init__(self):
        pass

    @staticmethod
    def extract_images(file_path):
        with open(file_path, "rb") as fs:
            buffers = fs.read()

        head = struct.unpack_from('>IIII', buffers, 0)

        offset = struct.calcsize('>IIII')
        img_num = head[1]
        width = head[2]
        height = head[3]
        # [60000]*28*28
        bits = img_num * width * height
        bits_string = '>' + str(bits) + 'B'  # like '>47040000B'

        imgs = struct.unpack_from(bits_string, buffers, offset)

        # imgs = np.reshape(imgs, [imgNum, 1, width * height])
        imgs = np.reshape(imgs, [img_num, width, height])

        return imgs

    @staticmethod
    def extract_labels(file_path):
        with open(file_path, "rb") as fs:
            buffer = fs.read()

        index = 0

        magic, labels = struct.unpack_from('>II', buffer, index)
        index += struct.calcsize('>II')

        label_arr = [0] * labels
        # labelArr = [0] * 2000

        for x in range(labels):
            label_arr[x] = int(struct.unpack_from('>B', buffer, index)[0])
            index += struct.calcsize('>B')

        return label_arr


if __name__ == '__main__':
    Mnist.extract_images('../mnist/t10k-images.idx3-ubyte')
    Mnist.extract_labels("../mnist/t10k-labels.idx1-ubyte")
