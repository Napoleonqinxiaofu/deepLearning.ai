# fileName: activation
# author: xiaofu.qin
# create at 2017/12/3
# description:
import numpy as np


def sigmoid(z):
    # print(z)
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def tanh(z):
    pos = np.exp(z)
    nag = np.exp(-z)
    return (pos - nag) / (pos + nag)


def tanh_prime(z):
    return 1 - tanh(z) ** 2


def relu(z):
    return np.maximum(z, 0)


def relu_prime(z):
    z = np.array(z)
    z[np.where(z >= 0)] = 1
    z[np.where(z < 0)] = 0
    return z


if __name__ == "__main__":
    pass
