# fileName: logistic
# author: xiaofu.qin
# create at 2017/11/25
# description:

import numpy as np


class Logistic(object):

    def __init__(self, trains, labels, tests, test_labels):
        self.trains = trains
        self.tests = tests
        self.labels = labels
        self.test_labels = test_labels
        self.weights = np.random.randn(trains.shape[0], 1)
        self.bias = np.random.randn(1, 1)
        # default learning rate, but actually I don't know how to choose a good learning rate.
        self.rate = 0.1
        self.sample_size = self.trains.shape[1]

    def train(self, epoch=50):
        """

        :param epoch: iteration times
        :return:
        """
        for i in range(epoch):
            z = np.dot(self.weights.T, self.trains) + self.bias
            a = Logistic.activation_function(z)

            dz = a - self.labels

            dw = 1 / self.sample_size * np.dot(self.trains, dz.T)
            db = 1 / self.sample_size * np.sum(dz)

            self.weights = self.weights - self.rate * dw
            self.bias = self.bias - self.rate * db
            print("iterate {i} time".format(i=i))

        # test
        z = np.dot(self.weights.T, self.trains) + self.bias
        a = Logistic.activation_function(z)
        print(a - 0.5)

    @staticmethod
    def activation_function(z):
        return 1 / (1 + np.exp(-z))


if __name__ == "__main__":
    # make training set number is 10, and feature of every training set is 20
    train = np.random.randn(20, 10)

    # labels
    labels = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0]).reshape(1, 10)

    log = Logistic(train, labels, None, None)

    log.train()
