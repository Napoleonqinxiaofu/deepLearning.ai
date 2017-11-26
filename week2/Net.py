# fileName: twoLayer
# author: xiaofu.qin
# create at 2017/11/25
# description: Two layer neuron network

import numpy as np


class Net(object):

    def __init__(self, trains, labels, layers):
        """

        :param trains:
        :param labels:
        :param layers: layer 0 1 2
        """
        self.trains = trains
        self.labels = labels
        self.layers = layers
        self.sample_size = self.trains.shape[1]
        self.feature_size = self.trains.shape[0]

        self.hidden_weights = np.random.randn(self.layers[1], self.layers[0])
        self.hidden_biases = np.random.randn(self.layers[1], 1)

        self.output_weights = np.random.randn(self.layers[2], self.layers[1])
        self.output_biases = np.random.randn(self.layers[2], 1)
        self.rate = 0.2

    def train(self, epoch=50):
        """

        :param epoch:
        :return:
        """
        for i in range(epoch):
            # Calculate the hidden layer output
            # print(self.hidden_weights.T.shape, self.trains.shape)
            z1 = np.dot(self.hidden_weights, self.trains) + self.hidden_biases
            a1 = Net.tanh(z1)

            # Calculate the output layer output
            z2 = np.dot(self.output_weights, a1) + self.output_biases
            a2 = Net.activation(z2)

            # back propagation
            dz2 = a2 - self.labels
            dw2 = 1 / self.sample_size * np.dot(dz2, a2.T)
            db2 = 1 / self.sample_size * np.sum(dz2, axis=1, keepdims=True)

            # hidden layer back propagation
            dz1 = np.dot(self.output_weights.T, dz2) * Net.tanh_prime(z1)
            dw1 = 1 / self.sample_size * np.dot(dz1, self.trains.T)
            db1 = 1 / self.sample_size * np.sum(dz1, axis=1, keepdims=True)

            # update weights and biases
            self.output_weights = self.output_weights - self.rate * dw2
            self.output_biases = self.output_biases - self.rate * db2
            self.hidden_weights = self.hidden_weights - self.rate * dw1
            self.hidden_biases = self.hidden_biases - self.rate * db1

            print("Iteration of {i} times".format(i=i))

    def predict(self, test):
        z1 = np.dot(self.hidden_weights, test) + self.hidden_biases
        a1 = Net.activation(z1)

        z2 = np.dot(self.output_weights, a1) + self.output_biases
        a2 = Net.tanh(z2)

        print(a2)

    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        return Net.sigmoid(z) * (1 - Net.sigmoid(z))

    @staticmethod
    def tanh(z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    @staticmethod
    def tanh_prime(z):
        return 1 - Net.tanh(z) ** 2


if __name__ == "__main__":
    # make training set number is 10, and feature of every training set is 20
    train = np.random.randn(20, 10)

    # labels
    labels = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0]).reshape(1, 10)

    test = np.random.randn(20, 1)

    # input layer is 20 neurons, hidden layer is 10 neurons, output is 1 neurons
    log = Net(train, labels, (20, 100, 1))

    log.train()

    log.predict(test)
