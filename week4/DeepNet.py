# fileName: twoLayer
# author: xiaofu.qin
# create at 2017/11/25
# description: Multiple layer neuron DeepNetwork

import numpy as np


class DeepNet(object):

    def __init__(self, trains, labels, layers):
        """

        :param trains:
        :param labels:
        :param layers: {tuple} (first layer units, second layer units, ……, output layer units)
        """
        self.trains = trains
        self.labels = labels
        self.layers = layers
        self.sample_size = self.trains.shape[1]

        self.weights = [np.random.randn(current_layer, prev_layer)\
                        for prev_layer, current_layer in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(current_layer, 1) for current_layer in layers[1:]]

        self.rate = 0.2

    def train(self, epoch=1):
        """

        :param epoch:
        :return:
        """
        for i in range(epoch):
            z = []
            a = [self.trains]
            # Iterate the layers
            for index, layer in enumerate(self.layers[1:]):
                temp_z = np.dot(self.weights[index], a[index]) + self.biases[index]
                # in the output layer, we use the sigmoid function as activation function,
                # and other's activation function is tanh function
                temp_a = DeepNet.tanh(temp_z) if index != (len(self.layers[1:])-1) else DeepNet.sigmoid(temp_z)

                z.append(temp_z)
                a.append(temp_a)

            (dw, db) = self._propagation(z, a)
            # update weights and biases
            self.weights = [w - self.rate * dw for w, dw in zip(self.weights, dw)]
            self.biases = [b - self.rate * db for b, db in zip(self.biases, db)]

            print("Iteration of {i} times".format(i=i))

    def _propagation(self, z, a):
        dz = []
        dw = []
        db = []

        for l in range(len(z)):
            if l == 0:
                dz_l = a[-(l+1)] - self.labels
            else:
                dz_l = np.dot(self.weights[-l].T, dz[l-1]) * DeepNet.tanh_prime(z[-(l+1)])
            dw_l = 1 / self.sample_size * np.dot(dz_l, a[-(l+2)].T)
            db_l = 1 / self.sample_size * np.sum(dz_l, axis=1, keepdims=True)

            dz.append(dz_l)
            dw.append(dw_l)
            db.append(db_l)

        return dw[::-1], db[::-1]

    def predict(self, test):
        """
        predict the test set
        :param test:
        :return:
        """
        a = [test]
        for index, layer in enumerate(self.layers[1:]):
            temp_z = np.dot(self.weights[index], a[index]) + self.biases[index]
            temp_a = DeepNet.tanh(temp_z) if index != (len(self.layers[1:]) - 1) else DeepNet.sigmoid(temp_z)

            a.append(temp_a)

        return a[-1]

    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        return DeepNet.sigmoid(z) * (1 - DeepNet.sigmoid(z))

    @staticmethod
    def tanh(z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    @staticmethod
    def tanh_prime(z):
        return 1 - DeepNet.tanh(z) ** 2


if __name__ == "__main__":
    # make training set number is 10, and feature of every training set is 20
    train = np.random.randn(20, 10)

    # labels
    labels = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0]).reshape(1, 10)

    test = np.random.randn(20, 1)

    # input layer is 20 neurons, hidden layer is 10 neurons, output is 1 neurons
    net = DeepNet(train, labels, (20, 100, 10, 50, 100, 1))

    net.train(epoch=100)

    net.predict(test)
