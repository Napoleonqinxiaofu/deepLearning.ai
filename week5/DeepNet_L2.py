# fileName: DeepNet_L2
# author: xiaofu.qin
# create at 2017/12/3
# description: DeepNet implement L2 regularization.

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

        # Optimize the initial weights by dividing a radicals of previous layer's unit numbers.
        self.weights = [np.random.randn(current_layer, prev_layer) * np.sqrt(1 / prev_layer)\
                        for prev_layer, current_layer in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(current_layer, 1) for current_layer in layers[1:]]

        self.rate = 0.2

        # regularization parameter
        self.lambd = 0.2

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
                # print(np.max(temp_z), np.min(temp_z))
                # in the output layer, we use the sigmoid function as activation function,
                # and other's activation function is relu function
                temp_a = DeepNet.tanh(temp_z) if index != (len(self.layers[1:])-1) else DeepNet.sigmoid(temp_z)

                z.append(temp_z)
                a.append(temp_a)

            # update weights and biases
            (dw, db) = self._propagation(z, a)
            self.weights = [w - self.rate * dw for w, dw in zip(self.weights, dw)]
            self.biases = [b - self.rate * db for b, db in zip(self.biases, db)]

            print("Iteration of {i} times".format(i=i))

    # implement back propagation
    def _propagation(self, z, a):
        dz = []
        dw = []
        db = []

        for l in range(len(z)):
            if l == 0:
                dz_l = a[-(l+1)] - self.labels
            else:
                dz_l = np.dot(self.weights[-l].T, dz[l-1]) * DeepNet.tanh_prime(z[-(l+1)])

            # implement the L2 regularization here
            l2 = self.lambd / (2 * self.sample_size) * self.weights[-(l+1)]
            dw_l = 1 / self.sample_size * np.dot(dz_l, a[-(l+2)].T) + l2
            db_l = 1 / self.sample_size * np.sum(dz_l, axis=1, keepdims=True)

            dz.append(dz_l)
            dw.append(dw_l)
            db.append(db_l)

        return (dw[::-1], db[::-1])

    def predict(self, dev):
        """
        predict the dev set
        :param dev:
        :return:
        """
        layers_count = len(self.layers[1:])
        a = [dev]
        for index, layer in enumerate(self.layers[1:]):
            temp_z = np.dot(self.weights[index], a[index]) + self.biases[index]
            temp_a = DeepNet.tanh(temp_z) if index != (layers_count - 1) else DeepNet.sigmoid(temp_z)

            a.append(temp_a)

        return a[-1]

    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid(z):
        # print(z)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        return DeepNet.sigmoid(z) * (1 - DeepNet.sigmoid(z))

    @staticmethod
    def tanh(z):
        pos = np.exp(z)
        nag = np.exp(-z)
        # return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        return (pos - nag) / (pos + nag)

    @staticmethod
    def tanh_prime(z):
        return 1 - DeepNet.tanh(z) ** 2

    @staticmethod
    def relu(z):
        return np.maximum(z, 0)

    @staticmethod
    def relu_prime(z):
        z = np.array(z)
        z[np.where(z >= 0)] = 1
        z[np.where(z < 0)] = 0
        return z


if __name__ == "__main__":
    # make training set number is 10, and feature of every training set is 20
    train = np.random.randn(20, 10)

    # labels
    labels = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0]).reshape(1, 10)

    test = np.random.randn(20, 1)

    # input layer is 20 neurons, hidden layer is 20 neurons, output is 1 neurons
    net = DeepNet(train, labels, (20, 100, 50, 100, 1))

    net.train(epoch=10)

    result = net.predict(test)

    print("The predict result is ", result)
