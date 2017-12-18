# fileName: Layer
# author: xiaofu.qin
# create at 2017/12/3
# description: single layer class
import numpy as np

from Utils.activation import sigmoid, sigmoid_prime, tanh, tanh_prime, relu, relu_prime


class Layer(object):

    def __init__(self, units, activation="relu", L2=True, dropout=1.0):
        """

        :param units {tuple}: units is like (prev_layer_unit, current_layer_units)
        :param activation {String}: activation function name
        :param L2 {Boolean}: use L2 regularization or not.
        :param dropout {number}: keep-prob number
        """
        (prev, current) = units
        # Initialize the weights and biases
        self.weights = np.random.randn(prev, current)\
                       * np.sqrt((1 / prev) if activation != "relu" else (2 / prev))
        self.biases = np.random.randn(current, 1)
        self.keep_prob = dropout

        # select different activation function in case of the function parameter activation.
        if activation == "relu":
            self.activation = relu
            self.activation_prime = relu_prime
        elif activation == "sigmoid":
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == "tanh":
            self.activation = tanh
            self.activation_prime = tanh_prime

    def back_propagation(self, ):
        pass

    def train(self, inputs):
        return np.dot(self.weights, inputs) + self.biases

    def predict(self, inputs):
        return self.train(inputs)

    def get_params(self):
        return self.weights, self.biases

    def set_params(self, params):
        (weights, biases) = params
        self.weights = weights
        self.biases = biases


if __name__ == "__main__":
    layer = Layer((10, 10))
