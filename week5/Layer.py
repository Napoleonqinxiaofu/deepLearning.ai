# fileName: Layer
# author: xiaofu.qin
# create at 2017/12/3
# description: single layer class
import numpy as np


class Layer(object):

    def __init__(self, units, activation="relu", L2=True, dropout=False):
        """

        :param units {tuple}: units is like (prev_layer_unit, current_layer_units)
        :param activation {String}: activation function name
        :param L2 {Boolean}: use L2 regularization or not.
        :param dropout {Boolean}: use dropout or not
        """
        (prev, current) = units
        # Initialize the weights and biases
        self.weights = np.random.randn(prev, current) * np.sqrt(1 / prev)

    def back_propagation(self):
        pass

    def train(self):
        pass

    def predict(self):
        return self.train()


if __name__ == "__main__":
    pass
