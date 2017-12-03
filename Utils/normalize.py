# fileName: normalize
# author: xiaofu.qin
# create at 2017/12/3
# description: Normalizing data

import numpy as np


def zero_mean(X):
    """
    :param X: belong to R(n, m), which n represent the feature number, and m represent the sample size.
    :return:
    """
    (n, m) = X.shape
    mean = 1 / m * np.tile(np.sum(X, axis=1).reshape(n, 1), (1, m))
    return X - mean


def normalize_variance(X):
    (n, m) = X.shape
    sigma = X ** 2 / m
    sigma = X / sigma
    sigma[np.where(np.isnan(sigma))] = 0
    return sigma


def normalize(X):
    return zero_mean(normalize_variance(X))


if __name__ == "__main__":
    a = np.array([1, 2, 3, 4, 5, 6]).reshape(3, 2)
    b = np.array([1, 2, 3, np.nan])
    b[np.where(np.isnan(b))] = 0
    print(b)
    # print(np.tile(b, (3, 1)))
