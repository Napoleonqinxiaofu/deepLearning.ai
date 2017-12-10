# fileName: one
# author: xiaofu.qin
# create at 2017/12/10
# description: This basic example of data parallelism using Pool.
from multiprocessing import Pool
import numpy as np


def square(x):
    return x ** 2


if __name__ == "__main__":
    with Pool(5) as p:
        print(p.map(square, [1, 2, 3, 4, 5]))
