# fileName: __init__.py
# author: xiaofu.qin
# create at 2017/11/30
# description:
from Mnist.Mnist import Mnist

__all__ = [Mnist]


def hello():
    print("hi, beauty")


if __name__ == "__main__":
    labels = Mnist.extract_labels("../mnist/t10k-labels.idx1-ubyte")
    print(labels[:10])
