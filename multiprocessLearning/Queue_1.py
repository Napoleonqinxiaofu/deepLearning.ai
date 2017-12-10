# fileName: Queue_1
# author: xiaofu.qin
# create at 2017/12/10
# description:
from multiprocessing import Process, Queue
import time


def f1(x):
    print("child process will start, and will sleep in {s} seconds".format(s=x))
    time.sleep(x)
    print("f1 is done at: {time}".format(time=time.ctime()))


if __name__ == "__main__":
    print("parent process start at: {time}".format(time=time.ctime()))
    p1 = Process(target=f1, args=(3,))
    p2 = Process(target=f1, args=(1,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print("all action is done at {time}".format(time=time.ctime()))
