# fileName: second
# author: xiaofu.qin
# create at 2017/12/10
# description:
from multiprocessing import Process
import numpy as np
import os


def child_process(name):
    print("current process name is {name}".format(name=name))
    print("Current process id is {id}, and parent's id is {pid}".format(id=os.getpid(), pid=os.getppid()))
    print(__name__)


if __name__ == "__main__":
    child_process("Without Pool")
    p = Process(target=child_process, args=("process with multiprocess",))
    p.start()
    p.join()

    print("all actions are done.");