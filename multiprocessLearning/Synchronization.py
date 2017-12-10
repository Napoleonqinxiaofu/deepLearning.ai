# fileName: Synchronization
# author: xiaofu.qin
# create at 2017/12/10
# description: 线程间同步的方法
# For instance one can use a lock to ensure that only one process prints to standard output at a time:
from multiprocessing import Process, Lock


data = []


def f(lock, i):
    global data
    lock.acquire()
    try:
        data.append(i)
        print('hello world', data)
    finally:
        # finally we need release the lock to allow the next process can be executed.
        # But actually it is just like one process.
        lock.release()


if __name__ == '__main__':
    lock = Lock()

    for num in range(10):
        p = Process(target=f, args=(lock, num)).start()

    print(data)
