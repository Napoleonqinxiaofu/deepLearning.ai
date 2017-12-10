# fileName: Queue
# author: xiaofu.qin
# create at 2017/12/10
# description:
from multiprocessing import Process, Queue


def write(q, data):
    q.put(data)
    print("put data into queue is done")
    q.close()


def read(q):
    print(q.get())
    print(q.get())


if __name__ == "__main__":
    import os
    q = Queue()
    write_p = Process(target=write, args=(q, [1, 2, 3]))
    write_p1 = Process(target=write, args=(q, [1, 2, 3, "xiaofu"]))
    read_p = Process(target=read, args=(q,))
    write_p.start()
    write_p1.start()
    read_p.start()

    write_p.join()
    write_p1.join()
    read_p.join()
    print("all action are done")
    print(os.cpu_count())
