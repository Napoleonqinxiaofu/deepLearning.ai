# fileName: pipes
# author: xiaofu.qin
# create at 2017/12/10
# description:
from multiprocessing import Process, Pipe


def write_p(conn):
    # send the data and call the close function, that is very important.
    conn.send([1, 2, 3, "xiaofu.qin"])
    conn.close()


if __name__ == "__main__":
    parent_conn, child_conn = Pipe()

    p = Process(target=write_p, args=(child_conn,))
    p.start()
    print(parent_conn.recv())
    p.join()
