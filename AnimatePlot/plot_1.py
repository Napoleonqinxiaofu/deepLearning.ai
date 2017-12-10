# fileName: plot_1
# author: xiaofu.qin
# create at 2017/12/10
# description:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

if __name__ == "__main__":
    def data_gen(t=0):
        cnt = 0
        while cnt < 1000:
            cnt += 1
            t += 0.1
            yield t, np.sin(2 * np.pi * t) * np.exp(-t / 10.)


    def init():
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(0, 10)
        del xdata[:]
        del ydata[:]
        line.set_data(xdata, ydata)
        return line,


    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.grid()
    xdata, ydata = [], []


    def run(data):
        # update the data
        t, y = data
        xdata.append(t)
        ydata.append(y)
        xmin, xmax = ax.get_xlim()

        if t >= xmax:
            ax.set_xlim(xmin, 2 * xmax)
            ax.figure.canvas.draw()
        line.set_data(xdata, ydata)

        return line,


    # frames = zip(np.linspace(0, 2 * np.pi, 128), np.linspace(2 * np.pi, 0 * np.pi, 128))

    ani = FuncAnimation(fig, run, frames=data_gen, blit=True, interval=1, init_func=init)
    # can not save into disk due to I can not install some encoding package in current python version.
    # ani.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()
