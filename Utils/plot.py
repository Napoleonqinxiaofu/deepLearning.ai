# fileName: plot
# author: xiaofu.qin
# create at 2017/12/4
# description: tool of drawing dynamic data

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Plot(object):

    def __init__(self, update_func, frames):
        self.x_data, self.y_data = [], []
        self.update_func = update_func
        self.frames = frames
        self.t = 0

    def draw(self):
        # fig, self.ax = plt.subplots()
        # self.line, = plt.plot([], [], 'ro')
        # fig, self.ax = plt.subplots()
        # self.line, = self.ax.plot([], [], lw=1)
        # self.ax.grid()

        fig = plt.figure()
        self.ax = plt.axes()
        self.line, = self.ax.plot([], [], lw=2)

        self.ani_ref = FuncAnimation(fig, self._update, frames=self.frames, blit=True,
                                     interval=1, init_func=self._animation_init)
        plt.show()

    def _animation_init(self):
        self.line.set_data(self.x_data, self.y_data)
        return self.line

    def _update(self, i):

        self.x_data, self.y_data = self.update_func(self.x_data, self.y_data)

        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()

        if np.max(self.x_data) >= x_max:
            x_max = np.max(self.x_data) + 10
        if np.min(self.x_data) <= x_min:
            x_min = np.min(self.x_data) - 10

        if np.max(self.y_data) >= y_max:
            y_max = np.max(self.y_data) + 10
        if np.min(self.y_data) <= y_min:
            y_min = np.min(self.y_data) - 10

        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.figure.canvas.draw()

        self.line.set_data(self.x_data, self.y_data)

        return self.line


if __name__ == "__main__":
    def generate_data():
        t = 0
        while t < 100:
            t += 0.1
            yield t, np.sin(2 * np.pi * t)


    def update(x_data, y_data):
        x, y = x_data[-1], np.sin(2 * np.pi * (x_data[-1] + 0.1))
        x_data.append(x)
        y_data.append(y)
        return x_data, y_data


    p = Plot(update_func=update, frames=100)
    p.draw()
