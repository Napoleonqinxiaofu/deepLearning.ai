# fileName: plot
# author: xiaofu.qin
# create at 2017/12/10
# description:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = [], []
ax.grid()
ln, = plt.plot([], [], 'ro', animated=True)


def init():
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1.1, 1.1)
    return ln,


def update(frame):
    print(frame)
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,


print(np.linspace(0, 2 * np.pi, 128))

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, 128),
                    init_func=init, blit=True)
plt.show()
