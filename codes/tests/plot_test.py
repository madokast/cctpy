import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    z = np.linspace(0, np.pi * 20, 360)
    x = np.sin(z)
    y = np.cos(z)

    ax.plot(x, y, z, 'r')

    ax.grid(False)
    plt.show()
