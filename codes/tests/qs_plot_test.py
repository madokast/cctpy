import matplotlib.pyplot as plt
import numpy as np
from cctpy.qs_hard_edge_magnet import QsHardEdgeMagnet
from cctpy.abstract_classes import LocalCoordinateSystem

if __name__ == '__main__':


    qs = QsHardEdgeMagnet(0.8, 0, 0, 0.06, LocalCoordinateSystem(
        location=np.array([1., 1., 0]),
        main_direction=np.array([1., 1., 0]),
        second_direction=np.array([-1., 1., 0.])
    ))

    lcs = qs.line_and_color()

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for lc in lcs:
        x = lc[0][:, 0]
        y = lc[0][:, 1]
        z = lc[0][:, 2]
        ax.plot(x, y, z, 'r')

    ax.grid(False)
    plt.show()
