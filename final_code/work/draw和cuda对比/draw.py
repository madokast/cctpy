from os import error, path
import sys
from typing import Set
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
sys.path.append(path.dirname(path.dirname(
    path.abspath(path.dirname(__file__)))))

from cctpy import *

from work.draw和cuda对比.A04run import create_gantry_beamline,run

def beamline_phase_ellipse_multi_delta(bl: Beamline, particle_number: int,
                                       dps: List[float], describles: str = ['r-', 'y-', 'b-', 'k-', 'g-', 'c-', 'm-']):
    if len(dps) > len(describles):
        raise ValueError(
            f'describles(size={len(describles)}) 长度应大于 dps(size={len(dps)})')
    xs = []
    ys = []
    for dp in dps:
        x, y = bl.track_phase_ellipse(
            x_sigma_mm=3.5, xp_sigma_mrad=7.5,
            y_sigma_mm=3.5, yp_sigma_mrad=7.5,
            delta=dp, particle_number=particle_number,
            kinetic_MeV=215, concurrency_level=16,
            footstep=100*MM
        )
        xs.append(x)
        ys.append(y)

    plt.subplot(121)

    for i in range(len(dps)):
        plt.plot(*P2.extract(xs[i]), describles[i])
    plt.xlabel(xlabel='x/mm')
    plt.ylabel(ylabel='xp/mr')
    plt.title(label='x-plane')
    plt.legend(['dp'+str(int(dp*100)) for dp in dps])
    plt.axis("equal")

    plt.subplot(122)
    for i in range(len(dps)):
        plt.plot(*P2.extract(ys[i]), describles[i])
    plt.xlabel(xlabel='y/mm')
    plt.ylabel(ylabel='yp/mr')
    plt.title(label='y-plane')
    plt.legend(['dp'+str(int(dp*100)) for dp in dps])
    plt.axis("equal")

    plt.show()


if __name__ == '__main__':
    BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main()
    param = [
        -9.637097934233304630e-02, 
        1.117754653599041248e+01, 
        2.232343668400407921e+01, 
        3.898027532185977151e+01, 
        -1.018410794537818583e+04, 
        2.300000000000000000e+01, 
        2.300000000000000000e+01, 
        1.174200552539830245e+00, 
        5.925234508968078018e-01, 
        5.886982145475472272e-01,
        2.723161025187265105e-01, 
        1.715738297020128755e-01,
    ]

    bl = create_gantry_beamline(param)

    # print(bl.get_length())


    # beamline_phase_ellipse_multi_delta(
    #     bl, 8, [-0.05]
    # )


    run(numpy.array([param]))





    # Plot3.plot_beamline(bl,
    #                     describes=['r-', 'r-', 'r-', 'b-', 'b-', 'g-', 'g-', 'b-', 'b-', 'b-', 'r-', 'r-', 'b-', 'b-',
    #                                'g-', 'g-', 'b-', 'b-'])
    # track = bl.track_ideal_particle(
    #     kinetic_MeV=215,
    #     s=0,
    #     footstep=1 * MM
    # )
    # Plot3.plot_p3s(track, describe='k-')
    # Plot3.show()