from cctpy import *
from hust_sc_gantry import *
from optimization.run_first_bending_part import create_beamline

param = [5.498,	-3.124, 	30.539, 	0.383,
         84.148, 	94.725,	82.377,
         100.672,	72.283 	, 99.973,
         -9807.602,	9999.989 	, 25.000,	24.000
         ]


def beamline_phase_ellipse_multi_delta(bl: Beamline, particle_number: int,
                                       dps: List[float], describles: str = ['r-', 'y-', 'b-', 'k-', 'g-', 'c-', 'm-'],
                                       foot_step: float = 20*MM, report: bool = True):
    if len(dps) > len(describles):
        print(
            f'describles(size={len(describles)}) 长度应大于等于 dps(size={len(dps)})')
    xs = []
    ys = []
    for dp in dps:
        x, y = bl.track_phase_ellipse(
            x_sigma_mm=3.5, xp_sigma_mrad=7.5,
            y_sigma_mm=3.5, yp_sigma_mrad=7.5,
            delta=dp, particle_number=particle_number,
            kinetic_MeV=215, concurrency_level=16,
            footstep=foot_step,
            report=report
        )
        xs.append(x + [x[0]])
        ys.append(y + [y[0]])

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
    bl = create_beamline(param)
    print(bl.point_at_end())

    beamline_phase_ellipse_multi_delta(
        bl,5,[-0.05,0,0.05]
    )
