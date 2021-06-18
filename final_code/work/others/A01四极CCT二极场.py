""""
四极 CCT 二级场
"""
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
sys.path.append(path.dirname(path.dirname(
    path.abspath(path.dirname(__file__)))))

from cctpy import *


if __name__ == '__main__':
    BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main()

    agcct3_winding_number = 25
    agcct4_winding_number = 40
    agcct5_winding_number = 34

    gantry1 = HUST_SC_GANTRY(
        qs3_gradient=5.546,
        qs3_second_gradient=-57.646,
        dicct345_tilt_angles=[30, 87.426, 92.151, 91.668],
        agcct345_tilt_angles=[94.503, 30, 72.425,	82.442],
        dicct345_current=0,
        agcct345_current=-5642.488,
        agcct3_winding_number=agcct3_winding_number,
        agcct4_winding_number=agcct4_winding_number,
        agcct5_winding_number=agcct5_winding_number,
        agcct3_bending_angle=-67.5*(agcct3_winding_number)/(
            agcct3_winding_number+agcct4_winding_number+agcct5_winding_number),
        agcct4_bending_angle=-67.5*(agcct4_winding_number)/(
            agcct3_winding_number+agcct4_winding_number+agcct5_winding_number),
        agcct5_bending_angle=-67.5*(agcct5_winding_number)/(
            agcct3_winding_number+agcct4_winding_number+agcct5_winding_number),

        DL1=0.9007765,
        GAP1=0.4301517,
        GAP2=0.370816,
        qs1_length=0.2340128,
        qs1_aperture_radius=60 * MM,
        qs1_gradient=0.0,
        qs1_second_gradient=0.0,
        qs2_length=0.200139,
        qs2_aperture_radius=60 * MM,
        qs2_gradient=0.0,
        qs2_second_gradient=0.0,

        DL2=2.35011,
        GAP3=0.43188,
        qs3_length=0.24379,

        agcct345_inner_small_r=83 * MM,
        agcct345_outer_small_r=98 * MM,  # 83+15
        dicct345_inner_small_r=114 * MM,  # 83+30+1
        dicct345_outer_small_r=130 * MM,  # 83+45 +2
    )

    gantry2 = HUST_SC_GANTRY(
        qs3_gradient=5.546,
        qs3_second_gradient=-57.646,
        dicct345_tilt_angles=[30, 87.426, 92.151, 91.668],
        agcct345_tilt_angles=[94.503, 30, 72.425,	82.442],
        dicct345_current=9445.242,
        agcct345_current=0,
        agcct3_winding_number=agcct3_winding_number,
        agcct4_winding_number=agcct4_winding_number,
        agcct5_winding_number=agcct5_winding_number,
        agcct3_bending_angle=-67.5*(agcct3_winding_number)/(
            agcct3_winding_number+agcct4_winding_number+agcct5_winding_number),
        agcct4_bending_angle=-67.5*(agcct4_winding_number)/(
            agcct3_winding_number+agcct4_winding_number+agcct5_winding_number),
        agcct5_bending_angle=-67.5*(agcct5_winding_number)/(
            agcct3_winding_number+agcct4_winding_number+agcct5_winding_number),

        DL1=0.9007765,
        GAP1=0.4301517,
        GAP2=0.370816,
        qs1_length=0.2340128,
        qs1_aperture_radius=60 * MM,
        qs1_gradient=0.0,
        qs1_second_gradient=0.0,
        qs2_length=0.200139,
        qs2_aperture_radius=60 * MM,
        qs2_gradient=0.0,
        qs2_second_gradient=0.0,

        DL2=2.35011,
        GAP3=0.43188,
        qs3_length=0.24379,

        agcct345_inner_small_r=83 * MM,
        agcct345_outer_small_r=98 * MM,  # 83+15
        dicct345_inner_small_r=114 * MM,  # 83+30+1
        dicct345_outer_small_r=130 * MM,  # 83+45 +2
    )

    bl1 = gantry1.create_second_bending_part_beamline()
    bl2 = gantry2.create_second_bending_part_beamline()

    bl1_bz = bl1.magnetic_field_bz_along(step=10*MM)
    bl2_bz = bl2.magnetic_field_bz_along(step=10*MM)

    Plot2.plot_p2s(bl1_bz,describe='r-')
    Plot2.plot_p2s(bl2_bz,describe='k-')

    Plot2.ylim()
    Plot2.legend("dipole","quad")
    Plot2.info("s/m","B/T","dipolar field along trajectory")

    Plot2.show()