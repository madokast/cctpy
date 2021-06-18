""""
小论文 敏感性分析
回顾
"""
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
sys.path.append(path.dirname(path.dirname(
    path.abspath(path.dirname(__file__)))))
from cctpy import *

if __name__ == '__main__':
    BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main()
    timer = BaseUtils.Timer()

    agcct3_winding_number = 25
    agcct4_winding_number = 40
    agcct5_winding_number = 34

    gantry = HUST_SC_GANTRY(
        qs3_gradient=5.546,
        qs3_second_gradient=-57.646,
        dicct345_tilt_angles=[30, 87.426, 92.151, 91.668],
        agcct345_tilt_angles=[94.503, 30, 72.425,	82.442],
        dicct345_current=9445.242,
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

    bl = gantry.create_second_bending_part_beamline()




    # beamline_phase_ellipse_multi_delta(
    #     bl, 8, [-0.05, 0.0, 0.05], foot_step=10*MM, report=False
    # )

    ga = GPU_ACCELERATOR(
        # float_number_type=GPU_ACCELERATOR.FLOAT64,
        # block_dim_x=256
    )


    if True:
        dds = [
            (d,"r-") for d in BaseUtils.linspace(-0.05,0.05,11)
        ]

        d_x:List[P2] = []
        d_y:List[P2] = []

        for d in dds:

            t = ga.track_phase_ellipse_in_multi_beamline(
                beamlines=[bl],
                x_sigma_mm=3.5,xp_sigma_mrad=7.5,
                y_sigma_mm=3.5,yp_sigma_mrad=7.5,
                delta=d[0],
                particle_number=16,kinetic_MeV=215,
                footstep=10*MM
            )[0]

            xs = t[0]
            ys = t[1]

            d_x.append(P2(
                x = d[0],
                y = BaseUtils.Statistic().add_all(P2.extract_x(xs)).half_width()
            ))

            d_y.append(P2(
                x = d[0],
                y = BaseUtils.Statistic().add_all(P2.extract_x(ys)).half_width()
            ))

            Plot2.subplot(121)
            Plot2.plot_p2s(xs,circle=True,describe=d[1])
            Plot2.subplot(122)
            Plot2.plot_p2s(ys,circle=True,describe=d[1])
            print(timer.period())

        Plot2.subplot(121)
        Plot2.info("x/mm","xp/mr","x-plane")
        Plot2.equal()

        Plot2.subplot(122)
        Plot2.info("y/mm","yp/mr","y-plane")
        Plot2.equal()


        print(timer.period())

        Plot2.show()

        print("d_x")
        for each in d_x:
            print(f"{each.x} {each.y}")

        print("d_y")
        for each in d_x:
            print(f"{each.x} {each.y}")
        
    
    print("----------------------------")