""""
小论文 敏感性分析

2021年6月17日
电流误差 1 A，暂时不需要分析了
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

    floatings:List[float] = BaseUtils.linspace(-0.02,0.02,6)

    bls = []

    for floating in floatings:

        gantry = HUST_SC_GANTRY(
            qs3_gradient=5.546,
            qs3_second_gradient=-57.646,
            dicct345_tilt_angles=[30, 87.426, 92.151, 91.668],
            agcct345_tilt_angles=[94.503, 30, 72.425,	82.442],
            dicct345_current=9445.242 * (floating+1),
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

        bls.append(bl)

    # beamline_phase_ellipse_multi_delta(
    #     bl, 8, [-0.05, 0.0, 0.05], foot_step=10*MM, report=False
    # )

    ga = GPU_ACCELERATOR(
        # float_number_type=GPU_ACCELERATOR.FLOAT64,
        # block_dim_x=256
    )


    if True:
        ds = BaseUtils.linspace(-0.05,0.05,11)

        ret:List[List[float]] = []

        for d in ds:

            results = ga.track_phase_ellipse_in_multi_beamline(
                beamlines=bls,
                x_sigma_mm=3.5,xp_sigma_mrad=7.5,
                y_sigma_mm=3.5,yp_sigma_mrad=7.5,
                delta=d,
                particle_number=16,kinetic_MeV=215,
                footstep=10*MM
            )

            results_info = BaseUtils.combine(floatings,results)

            for i in range(len(results_info)):
                result_info = results_info[i]
                floating = result_info[0]
                result = result_info[1]
                xs = result[0]
                ys = result[1]

                x_width = BaseUtils.Statistic().add_all(P2.extract_x(xs)).half_width()
                y_width = BaseUtils.Statistic().add_all(P2.extract_x(ys)).half_width()

                ret.append(
                    [d, floating, x_width, y_width]
                )
            
            print(timer.period())
        

        refrom = []
        for i in range(0,len(ret),len(floatings)):
            appending = [
                ret[i][0],
                ret[i][1],
                ret[i][2],
                ret[i][3],
            ]

            for j in range(1,len(floatings)):
                appending = appending + [              
                    ret[i+j][1],
                    ret[i+j][2],
                    ret[i+j][3],
                ]

            refrom.append(appending)        

        for r in refrom:
            print(*r,sep=' ')

        
    
    print("----------------------------")