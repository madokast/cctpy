""""
小论文 敏感性分析
A03后偏转段机架敏感性分析之二极CCT径向error.py
"""
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
sys.path.append(path.dirname(path.dirname(
    path.abspath(path.dirname(__file__)))))
from cctpy import *



delta = 0.05
ids:List[int] =[1]*200
sigma = 0.5*MM

if __name__ == '__main__':
    for sigma in BaseUtils.linspace(-0.5*MM,0.5*MM,11):

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

        bl0 = gantry.create_second_bending_part_beamline()

        # 所有磁铁
        ms = bl0.get_magnets()

        # 分成 二极 cct、四极 cct 和非 cct
        diccts:List[CCT] = []
        quccts:List[CCT] = []
        other_magnets:List[Magnet] = []

        for m in ms:
            if isinstance(m,CCT):
                cct = CCT.as_cct(m)
                if BaseUtils.equal(30,abs(cct.tilt_angles[0])):
                    diccts.extend(CCT.cut_to_single_winding_cct(cct))
                elif BaseUtils.equal(30,abs(cct.tilt_angles[1])):
                    quccts.extend(CCT.cut_to_single_winding_cct(cct))
                else:
                    raise ValueError(f"无法区分CCT是二极还是四极，cct=\n{cct}")
            else:
                other_magnets.append(m)

        bls = []
        for id_ in ids:
            bl = Beamline(trajectory=bl0.get_trajectory())
            bl.magnets = []
            bl.magnets.extend(quccts)
            bl.magnets.extend(other_magnets)

            for dicct in diccts:
                bl.magnets.append(CCT.create_by_existing_cct(
                    existing_cct=dicct,
                    # ---------------------------------------------------------------------------
                    # small_r = qucct.small_r + BaseUtils.Random.gauss_limited(0,0.05*MM,0.1*MM)
                    # small_r = qucct.small_r + BaseUtils.Random.uniformly_distribution(sigma,-sigma)
                    small_r = dicct.small_r + sigma
                    # ---------------------------------------------------------------------------
                ))
            bls.append(bl)

        ga = GPU_ACCELERATOR(
            # float_number_type=GPU_ACCELERATOR.FLOAT64,
            # block_dim_x=256
        )

        results = ga.track_phase_ellipse_in_multi_beamline(
            beamlines=[bl0]+bls,
            x_sigma_mm=3.5,xp_sigma_mrad=7.5,
            y_sigma_mm=3.5,yp_sigma_mrad=7.5,
            delta=delta,
            particle_number=16,kinetic_MeV=215,
            footstep=20*MM
        )


        result0 = results[0]
        xs0,ys0 = result0
        xm,ym = 0,0
        
        # Plot2.subplot(121)
        # Plot2.plot_p2s(xs0,circle=True,describe='k-')
        # Plot2.subplot(122)
        # Plot2.plot_p2s(ys0,circle=True,describe='k-')

        for i in range(1,len(results)):
            result = results[i]
            xs,ys = result

            # Plot2.subplot(121)
            # Plot2.plot_p2s(xs,circle=True,describe='r-')
            # Plot2.subplot(122)
            # Plot2.plot_p2s(ys,circle=True,describe='r-')

            for i in range(len(ys0)):
                xm = max(xm,abs(xs[i].x-xs0[i].x))
                ym = max(ym,abs(ys[i].x-ys0[i].x))

        # Plot2.subplot(121)
        # Plot2.info("x/mm","xp/mr","x-plane")
        # Plot2.legend("0","err")
        # Plot2.equal()
        # Plot2.subplot(122)
        # Plot2.info("y/mm","yp/mr","y-plane")
        # Plot2.legend("0","err")
        # Plot2.equal()

        # Plot2.show()

        print(sigma,max(xm,ym))

