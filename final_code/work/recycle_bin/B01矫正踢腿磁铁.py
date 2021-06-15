"""
2021年5月24日

矫正踢腿磁铁设计

现在的情况是这样的：
前偏转段优化后，不同动量分散下的相椭圆形状、Δx、Δy、Δxp都可以，唯独 Δxp 会变动，大约是 4mr/%
现在打算加入矫正踢腿磁铁

先看看没有动量分散下，全段情况
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from work.A01run import *
from cctpy import *


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
    plt.legend(['dp'+str(int(dp*1000)/10) for dp in dps])
    plt.axis("equal")

    plt.subplot(122)
    for i in range(len(dps)):
        plt.plot(*P2.extract(ys[i]), describles[i])
    plt.xlabel(xlabel='y/mm')
    plt.ylabel(ylabel='yp/mr')
    plt.title(label='y-plane')
    plt.legend(['dp'+str(int(dp*1000)/10) for dp in dps])
    plt.axis("equal")

    plt.show()



if __name__ == "__main__":
    BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main()

    param = [5.498,	-3.124, 	30.539, 	0.383,
             84.148, 	94.725,	82.377,
             100.672,	72.283 	, 99.973,
             -9807.602,	9999.989 	, 25.000,	24.000
             ]

    qs1_g = param[0]
    qs2_g = param[1]

    qs1_s = param[2]
    qs2_s = param[3]

    dicct_tilt_1 = param[4]
    dicct_tilt_2 = param[5]
    dicct_tilt_3 = param[6]

    agcct_tilt_0 = param[7]
    agcct_tilt_2 = param[8]
    agcct_tilt_3 = param[9]

    dicct_current = param[10]
    agcct_current = param[11]

    agcct1_wn = int(param[12])
    agcct2_wn = int(param[13])


    qs1_gradient=qs1_g
    qs2_gradient=qs2_g
    qs1_second_gradient=qs1_s
    qs2_second_gradient=qs2_s

    qs1_aperture_radius=60*MM
    qs2_aperture_radius=60*MM

    dicct12_tilt_angles=[30, dicct_tilt_1, dicct_tilt_2, dicct_tilt_3]
    agcct12_tilt_angles=[agcct_tilt_0, 30, agcct_tilt_2, agcct_tilt_3]

    dicct12_current=dicct_current
    agcct12_current=agcct_current

    agcct1_winding_number=agcct1_wn
    agcct2_winding_number=agcct2_wn
    dicct12_winding_number=42


    agcct1_bending_angle=22.5 * (agcct1_wn / (agcct1_wn + agcct2_wn))
    agcct2_bending_angle=22.5 * (agcct2_wn / (agcct1_wn + agcct2_wn))

    DL1=0.9007765
    GAP1=0.4301517
    GAP2=0.370816
    qs1_length=0.2340128
    qs2_length=0.200139

    DL2=2.35011
    GAP3=0.43188
    qs3_length=0.24379
    qs3_aperture_radius=60 * MM
    qs3_gradient=-7.3733
    qs3_second_gradient=-45.31 * 2

    agcct12_inner_small_r=92.5 * MM - 20 * MM  # 92.5
    agcct12_outer_small_r=108.5 * MM - 20 * MM  # 83+15
    dicct12_inner_small_r=124.5 * MM - 20 * MM  # 83+30+1
    dicct12_outer_small_r=140.5 * MM - 20 * MM  # 83+45 +2

    dicct345_tilt_angles=[30, 88.773,	98.139, 91.748]
    agcct345_tilt_angles=[101.792, 30, 62.677,	89.705]
    dicct345_current=9409.261
    agcct345_current=-7107.359
    agcct3_winding_number=25
    agcct4_winding_number=40
    agcct5_winding_number=34
    agcct3_bending_angle=-67.5 * (25 / (25 + 40 + 34))
    agcct4_bending_angle=-67.5 * (40 / (25 + 40 + 34))
    agcct5_bending_angle=-67.5 * (34 / (25 + 40 + 34))

    agcct345_inner_small_r=92.5 * MM + 0.1*MM  # 92.5
    agcct345_outer_small_r=108.5 * MM + 0.1*MM  # 83+15
    dicct345_inner_small_r=124.5 * MM + 0.1*MM  # 83+30+1
    dicct345_outer_small_r=140.5 * MM + 0.1*MM  # 83+45 +2

    dicct345_winding_number=128
    part_per_winding=120

    bl = Beamline = (
                Beamline.set_start_point(P2.origin())  # 设置束线的起点
                # 设置束线中第一个漂移段（束线必须以漂移段开始）
                .first_drift(direct=P2.x_direct(), length=DL1)
                .append_agcct(  # 尾接 acgcct
                    big_r=0.95,  # 偏转半径
                    # 二极 CCT 和四极 CCT 孔径
                    small_rs=[dicct12_outer_small_r, dicct12_inner_small_r,
                              agcct12_outer_small_r, agcct12_inner_small_r],
                    bending_angles=[agcct1_bending_angle,
                                    agcct2_bending_angle],  # agcct 每段偏转角度
                    tilt_angles=[dicct12_tilt_angles,
                                 agcct12_tilt_angles],  # 二极 CCT 和四极 CCT 倾斜角
                    winding_numbers=[[dicct12_winding_number], [
                        agcct1_winding_number, agcct2_winding_number]],  # 二极 CCT 和四极 CCT 匝数
                    # 二极 CCT 和四极 CCT 电流
                    currents=[dicct12_current, agcct12_current],
                    disperse_number_per_winding=part_per_winding  # 每匝分段数目
                )
                .append_drift(GAP1)  # 尾接漂移段
                .append_qs(  # 尾接 QS 磁铁
                    length=qs1_length,
                    gradient=qs1_gradient,
                    second_gradient=qs1_second_gradient,
                    aperture_radius=qs1_aperture_radius
                )
                .append_drift(GAP2)
                .append_qs(
                    length=qs2_length,
                    gradient=qs2_gradient,
                    second_gradient=qs2_second_gradient,
                    aperture_radius=qs2_aperture_radius
                )
                .append_drift(GAP2)
                .append_qs(
                    length=qs1_length,
                    gradient=qs1_gradient,
                    second_gradient=qs1_second_gradient,
                    aperture_radius=qs1_aperture_radius
                )
                .append_drift(GAP1)
                .append_agcct(
                    big_r=0.95,
                    small_rs=[dicct12_outer_small_r, dicct12_inner_small_r,
                              agcct12_outer_small_r, agcct12_inner_small_r],
                    bending_angles=[agcct2_bending_angle,
                                    agcct1_bending_angle],
                    tilt_angles=[dicct12_tilt_angles,
                                 agcct12_tilt_angles],
                    winding_numbers=[[dicct12_winding_number], [
                        agcct2_winding_number, agcct1_winding_number]],
                    currents=[dicct12_current, agcct12_current],
                    disperse_number_per_winding=part_per_winding
                )
                .append_drift(DL1-0.1)
                .append_straight_dipole_magnet(
                    magnetic_field=-0.1,
                    length=0.2,
                    aperture_radius=60*MM
                )
                # 第二段
                .append_drift(DL2-0.1)
                .append_agcct(
                    big_r=0.95,
                    small_rs=[dicct345_outer_small_r, dicct345_inner_small_r,
                              agcct345_outer_small_r, agcct345_inner_small_r],
                    bending_angles=[agcct3_bending_angle,
                                    agcct4_bending_angle, agcct5_bending_angle],
                    tilt_angles=[dicct345_tilt_angles,
                                 agcct345_tilt_angles],
                    winding_numbers=[[dicct345_winding_number], [
                        agcct3_winding_number, agcct4_winding_number, agcct5_winding_number]],
                    currents=[dicct345_current, agcct345_current],
                    disperse_number_per_winding=part_per_winding
                )
                .append_drift(GAP3)
                .append_qs(
                    length=qs3_length,
                    gradient=qs3_gradient,
                    second_gradient=qs3_second_gradient,
                    aperture_radius=qs3_aperture_radius
                )
                .append_drift(GAP3)
                .append_agcct(
                    big_r=0.95,
                    small_rs=[dicct345_outer_small_r, dicct345_inner_small_r,
                              agcct345_outer_small_r, agcct345_inner_small_r],
                    bending_angles=[agcct5_bending_angle,
                                    agcct4_bending_angle, agcct3_bending_angle],
                    tilt_angles=[dicct345_tilt_angles,
                                 agcct345_tilt_angles],
                    winding_numbers=[[dicct345_winding_number], [
                        agcct5_winding_number, agcct4_winding_number, agcct3_winding_number]],
                    currents=[dicct345_current, agcct345_current],
                    disperse_number_per_winding=part_per_winding
                )
                .append_drift(DL2)
            )

    

    # beamline_phase_ellipse_multi_delta(
    #     bl,5,[-0.005,0,0.005]
    # )

    # Plot2.plot_beamline(bl)
    # Plot2.show()
    
    # kms = [200,205,210,215,220,225]
    kms = [200]
    cs = ['r-', 'y-', 'b-', 'k-', 'g-', 'c-', 'm-']
    cs = ['r-']
    for i in range(len(kms)):
        ip = ParticleFactory.create_proton_along(bl,0,kms[i])
        tps = ParticleRunner.run_get_all_info(ip,bl,bl.get_length(),10*MM)
        x = []
        y = []
        for p in tps:
            ipc = ParticleFactory.create_proton_along(bl,s=p.distance,kinetic_MeV=kms[i])
            pp = PhaseSpaceParticle.create_from_running_particle(
                ideal_particle=ipc,
                coordinate_system=ipc.get_natural_coordinate_system(),
                running_particle=p
            )
            x.append(P2(p.distance,pp.x))
            y.append(P2(p.distance,pp.y))
        
        Plot2.plot_p2s(x,describe=cs[i])
    Plot2.legend(*[str(s) for s in kms])
    Plot2.show()

