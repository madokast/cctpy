"""
CCT 建模优化代码
前偏转段 Beamline 设计

作者：赵润晓
日期：2021年5月3日
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

DL1=0.9007765
GAP1=0.4301517
GAP2=0.370816
# qs 磁铁
qs1_length=0.2340128
qs1_aperture_radius=60 * MM
qs1_gradient=5.67
qs1_second_gradient=-127.78
qs2_length=0.200139
qs2_aperture_radius=60 * MM
qs2_gradient=12.83
qs2_second_gradient=72.22
# cct 偏转半径
cct12_big_r=0.95
# cct 孔径
agcct12_inner_small_r=72.5 * MM
agcct12_outer_small_r=88.5 * MM
dicct12_inner_small_r=104.5 * MM
dicct12_outer_small_r=120.5 * MM
# cct 匝数1
agcct1_winding_number=22
agcct2_winding_number=23
dicct12_winding_number=42
# cct 角度
dicct12_bending_angle=22.5
agcct1_bending_angle=11
agcct2_bending_angle=11.5
# cct 倾斜角（倾角 90 度表示不倾斜）
dicct12_tilt_angles=[30,80]
agcct12_tilt_angles=[90,30]
# cct 电流
dicct12_current=-6192
agcct12_current=-3319


# 匹配结果
if True:
    param = [-7.306812648,	0.550519664,	96.27929383,	-28.79607736,107.9136697,	64.40246184,	83.98949224,	105.4492682,64.2875468,	84.84352535,	766.2625325,	3965.557502,	21,	25]
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




first_bending_part_beamline: Beamline = (
    Beamline.set_start_point(P2.origin())  # 设置束线的起点
    # 设置束线中第一个漂移段（束线必须以漂移段开始）
    .first_drift(direct=P2.x_direct(), length=DL1)
    .append_agcct(  # 尾接 acgcct
        big_r=cct12_big_r,  # 偏转半径
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
        disperse_number_per_winding=120  # 每匝分段数目
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
        big_r=cct12_big_r,
        small_rs=[dicct12_outer_small_r, dicct12_inner_small_r,
                    agcct12_outer_small_r, agcct12_inner_small_r],
        bending_angles=[agcct2_bending_angle,
                        agcct1_bending_angle],
        tilt_angles=[dicct12_tilt_angles,
                        agcct12_tilt_angles],
        winding_numbers=[[dicct12_winding_number], [
            agcct2_winding_number, agcct1_winding_number]],
        currents=[dicct12_current, agcct12_current],
        disperse_number_per_winding=120
    )
    .append_drift(DL1)
)


# Plot2.plot(first_bending_part_beamline)
# Plot2.info(
#     x_label="x/m",
#     y_label="x/m",
#     title="first beading part",
#     font_size=32
# )
# Plot2.equal()
# Plot2.show()

# bz = first_bending_part_beamline.magnetic_field_bz_along(step=10*MM)
# Plot2.plot(bz)
# Plot2.info(
#     x_label="s/m",
#     y_label="field/T",
#     title="field along first beading part",
#     font_size=32
# )
# Plot2.show()

# gfa = first_bending_part_beamline.graident_field_along()
# Plot2.plot(gfa)
# Plot2.info(
#     x_label="s/m",
#     y_label="field/T",
#     title="graident field along first beading part",
#     font_size=32
# )
# Plot2.show()

if __name__=="__main__":
    BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main()
    # x,y = first_bending_part_beamline.track_phase_ellipse(
    #     x_sigma_mm=3.5,xp_sigma_mrad=7.5,
    #     y_sigma_mm=3.5,yp_sigma_mrad=7.5,
    #     delta=0.0,particle_number=16,kinetic_MeV=215,
    #     footstep=20*MM,concurrency_level=16
    # )

    # Plot2.plot(x)
    # Plot2.equal()
    # Plot2.show()

    pps = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_xxp_plane(
        xMax=3.5*MM,xpMax=7.5*MRAD,delta=0.0,number=16
    )

    ip = ParticleFactory.create_proton_along(first_bending_part_beamline,kinetic_MeV=215)
    ip_end = ParticleFactory.create_proton_along(first_bending_part_beamline,kinetic_MeV=215,s=first_bending_part_beamline.get_length())

    ps = ParticleFactory.create_from_phase_space_particles(
        ip,ip.get_natural_coordinate_system(),pps
    )

    # ParticleRunner.run_only(ps,first_bending_part_beamline,length=first_bending_part_beamline.get_length(),footstep=20*MM,concurrency_level=16)

    GPU_ACCELERATOR(float_number_type=GPU_ACCELERATOR.FLOAT64,block_dim_x=512).track_multi_particle_for_magnet_with_multi_qs(
        bl=first_bending_part_beamline,ps=ps,distance=first_bending_part_beamline.get_length(),footstep=20*MM
    )

    pps = PhaseSpaceParticle.create_from_running_particles(ip_end,ip_end.get_natural_coordinate_system(),ps)

    x = PhaseSpaceParticle.phase_space_particles_project_to_xxp_plane(pps,True)

    Plot2.plot_p2s(x,'r.')
    Plot2.equal()
    Plot2.show()

