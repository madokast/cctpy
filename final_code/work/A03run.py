"""
CCT 建模优化代码

作者：赵润晓
日期：2021年5月21日
"""

from cctpy import *
import time
import numpy as np

# 可变参数
# 动量分散
# momentum_dispersions = [-0.05, -0.0167, 0.0167, 0.05]
momentum_dispersions = [0.0]
# 每平面、每动量分散粒子数目
particle_number_per_plane_per_dp = 12
# 每个机架（束线）粒子数目
particle_number_per_gantry = len(
    momentum_dispersions) * particle_number_per_plane_per_dp * 2


ga32 = GPU_ACCELERATOR()


def create_gantry_beamline(param=[]):
    q1 = param[0] if len(param) > 0 else 0
    q2 = param[1] if len(param) > 0 else 0
    q3 = param[2] if len(param) > 0 else 0

    qs1_g = 5.498
    qs2_g = -3.124

    qs1_s = 30.539
    qs2_s = 0.383

    dicct_tilt_1 = 84.148
    dicct_tilt_2 = 94.725
    dicct_tilt_3 = 82.377

    agcct_tilt_0 = 100.672
    agcct_tilt_2 = 72.283
    agcct_tilt_3 = 99.973

    dicct_current = -9807.602
    agcct_current = 9999.989

    agcct1_wn = 25
    agcct2_wn = 24

    qs1_gradient = qs1_g
    qs2_gradient = qs2_g
    qs1_second_gradient = qs1_s
    qs2_second_gradient = qs2_s

    qs1_aperture_radius = 60*MM
    qs2_aperture_radius = 60*MM

    dicct12_tilt_angles = [30, dicct_tilt_1, dicct_tilt_2, dicct_tilt_3]
    agcct12_tilt_angles = [agcct_tilt_0, 30, agcct_tilt_2, agcct_tilt_3]

    dicct12_current = dicct_current
    agcct12_current = agcct_current

    agcct1_winding_number = agcct1_wn
    agcct2_winding_number = agcct2_wn
    dicct12_winding_number = 42

    agcct1_bending_angle = 22.5 * (agcct1_wn / (agcct1_wn + agcct2_wn))
    agcct2_bending_angle = 22.5 * (agcct2_wn / (agcct1_wn + agcct2_wn))

    DL1 = 0.9007765
    GAP1 = 0.4301517
    GAP2 = 0.370816
    qs1_length = 0.2340128
    qs2_length = 0.200139

    DL2 = 2.35011
    GAP3 = 0.43188
    qs3_length = 0.24379
    qs3_aperture_radius = 60 * MM
    qs3_gradient = -7.3733
    qs3_second_gradient = -45.31 * 2

    agcct12_inner_small_r = 92.5 * MM - 20 * MM  # 92.5
    agcct12_outer_small_r = 108.5 * MM - 20 * MM  # 83+15
    dicct12_inner_small_r = 124.5 * MM - 20 * MM  # 83+30+1
    dicct12_outer_small_r = 140.5 * MM - 20 * MM  # 83+45 +2

    dicct345_tilt_angles = [30, 88.773,	98.139, 91.748]
    agcct345_tilt_angles = [101.792, 30, 62.677,	89.705]
    dicct345_current = 9409.261
    agcct345_current = -7107.359
    agcct3_winding_number = 25
    agcct4_winding_number = 40
    agcct5_winding_number = 34
    agcct3_bending_angle = -67.5 * (25 / (25 + 40 + 34))
    agcct4_bending_angle = -67.5 * (40 / (25 + 40 + 34))
    agcct5_bending_angle = -67.5 * (34 / (25 + 40 + 34))

    agcct345_inner_small_r = 92.5 * MM + 0.1*MM  # 92.5
    agcct345_outer_small_r = 108.5 * MM + 0.1*MM  # 83+15
    dicct345_inner_small_r = 124.5 * MM + 0.1*MM  # 83+30+1
    dicct345_outer_small_r = 140.5 * MM + 0.1*MM  # 83+45 +2

    dicct345_winding_number = 128
    part_per_winding = 120

    return (
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
        .append_drift(DL1-0.3)
        .append_qs(length=0.2,gradient=q1,second_gradient=0.0,aperture_radius=60*MM)
        .append_drift(0.1)
        # 第二段
        .append_qs(length=0.2,gradient=q2,second_gradient=0.0,aperture_radius=60*MM)
        .append_drift(0.2)
        .append_qs(length=0.2,gradient=q3,second_gradient=0.0,aperture_radius=60*MM)
        .append_drift(0.2)
        .append_drift(0.2)
        .append_drift(DL2-1)
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


total_beamline = create_gantry_beamline()
total_beamline_length = total_beamline.get_length()

# 起点理想粒子
ip_start = ParticleFactory.create_proton_along(
    trajectory=total_beamline,
    s=0.0,
    kinetic_MeV=215
)


# 终点理想粒子
ip_end = ParticleFactory.create_proton_along(
    trajectory=total_beamline,
    s=total_beamline_length,
    kinetic_MeV=215
)

# 相空间相椭圆粒子
pps = []
for dp in momentum_dispersions:
    pps.extend(PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_xxp_plane(
        xMax=3.5 * MM, xpMax=7.5 * MM, delta=dp, number=particle_number_per_plane_per_dp
    ))
    pps.extend(PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_yyp_plane(
        yMax=3.5 * MM, ypMax=7.5 * MM, delta=dp, number=particle_number_per_plane_per_dp
    ))

# 迭代次数
times = 1

# 所有的参数/变量和目标
params_and_objs = []

# 运行，载入变量


def run(params: np.ndarray):
    global times
    start_time = time.time()

    gantry_number = params.shape[0]

    print(f"机架数目{gantry_number}")

    # 创建机架
    beamlines = create_beamlines(gantry_number, params)

    print(f"制作机架用时{time.time() - start_time}")

    # 将相空间相椭圆粒子转为实际粒子
    ps = ParticleFactory.create_from_phase_space_particles(
        ip_start, ip_start.get_natural_coordinate_system(), pps
    )

    print(f"粒子总数{len(ps) * gantry_number}")

    # 核心，调用 GPU 运行
    ps_end_list_list = ga32.track_multi_particle_beamline_for_magnet_with_multi_qs(
        bls=beamlines,
        ps=ps,
        distance=total_beamline_length,
        footstep=50 * MM
    )

    # 统计器
    statistic_x = BaseUtils.Statistic()
    statistic_y = BaseUtils.Statistic()
    statistic_xp = BaseUtils.Statistic()
    statistic_yp = BaseUtils.Statistic()

    # 所有机架 所有目标
    objs: List[List[float]] = []

    # 对于每个机架
    for gid in range(gantry_number):  # ~120
        #
        ps_end_list_each_gantry: List[RunningParticle] = ps_end_list_list[gid]
        # 不知道为什么，有些粒子的速率 speed 和速度 velocity 差别巨大
        for p in ps_end_list_each_gantry:
            p.speed = p.velocity.length()

        pps_end_each_gantry: List[PhaseSpaceParticle] = PhaseSpaceParticle.create_from_running_particles(
            ip_end, ip_end.get_natural_coordinate_system(), ps_end_list_each_gantry
        )

        # 单机架目标
        obj: List[float] = []

        # 对于所有粒子
        for pid in range(0, len(pps_end_each_gantry), particle_number_per_plane_per_dp):
            # 每 12 个粒子（每平面每组动量分散）
            # 每 particle_number_per_plane_per_dp 个一组
            for pp in pps_end_each_gantry[pid:pid + particle_number_per_plane_per_dp]:
                # 统计 x 和 y
                statistic_x.add(pp.x / MM)
                statistic_y.add(pp.y / MM)  # mm
                statistic_xp.add(pp.xp / MRAD)
                statistic_yp.add(pp.yp / MRAD)  # mrad
            # 分别求束斑
            beam_size_x = (statistic_x.max() - statistic_x.min()) / 2
            beam_size_y = (statistic_y.max() - statistic_y.min()) / 2

            beam_xp = (statistic_xp.max() - statistic_xp.min()) / 2
            beam_yp = (statistic_yp.max() - statistic_yp.min()) / 2

            # 只有 x 和 y 中大的我需要
            beam_size = max(beam_size_x, beam_size_y)
            beam_p = max(beam_xp, beam_yp)

            statistic_x.clear()
            statistic_y.clear()

            obj.append(abs(beam_size-3.5))
            obj.append(abs(beam_p-7.5))

        # print(obj)
        objs.append(obj)
        # print(objs)

    objs_np = np.array(objs)

    for gid in range(gantry_number):
        param = params[gid]
        obj = objs_np[gid]
        params_and_objs.append(np.concatenate((param, obj)))

    np.savetxt(fname='./record/' + str(times) + '.txt', X=params_and_objs)

    times += 1

    print(f"用时{time.time() - start_time} s")

    return objs_np


def create_beamlines(gantry_number, params):
    return BaseUtils.submit_process_task(
        task=create_gantry_beamline,
        param_list=[[params[i]] for i in range(gantry_number)]
    )



