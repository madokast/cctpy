"""
CCT 建模优化代码

作者：赵润晓
日期：2021年5月21日
"""

from cctpy import *
from hust_sc_gantry import HUST_SC_GANTRY
import time
import numpy as np

# 可变参数
# 动量分散
momentum_dispersions = [-0.05, -0.0167, 0.0167, 0.05]
# 每平面、每动量分散粒子数目
particle_number_per_plane_per_dp = 12
# 每个机架（束线）粒子数目
particle_number_per_gantry = len(
    momentum_dispersions) * particle_number_per_plane_per_dp * 2


ga32 = GPU_ACCELERATOR()


def create_gantry_beamline(param=[]):
    qs1_gradient = param[0] if len(param) > 0 else 0
    qs2_gradient = param[1] if len(param) > 0 else 0
    qs3_gradient = 0.0

    qs1_second_gradient = param[2] if len(param) > 0 else 0
    qs2_second_gradient = param[3] if len(param) > 0 else 0
    qs3_second_gradient = 0.0

    dicct_tilt_1 = param[4] if len(param) > 0 else 90
    dicct_tilt_2 = param[5] if len(param) > 0 else 90
    dicct_tilt_3 = param[6] if len(param) > 0 else 90

    agcct_tilt_0 = param[7] if len(param) > 0 else 90
    agcct_tilt_2 = param[8] if len(param) > 0 else 90
    agcct_tilt_3 = param[9] if len(param) > 0 else 90

    dicct12_current = param[10] if len(param) > 0 else 10000
    agcct12_current = param[11] if len(param) > 0 else 10000

    agcct1_wn = int(param[12]) if len(param) > 0 else 20
    agcct2_wn = int(param[13]) if len(param) > 0 else 20

    ####################################
    DL1 = 900.78*MM
    GAP1 = 430.15*MM
    GAP2 = 370.82*MM
    GAP3 = 431.88*MM
    qs1_length = 234.01*MM
    qs2_length = 200.14*MM
    qs3_length = 243.79*MM
    DL2 = 2350.11*MM

    qs1_aperture_radius = 60 * MM
    qs2_aperture_radius = 60 * MM
    qs3_aperture_radius = 60 * MM

    dicct12_tilt_angles = [30, dicct_tilt_1, dicct_tilt_2, dicct_tilt_3]
    agcct12_tilt_angles = [agcct_tilt_0, 30, agcct_tilt_2, agcct_tilt_3]

    agcct1_winding_number = agcct1_wn
    agcct2_winding_number = agcct2_wn
    dicct12_winding_number = 42

    agcct1_bending_angle = 22.5 * (agcct1_wn / (agcct1_wn + agcct2_wn))
    agcct2_bending_angle = 22.5 * (agcct2_wn / (agcct1_wn + agcct2_wn))

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
    part_per_winding = 60
    cct345_big_r = 0.95
    cct12_big_r = 0.95
    return HUST_SC_GANTRY(
        # 漂移段
        DL1=DL1,
        GAP1=GAP1,
        GAP2=GAP2,
        # qs 磁铁
        qs1_length=qs1_length,
        qs1_aperture_radius=qs1_aperture_radius,
        qs1_gradient=qs1_gradient,
        qs1_second_gradient=qs1_second_gradient,
        qs2_length=qs2_length,
        qs2_aperture_radius=qs2_aperture_radius,
        qs2_gradient=qs2_gradient,
        qs2_second_gradient=qs2_second_gradient,
        # cct 偏转半径
        cct12_big_r=cct12_big_r,
        # cct 孔径
        agcct12_inner_small_r=agcct12_inner_small_r,
        agcct12_outer_small_r=agcct12_outer_small_r,
        dicct12_inner_small_r=dicct12_inner_small_r,
        dicct12_outer_small_r=dicct12_outer_small_r,
        # cct 匝数1
        agcct1_winding_number=agcct1_winding_number,
        agcct2_winding_number=agcct2_winding_number,
        dicct12_winding_number=dicct12_winding_number,
        # cct 角度
        dicct12_bending_angle=22.5,
        agcct1_bending_angle=agcct1_bending_angle,
        agcct2_bending_angle=agcct2_bending_angle,
        # cct 倾斜角（倾角 90 度表示不倾斜）
        dicct12_tilt_angles=dicct12_tilt_angles,
        agcct12_tilt_angles=agcct12_tilt_angles,
        # cct 电流
        dicct12_current=dicct12_current,
        agcct12_current=agcct12_current,
        # ------------------ 后偏转段 ---------------#
        # 漂移段
        DL2=DL2,
        GAP3=GAP3,
        # qs 磁铁
        qs3_length=qs3_length,
        qs3_aperture_radius=qs3_aperture_radius,
        qs3_gradient=qs3_gradient,
        qs3_second_gradient=qs3_second_gradient,
        # cct 偏转半径
        cct345_big_r=cct345_big_r,
        # cct 孔径
        agcct345_inner_small_r=agcct345_inner_small_r,
        agcct345_outer_small_r=agcct345_outer_small_r,
        dicct345_inner_small_r=dicct345_inner_small_r,
        dicct345_outer_small_r=dicct345_outer_small_r,
        # cct 匝数
        agcct3_winding_number=agcct3_winding_number,
        agcct4_winding_number=agcct4_winding_number,
        agcct5_winding_number=agcct5_winding_number,
        dicct345_winding_number=dicct345_winding_number,
        # cct 角度（负数表示顺时针偏转）
        dicct345_bending_angle=-67.5,
        agcct3_bending_angle=agcct3_bending_angle,
        agcct4_bending_angle=agcct4_bending_angle,
        agcct5_bending_angle=agcct5_bending_angle,
        # cct 倾斜角（倾角 90 度表示不倾斜）
        dicct345_tilt_angles=dicct345_tilt_angles,
        agcct345_tilt_angles=agcct345_tilt_angles,
        # cct 电流
        dicct345_current=dicct345_current,
        agcct345_current=agcct345_current,

        part_per_winding=part_per_winding,
    ).create_first_bending_part_beamline()


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
    statistic_beam_sizes = BaseUtils.Statistic()

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
            # 分别求束斑
            beam_size_x = (statistic_x.max() - statistic_x.min()) / 2
            beam_size_y = (statistic_y.max() - statistic_y.min()) / 2

            statistic_x.clear()
            statistic_y.clear()

            # 只有 x 和 y 中大的我需要
            beam_size = max(beam_size_x, beam_size_y)

            statistic_beam_sizes.add(beam_size)  # 用于统计均值

            obj.append(beam_size)  # 用于记录每次束斑

        # 均值
        beam_size_avg = statistic_beam_sizes.average()
        objs.append([abs(bs - beam_size_avg) for bs in obj] + [beam_size_avg])
        statistic_beam_sizes.clear()

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
