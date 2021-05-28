"""
CCT 建模优化代码
机架束线优化核心部分 前偏转段优化

作者：赵润晓
日期：2021年5月6日
"""

from cctpy import *
from hust_sc_gantry import HUST_SC_GANTRY
import time
import numpy as np

# 可变参数
# 动量分散
momentum_dispersions = [-0.05, -0.025, 0.0, 0.025, 0.05]
# 每平面、每动量分散粒子数目
particle_number_per_plane_per_dp = 12
# 每个机架（束线）粒子数目
particle_number_per_gantry = len(momentum_dispersions) * particle_number_per_plane_per_dp * 2



ga32 = GPU_ACCELERATOR()


default_gantry = HUST_SC_GANTRY(
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
)

first_bending_part = default_gantry.create_first_bending_part_beamline()
first_bending_length = first_bending_part.get_length()

# 起点理想粒子
ip_start = ParticleFactory.create_proton_along(
    trajectory=first_bending_part,
    s=0.0,
    kinetic_MeV=215
)


# 终点理想粒子
ip_end = ParticleFactory.create_proton_along(
    trajectory=first_bending_part,
    s=first_bending_length,
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
        distance=first_bending_length,
        footstep=20 * MM
    )

    # 统计器
    statistic_x = BaseUtils.Statistic()
    statistic_y = BaseUtils.Statistic()
    statistic_beam_sizes = BaseUtils.Statistic()
    statistic_expend_max = BaseUtils.Statistic()

    # 所有机架 所有目标
    objs: List[List[float]] = []

    # 对于每个机架
    for gid in range(gantry_number):  # ~120
        # 
        ps_end_list_each_gantry:List[RunningParticle] = ps_end_list_list[gid]
        # 不知道为什么，有些粒子的速率 speed 和速度 velocity 差别巨大
        for p in ps_end_list_each_gantry:
            p.speed=p.velocity.length()

        pps_end_each_gantry:List[PhaseSpaceParticle] = PhaseSpaceParticle.create_from_running_particles(
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

            statistic_expend_max.add(max(
                abs(statistic_x.max()),
                abs(statistic_x.min()),
                abs(statistic_y.max()),
                abs(statistic_y.min()),
            ))

            statistic_x.clear()
            statistic_y.clear()

            # 只有 x 和 y 中大的我需要
            # beam_size = max(beam_size_x, beam_size_y)

            statistic_beam_sizes.add(beam_size_x)  # 用于统计均值
            statistic_beam_sizes.add(beam_size_y)  # 用于统计均值
            obj.append(beam_size_x)  # 用于记录每次束斑
            obj.append(beam_size_y)  # 用于记录每次束斑

        # 均值
        beam_size_avg = statistic_beam_sizes.average()
        objs.append([abs(bs - beam_size_avg) for bs in obj] + [statistic_expend_max.max()])
        statistic_beam_sizes.clear()
        statistic_expend_max.clear()

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
        task=create_beamline,
        param_list=[[params[i]] for i in range(gantry_number)]
    )


def create_beamline(param) -> Beamline:
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


    return HUST_SC_GANTRY(
        qs1_gradient=qs1_g,
        qs2_gradient=qs2_g,
        qs1_second_gradient=qs1_s,
        qs2_second_gradient=qs2_s,

        qs1_aperture_radius=60*MM,
        qs2_aperture_radius=60*MM,

        dicct12_tilt_angles=[30, dicct_tilt_1, dicct_tilt_2, dicct_tilt_3],
        agcct12_tilt_angles=[agcct_tilt_0, 30, agcct_tilt_2, agcct_tilt_3],

        dicct12_current=dicct_current,
        agcct12_current=agcct_current,

        agcct1_winding_number=agcct1_wn,
        agcct2_winding_number=agcct2_wn,
        dicct12_winding_number=42,


        agcct1_bending_angle=22.5 * (agcct1_wn / (agcct1_wn + agcct2_wn)),
        agcct2_bending_angle=22.5 * (agcct2_wn / (agcct1_wn + agcct2_wn)),

        DL1=0.9007765,
        GAP1=0.4301517,
        GAP2=0.370816,
        qs1_length=0.2340128,
        qs2_length=0.200139,

        DL2=2.35011,
        GAP3=0.43188,
        qs3_length=0.24379,

        agcct12_inner_small_r=92.5 * MM - 20 *MM,  # 92.5
        agcct12_outer_small_r=108.5 * MM- 20 *MM,  # 83+15
        dicct12_inner_small_r=124.5 * MM- 20 *MM,  # 83+30+1
        dicct12_outer_small_r=140.5 * MM- 20 *MM,  # 83+45 +2
    ).create_first_bending_part_beamline()

