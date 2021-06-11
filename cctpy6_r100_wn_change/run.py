# from visdom import Visdom

from cctpy import *
from ccpty_cuda import *
import time
import numpy as np

VIZ_PORT = 8098

ga32 = GPU_ACCELERATOR()

momentum_dispersions = [-0.05, -0.025, 0.0, 0.025, 0.05]
particle_number_per_plane_per_dp = 12

particle_number_per_gantry = len(momentum_dispersions) * particle_number_per_plane_per_dp * 2

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
default_beamline = default_gantry.create_beamline()
first_bending_length = default_gantry.first_bending_part_length()
run_distance = default_beamline.get_length() - first_bending_length

second_bending_part_start_point = default_beamline.trajectory.point_at(first_bending_length)
second_bending_part_start_direct = default_beamline.trajectory.direct_at(first_bending_length)

ip = ParticleFactory.create_proton_along(
    trajectory=default_beamline.trajectory,
    s=first_bending_length,
    kinetic_MeV=215
)

ip_ran = ParticleFactory.create_proton_along(
    trajectory=default_beamline.trajectory,
    s=default_beamline.get_length(),
    kinetic_MeV=215
)

pps = []
for dp in momentum_dispersions:
    pps.extend(PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_xxp_plane(
        xMax=3.5 * MM, xpMax=7.5 * MM, delta=dp, number=particle_number_per_plane_per_dp
    ))
    pps.extend(PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_yyp_plane(
        yMax=3.5 * MM, ypMax=7.5 * MM, delta=dp, number=particle_number_per_plane_per_dp
    ))

times = 1

params_and_objs = []


def run(params: np.ndarray):
    global times
    start_time = time.time()

    gantry_number = params.shape[0]

    print(f"机架数目{gantry_number}")

    beamlines = create_beamlines(gantry_number, params)

    print(f"制作机架用时{time.time() - start_time}")
    ps = ParticleFactory.create_from_phase_space_particles(
        ip, ip.get_natural_coordinate_system(), pps
    )

    print(f"粒子总数{len(ps) * gantry_number}")

    ps_ran_list = ga32.track_multi_particle_beamlime_for_magnet_with_single_qs(
        bls=beamlines,
        ps=ps,
        distance=run_distance,
        footstep=20 * MM
    )

    statistic_x = BaseUtils.Statistic()
    statistic_y = BaseUtils.Statistic()
    statistic_beam_sizes = BaseUtils.Statistic()
    objs: List[List[float]] = []
    for gid in range(gantry_number):  # ~120
        ps_ran = ps_ran_list[gid]
        pps_ran = PhaseSpaceParticle.create_from_running_particles(
            ip_ran, ip_ran.get_natural_coordinate_system(), ps_ran
        )
        obj: List[float] = []
        # 对于所有粒子
        for pid in range(0, len(pps_ran), particle_number_per_plane_per_dp):
            # 每 particle_number_per_plane_per_dp 个一组
            for pp in pps_ran[pid:pid + particle_number_per_plane_per_dp]:
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
        statistic_beam_sizes.clear()
        objs.append([abs(bs - beam_size_avg) for bs in obj] + [beam_size_avg])

    objs_np = np.array(objs)

    for gid in range(gantry_number):
        param = params[gid]
        obj = objs_np[gid]
        params_and_objs.append(np.concatenate((param, obj)))

    np.savetxt(fname='./record/' + str(times) + '.txt', X=params_and_objs)
    try:
        # draw_viz(params_and_objs)
        pass
    except Exception as e:
        print(e)
        pass
    times += 1

    print(f"用时{time.time() - start_time} s")

    return objs_np


def create_beamlines(gantry_number, params):
    return BaseUtils.submit_process_task(
        task=create_beamline,
        param_list=[
            [params[i], second_bending_part_start_point, second_bending_part_start_direct] for i in range(gantry_number)
        ]
    )


def create_beamline(param, second_bending_part_start_point, second_bending_part_start_direct) -> Beamline:
    qs3_g = param[0]
    qs3_sg = param[1]

    dicct_tilt_1 = param[2]
    dicct_tilt_2 = param[3]
    dicct_tilt_3 = param[4]

    agcct_tilt_0 = param[5]
    agcct_tilt_2 = param[6]
    agcct_tilt_3 = param[7]

    dicct_current = param[8]
    agcct_current = param[9]

    agcct3_wn = int(param[10])
    agcct4_wn = int(param[11])
    agcct5_wn = int(param[12])

    return HUST_SC_GANTRY(
        qs3_gradient=qs3_g,
        qs3_second_gradient=qs3_sg,
        dicct345_tilt_angles=[30, dicct_tilt_1, dicct_tilt_2, dicct_tilt_3],
        agcct345_tilt_angles=[agcct_tilt_0, 30, agcct_tilt_2, agcct_tilt_3],
        dicct345_current=dicct_current,
        agcct345_current=agcct_current,
        agcct3_winding_number=agcct3_wn,
        agcct4_winding_number=agcct4_wn,
        agcct5_winding_number=agcct5_wn,
        agcct3_bending_angle=-67.5 * (agcct3_wn / (agcct3_wn + agcct4_wn + agcct5_wn)),
        agcct4_bending_angle=-67.5 * (agcct4_wn / (agcct3_wn + agcct4_wn + agcct5_wn)),
        agcct5_bending_angle=-67.5 * (agcct5_wn / (agcct3_wn + agcct4_wn + agcct5_wn)),

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

        agcct345_inner_small_r=92.5 * MM + 17.1 * MM,# 92.5
        agcct345_outer_small_r=108.5 * MM + 17.1 * MM,  # 83+15
        dicct345_inner_small_r=124.5 * MM + 17.1 * MM,  # 83+30+1
        dicct345_outer_small_r=140.5 * MM + 17.1 * MM,  # 83+45 +2
    ).create_second_bending_part(
        start_point=second_bending_part_start_point,
        start_driect=second_bending_part_start_direct
    )


wins = []  # 画图窗口


def draw_viz(params_and_objs):
    viz = Visdom(server='Http://127.0.0.1', port=VIZ_PORT)
    assert viz.check_connection()

    data = np.array(params_and_objs)

    x = np.array(list(range(data.shape[0])))

    xd = np.concatenate((x.reshape((-1, 1)), data), axis=1)

    # xd 每一列的意义
    # 0 编号 0-34265
    # 12 qs参数
    # 345 / 678 CCT倾斜角参数
    # 9 10 电流
    # 11 12 13 匝数
    # 14 15 16 17 18
    # 19 20 21 22 23 束斑和均值差
    # 24 束斑均值

    lables = ['qs-q', 'qs-s',
              'dicct-t4', 'dicct-t6', 'dicct-t8',
              'agcct-t2', 'agcct-t6', 'agcct-t8',
              'dicct-I', 'agcct-I',
              'agcct-wn0', 'agcct-wn1', 'agcct-wn2',
              'diff_size1', 'diff_size2', 'diff_size3', 'diff_size4', 'diff_size5',
              'diff_size6', 'diff_size7', 'diff_size8', 'diff_size9', 'diff_size0',
              'beam_avg', 'max_diff_size']

    for i in range(len(lables)):
        if len(wins) != len(lables):
            if i == len(lables) - 1:  # last
                wins.append(viz.scatter(
                    X=np.vstack((xd[:, 0], np.max(xd[:, 14:24], axis=1))).T,
                    opts={
                        'title': lables[i] + ' vs individual',
                        'xlabel': 'individual',
                        'ylabel': lables[i],
                        'markersize': 2
                    }
                ))
            else:
                wins.append(viz.scatter(
                    X=np.vstack((xd[:, 0], xd[:, i + 1])).T,
                    opts={
                        'title': lables[i] + ' vs individual',
                        'xlabel': 'individual',
                        'ylabel': lables[i],
                        'markersize': 2
                    }
                ))
        else:
            if i == len(lables) - 1:  # last
                wins[i] = viz.scatter(
                    X=np.vstack((xd[:, 0], np.max(xd[:, 14:24], axis=1))).T,
                    win=wins[i],
                    opts={
                        'title': lables[i] + ' vs individual',
                        'xlabel': 'individual',
                        'ylabel': lables[i],
                        'markersize': 2
                    }
                )
            else:
                viz.scatter(
                    X=np.vstack((xd[:, 0], xd[:, i + 1])).T,
                    win=wins[i],
                    opts={
                        'title': lables[i] + ' vs individual',
                        'xlabel': 'individual',
                        'ylabel': lables[i],
                        'markersize': 2
                    }
                )
