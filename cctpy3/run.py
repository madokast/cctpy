from cctpy import *
from ccpty_cuda import *
import time
import numpy as np
import math

ga32 = GPU_ACCELERATOR()

momentum_dispersions = [-0.05, -0.025, 0.0, 0.025, 0.05]
particle_number_per_plane_per_dp = 24

particle_number_per_gantry = len(momentum_dispersions) * particle_number_per_plane_per_dp * 2

default_gantry = HUST_SC_GANTRY()
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
        footstep=10 * MM
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

    np.savetxt(fname='./record_5000_careful/' + str(times) + '.txt', X=params_and_objs)
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

    agcct3_wn = int(np.round(param[10]))
    agcct4_wn = int(np.round(param[11]))
    agcct5_wn = int(np.round(param[12]))

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
        agcct3_bending_angle=-67.5*(agcct3_wn/(agcct3_wn+agcct4_wn+agcct5_wn)),
        agcct4_bending_angle=-67.5*(agcct4_wn/(agcct3_wn+agcct4_wn+agcct5_wn)),
        agcct5_bending_angle=-67.5*(agcct5_wn/(agcct3_wn+agcct4_wn+agcct5_wn)),
    ).create_second_bending_part(
        start_point=second_bending_part_start_point,
        start_driect=second_bending_part_start_direct
    )


