"""
数据输入输出
"""
from typing import List

import numpy as np

from cctpy.constant import MM
from cctpy.particle import PhaseSpaceParticle, RunningParticle, ParticleFactory
from cuda_optim import COSY_MAPS
from cctpy.baseutils import Vectors, Stream, Average, Statistic

# 优化匹配的变量数目
VARIABLE_NUMBER: int = 10

# 第二偏转段入口参考粒子
IP_GANTRY_PART2_ENTRY: RunningParticle = ParticleFactory.create_proton_by_position_and_velocity(
    position=Vectors.create(3.703795764767298, 1.5341624380266436, 0.0),
    velocity=Vectors.create(1.2326128074269669E8, 1.2326128074269663E8, 0.0)
)
IP_ISOC: RunningParticle = ParticleFactory.create_proton_by_position_and_velocity(
    position=Vectors.create(7.407589792475514, 0.0, 0.0),
    velocity=Vectors.create(0.0, -1.7431777494179922E8, 0.0)
)

# CUDA 每个粒子参数
CUDA_PARAMS_PER_PARTICLE: int = 9
CUDA_RESULT_GROUP_LEN: int = 6


def read_param() -> np.ndarray:
    """
    读取输入，格式如下：
    2 输入数目
    1 编号，从 1 开始
    -9.208 以下是 10 个参数
    -53.455
    80.
    88.
    92.
    107.1
    83.355
    77.87
    -9507.95
    -5608.6
    2 第二组
    -9.208
    53.455
    80.
    88.
    92.
    107.1
    83.355
    77.87
    -9507.95
    -5708.6



    转为如下格式：
    [[-9.20800e+00 -5.34550e+01  8.00000e+01  8.80000e+01  9.20000e+01
       1.07100e+02  8.33550e+01  7.78700e+01 -9.50795e+03 -5.60860e+03]
     [-9.20800e+00  5.34550e+01  8.00000e+01  8.80000e+01  9.20000e+01
       1.07100e+02  8.33550e+01  7.78700e+01 -9.50795e+03 -5.70860e+03]]

    用于 cct345_data_generator.py 处理
    -------

    """
    input_file = np.loadtxt('input.txt', dtype=np.float64)
    gantry_number = int(input_file[0])
    data = np.empty((gantry_number, VARIABLE_NUMBER), dtype=np.float64)
    for i in range(gantry_number):
        if int(input_file[i * (VARIABLE_NUMBER + 1) + 1]) != i + 1:
            raise ValueError("输入文件不合法")

        data[i, :] = input_file[i * (VARIABLE_NUMBER + 1) + 2:(i + 1) * (VARIABLE_NUMBER + 1) + 1]

    return data


xMax = 3.5 * MM  # yMax == xMax
xpMax = 7.5 * MM  # ypMax == xpMax


def particle_to_cuda_ndarry(p: RunningParticle, distance: float) -> np.ndarray:
    """
    实际粒子，转为 CUDA 内的 float[] 数组
    px py pz  粒子绝对坐标
    vx vy vz  粒子速度
    rm speed distance 粒子动质量、速率和要运行的距离
    Parameters
    ----------
    p 粒子
    distance 要运行的距离

    Returns
    -------

    """
    return np.array([
        p.position[0],
        p.position[1],
        p.position[2],
        p.velocity[0],
        p.velocity[1],
        p.velocity[2],
        p.relativistic_mass,
        p.speed,
        distance
    ], dtype=np.float32)


def cuda_data_running_particles_at_second_part_entry(momentum_dispersion_list: List[float], particle_number: int,
                                                     run_distance: float = 7.104727865682728) -> np.array:
    """
    输入动量分散列表，如 [0, 0.05] 和粒子个数，如 4
    生成 cuda 需要的粒子参数，注意是 float32 格式

    格式如下
    一维数组，每 9 个一组，每组的含义如下：
    px py pz  粒子绝对坐标
    vx vy vz  粒子速度
    rm speed distance 粒子动质量、速率和要运行的距离

    典型格式如下：
    particle_data = np.array([
        3.5096529382644635, 1.4572458462517228, 0.0,
        1.240730221668078E8, 1.2407302216680774E8, 0.0,
        2.062868061238519E-27, 1.7546575067291722E8, 7.104727865682728,

        3.513714575731084, 1.4531842087851023, 0.0,
        1.2335890489598374E8, 1.247871394376318E8, 0.0,
        2.062868061238519E-27, 1.7546575067291722E8, 7.104727865682728,

        3.5137121631324737, 1.4531866213837126, 0.0,
        1.2478789633775285E8, 1.233581479958627E8, 0.0,
        2.062868061238519E-27, 1.7546575067291722E8, 7.104727865682728,
    ], dtype=np.float32)

    Parameters
    ----------
    momentum_dispersion_list 动量分散列表，如 [0, 0.05]
    particle_number 粒子个数，如 4

    Returns cuda 需要的粒子参数
    -------

    """
    pp = []
    for plane in [PhaseSpaceParticle.XXP_PLANE, PhaseSpaceParticle.YYP_PLANE]:
        for dp in momentum_dispersion_list:
            pp.extend(PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_plane(
                plane, xMax, xpMax, dp, particle_number
            ))  # list.extend 相当于 Java 的 addAll

    # 转为能量分散，用于 COSY
    pp = PhaseSpaceParticle.convert_delta_from_momentum_dispersion_to_energy_dispersion_for_list(pp, 215)

    # run cosy
    pp_end = COSY_MAPS.MAP_LIAOYICHENG_PART1.apply_phase_space_particles(pp, order=5)

    # 能量分散转回去
    pp_end = PhaseSpaceParticle. \
        convert_delta_from_energy_dispersion_to_energy_dispersion_momentum_dispersion_for_list(pp_end, 215)

    # 变成实际粒子
    rps = ParticleFactory.create_from_phase_space_particles(
        IP_GANTRY_PART2_ENTRY, IP_GANTRY_PART2_ENTRY.get_natural_coordinate_system(), pp_end
    )

    # 粒子数目
    total_num = len(rps)

    particle_data = np.empty((total_num * CUDA_PARAMS_PER_PARTICLE,), dtype=np.float32)

    for i in range(total_num):
        particle_data[i * CUDA_PARAMS_PER_PARTICLE:(i + 1) * CUDA_PARAMS_PER_PARTICLE] = particle_to_cuda_ndarry(rps[i],
                                                                                                                 run_distance)

    return particle_data


def cosy_particles_at_ISOC(momentum_dispersion_list: List[float], particle_number: int) -> List[PhaseSpaceParticle]:
    pp = []
    for plane in [PhaseSpaceParticle.XXP_PLANE, PhaseSpaceParticle.YYP_PLANE]:
        for dp in momentum_dispersion_list:
            pp.extend(PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_plane(
                plane, xMax, xpMax, dp, particle_number
            ))  # list.extend 相当于 Java 的 addAll

    # 转为能量分散，用于 COSY
    pp = PhaseSpaceParticle.convert_delta_from_momentum_dispersion_to_energy_dispersion_for_list(pp, 215)

    # run cosy
    pp_end = COSY_MAPS.MAP_LIAOYICHENG_ALL_GANTRY.apply_phase_space_particles(pp, order=5)

    # 能量分散转回去
    return PhaseSpaceParticle. \
        convert_delta_from_energy_dispersion_to_energy_dispersion_momentum_dispersion_for_list(pp_end, 215)


def cuda_particle_data_to_running_particle(data: np.ndarray, gantry_number: int, particle_number_per_gantry: int) -> \
        List[List[RunningParticle]]:
    """
    cuda 返回的 result 数据，转回 RunningParticle

    关于 result 的格式，是一个一维的 float32 数组
    每 6 个一组，6 个参数分别是：
    px py pz
    vx vy vz

    Parameters
    ----------
    data cuda 返回值
    particle_number 粒子数目
    gantry_number 机架数目，即组数

    Returns 二维数组，第一维 机架编号 从 0 开始，第二维 粒子编号，从 0 开始
    -------

    """
    total_len = data.shape[0]
    if total_len != CUDA_RESULT_GROUP_LEN * gantry_number * particle_number_per_gantry:
        raise ValueError(f"CUDA返回变量个数{total_len}和实际粒子数目不符{gantry_number * particle_number_per_gantry}")

    ret: List[List[RunningParticle]] = []

    for gid in range(gantry_number):
        ret_per_gantry: List[RunningParticle] = []
        for pid in range(particle_number_per_gantry):
            ret_per_gantry.append(ParticleFactory.create_proton_by_position_and_velocity(
                position=Vectors.create(
                    data[gid * particle_number_per_gantry * CUDA_RESULT_GROUP_LEN + pid * CUDA_RESULT_GROUP_LEN],
                    data[gid * particle_number_per_gantry * CUDA_RESULT_GROUP_LEN + pid * CUDA_RESULT_GROUP_LEN + 1],
                    data[gid * particle_number_per_gantry * CUDA_RESULT_GROUP_LEN + pid * CUDA_RESULT_GROUP_LEN + 2],
                ),
                velocity=Vectors.create(
                    data[gid * particle_number_per_gantry * CUDA_RESULT_GROUP_LEN + pid * CUDA_RESULT_GROUP_LEN + 3],
                    data[gid * particle_number_per_gantry * CUDA_RESULT_GROUP_LEN + pid * CUDA_RESULT_GROUP_LEN + 4],
                    data[gid * particle_number_per_gantry * CUDA_RESULT_GROUP_LEN + pid * CUDA_RESULT_GROUP_LEN + 5],
                )
            ))
        ret.append(ret_per_gantry)

    return ret


# 和 COSY 每个粒子对比
def analyze_and_output(ps: List[List[RunningParticle]], gantry_number: int, total_particle_number: int,
                       momentum_dispersion_list: List[float]) -> None:
    """
    分析运行结果，并写到 output.txt 中
    类似 1 2.9106590546670255 3.9272244111035284 1.9234584254384846 0.45806934921638964
    Parameters
    ----------
    ps 运行后的所有粒子
    gantry_number 机架数目 / 组数
    total_particle_number 总粒子数

    Returns 无
    -------

    """
    if gantry_number != len(ps):
        raise ValueError(f"数据错误，gantry_number{gantry_number}！=len(ps){len(ps)}")
    if int(total_particle_number) != int(len(ps[0]) * gantry_number):
        raise ValueError(
            f"数据错误，total_particle_number({total_particle_number})！=len(ps[0])*gantry_number({len(ps[0]) * gantry_number})")

    particle_number_per_plane_per_dp: int = total_particle_number // len(momentum_dispersion_list) // 2 // gantry_number

    # isoc 处 cosy pp
    cosy_pp = cosy_particles_at_ISOC(momentum_dispersion_list, particle_number_per_plane_per_dp)

    # 映射到 x y 平面 [[x1,xp1],[x2,xp2]]
    ppx_cosy = PhaseSpaceParticle.phase_space_particles_project_to_xxp_plane(cosy_pp)
    ppy_cosy = PhaseSpaceParticle.phase_space_particles_project_to_yyp_plane(cosy_pp)

    result: List[List[float]] = []

    # 对于每个机架 / 组
    for gid in range(gantry_number):
        avg_x = Average()
        avg_xp = Average()
        avg_y = Average()
        avg_yp = Average()

        # 第一列存访机架编号
        result_per_group: List[float] = [gid + 1]

        # 这组机架 track 得到的 List[RunningParticle]
        particle_group = ps[gid]

        # 转到 pp
        pp_group = PhaseSpaceParticle.create_from_running_particles(
            IP_ISOC, IP_ISOC.get_natural_coordinate_system(), particle_group)

        # 映射到 x p 平面
        ppx_track = PhaseSpaceParticle.phase_space_particles_project_to_xxp_plane(pp_group)
        ppy_track = PhaseSpaceParticle.phase_space_particles_project_to_yyp_plane(pp_group)

        # print(
        #    f"2020年11月19日 -- 每组机架粒子数目{ppx_track.shape[0]} particle_number_per_plane_per_dp={particle_number_per_plane_per_dp}")

        # 对于 x 平面，List[RunningParticle]中前一半
        for i in range(0, particle_number_per_plane_per_dp * len(momentum_dispersion_list)):
            pc = ppx_cosy[i]
            pt = ppx_track[i]
            avg_x.add(np.abs(pc[0] - pt[0]) / MM)
            avg_xp.add(np.abs(pc[1] - pt[1]) / MM)

        # 对于 y 平面
        for i in range(particle_number_per_plane_per_dp * len(momentum_dispersion_list),
                       particle_number_per_plane_per_dp * len(momentum_dispersion_list) * 2):
            pc = ppy_cosy[i]
            pt = ppy_track[i]
            avg_y.add(np.abs(pc[0] - pt[0]) / MM)
            avg_yp.add(np.abs(pc[1] - pt[1]) / MM)

        result_per_group.append(avg_x.average())
        result_per_group.append(avg_xp.average())
        result_per_group.append(avg_y.average())
        result_per_group.append(avg_yp.average())

        result.append(result_per_group)

    np.savetxt('output.txt', np.array(result))


# 不和 COSY 对比，仅仅计算束斑，目标为 abs(X/Y束斑-3.5MM) ，动量分散合起来，目标两个
def analyze1119_and_output(ps: List[List[RunningParticle]], gantry_number: int, total_particle_number: int,
                           momentum_dispersion_list: List[float]) -> None:
    """
    分析运行结果，并写到 output.txt 中
    类似 1 2.9106590546670255 3.9272244111035284 1.9234584254384846 0.45806934921638964
    Parameters
    ----------
    ps 运行后的所有粒子
    gantry_number 机架数目 / 组数
    total_particle_number 总粒子数

    Returns 无
    -------

    """
    if gantry_number != len(ps):
        raise ValueError(f"数据错误，gantry_number{gantry_number}！=len(ps){len(ps)}")
    if int(total_particle_number) != int(len(ps[0]) * gantry_number):
        raise ValueError(
            f"数据错误，total_particle_number({total_particle_number})！=len(ps[0])*gantry_number({len(ps[0]) * gantry_number})")

    particle_number_per_plane_per_dp: int = total_particle_number // len(momentum_dispersion_list) // 2 // gantry_number

    result: List[List[float]] = []

    # 对于每个机架 / 组
    for gid in range(gantry_number):
        # 第一列存访机架编号
        result_per_group: List[float] = [gid + 1]

        # 这组机架 track 得到的 List[RunningParticle]
        particle_group = ps[gid]

        # 转到 pp
        pp_group = PhaseSpaceParticle.create_from_running_particles(
            IP_ISOC, IP_ISOC.get_natural_coordinate_system(), particle_group)

        # 映射到 x p 平面
        ppx_track = PhaseSpaceParticle.phase_space_particles_project_to_xxp_plane(pp_group)
        ppy_track = PhaseSpaceParticle.phase_space_particles_project_to_yyp_plane(pp_group)
        # 对于 x 平面，List[RunningParticle]中前一半

        statistic = Statistic()

        # 对于 x 平面
        for i in range(0, particle_number_per_plane_per_dp * len(momentum_dispersion_list)):
            pt = ppx_track[i]
            statistic.add(pt[0])

        result_per_group.append(
            # x 方向所有动量分散，粒子叠起来，求束斑，和 3.5 做差，取绝对值
            np.abs(3.5 * MM - (statistic.max() - statistic.min()) / MM / 2.0)
        )
        statistic.clear()

        # 对于 y 平面
        for i in range(particle_number_per_plane_per_dp * len(momentum_dispersion_list),
                       particle_number_per_plane_per_dp * len(momentum_dispersion_list) * 2):
            pt = ppy_track[i]
            statistic.add(pt[0])
        result_per_group.append(
            np.abs(3.5 * MM - (statistic.max() - statistic.min()) / MM / 2.0)
        )
        statistic.clear()
        result.append(result_per_group)

    np.savetxt('output.txt', np.array(result))


# 不和 COSY 对比，仅仅计算束斑，目标为 abs(束斑-3.5MM) 因此目标个数 = 平面 2 * 动量分散数目 k = 2k
def analyze1121_and_output(ps: List[List[RunningParticle]], gantry_number: int, total_particle_number: int,
                           momentum_dispersion_list: List[float]) -> None:
    """
    分析运行结果，并写到 output.txt 中
    类似 1 2.9106590546670255 3.9272244111035284 1.9234584254384846 0.45806934921638964
    Parameters
    ----------
    ps 运行后的所有粒子
    gantry_number 机架数目 / 组数
    total_particle_number 总粒子数

    Returns 无
    -------

    """
    if gantry_number != len(ps):
        raise ValueError(f"数据错误，gantry_number{gantry_number}！=len(ps){len(ps)}")
    if int(total_particle_number) != int(len(ps[0]) * gantry_number):
        raise ValueError(
            f"数据错误，total_particle_number({total_particle_number})！=len(ps[0])*gantry_number({len(ps[0]) * gantry_number})")

    particle_number_per_plane_per_dp: int = total_particle_number // len(momentum_dispersion_list) // 2 // gantry_number

    result: List[List[float]] = []

    # 对于每个机架 / 组
    for gid in range(gantry_number):
        # 第一列存访机架编号
        result_per_group: List[float] = [gid + 1]

        # 这组机架 track 得到的 List[RunningParticle]
        particle_group = ps[gid]

        # 转到 pp
        pp_group = PhaseSpaceParticle.create_from_running_particles(
            IP_ISOC, IP_ISOC.get_natural_coordinate_system(), particle_group)

        # 映射到 x p 平面
        ppx_track = PhaseSpaceParticle.phase_space_particles_project_to_xxp_plane(pp_group)
        ppy_track = PhaseSpaceParticle.phase_space_particles_project_to_yyp_plane(pp_group)
        # 对于 x 平面，List[RunningParticle]中前一半

        statistic = Statistic()

        # 对于 x 平面
        for i in range(0, particle_number_per_plane_per_dp * len(momentum_dispersion_list),
                       particle_number_per_plane_per_dp):
            for j in range(0, particle_number_per_plane_per_dp):
                pt = ppx_track[i + j]
                statistic.add(pt[0])
            result_per_group.append(
                # x 方向当前动量分散包络，和 3.5 做差，取绝对值
                np.abs(3.5 - (statistic.max() - statistic.min()) / MM / 2.0)
            )
            statistic.clear()

        # y 平面
        for i in range(particle_number_per_plane_per_dp * len(momentum_dispersion_list),
                       particle_number_per_plane_per_dp * len(momentum_dispersion_list) * 2,
                       particle_number_per_plane_per_dp):
            for j in range(0, particle_number_per_plane_per_dp):
                pt = ppy_track[i + j]
                statistic.add(pt[0])
            result_per_group.append(
                # x 方向当前动量分散包络，和 3.5 做差，取绝对值
                np.abs(3.5 - (statistic.max() - statistic.min()) / MM / 2.0)
            )
            statistic.clear()

        result.append(result_per_group)

    np.savetxt('output.txt', np.array(result))


# 不和 COSY 对比，匹配方差 输出 1 行
def analyze1123_and_output(ps: List[List[RunningParticle]], gantry_number: int, total_particle_number: int,
                           momentum_dispersion_list: List[float]) -> None:
    """
    分析运行结果，并写到 output.txt 中
    类似 1 2.9106590546670255 3.9272244111035284 1.9234584254384846 0.45806934921638964
    Parameters
    ----------
    ps 运行后的所有粒子
    gantry_number 机架数目 / 组数
    total_particle_number 总粒子数

    Returns 无
    -------

    """
    if gantry_number != len(ps):
        raise ValueError(f"数据错误，gantry_number{gantry_number}！=len(ps){len(ps)}")
    if int(total_particle_number) != int(len(ps[0]) * gantry_number):
        raise ValueError(
            f"数据错误，total_particle_number({total_particle_number})！=len(ps[0])*gantry_number({len(ps[0]) * gantry_number})")

    particle_number_per_plane_per_dp: int = total_particle_number // len(momentum_dispersion_list) // 2 // gantry_number

    result: List[List[float]] = []

    # 对于每个机架 / 组
    for gid in range(gantry_number):
        # 第一列存访机架编号
        result_per_group: List[float] = [gid + 1]

        # 这组机架 track 得到的 List[RunningParticle]
        particle_group = ps[gid]

        # 转到 pp
        pp_group = PhaseSpaceParticle.create_from_running_particles(
            IP_ISOC, IP_ISOC.get_natural_coordinate_system(), particle_group)

        # 映射到 x p 平面
        ppx_track = PhaseSpaceParticle.phase_space_particles_project_to_xxp_plane(pp_group)
        ppy_track = PhaseSpaceParticle.phase_space_particles_project_to_yyp_plane(pp_group)
        # 对于 x 平面，List[RunningParticle]中前一半

        statistic = Statistic()
        outer_statistic = Statistic()

        # 对于 x 平面
        for i in range(0, particle_number_per_plane_per_dp * len(momentum_dispersion_list),
                       particle_number_per_plane_per_dp):
            for j in range(0, particle_number_per_plane_per_dp):
                pt = ppx_track[i + j]
                statistic.add(pt[0])
            outer_statistic.add((statistic.max() - statistic.min()) / MM / 2.0)
            statistic.clear()

        # y 平面
        for i in range(particle_number_per_plane_per_dp * len(momentum_dispersion_list),
                       particle_number_per_plane_per_dp * len(momentum_dispersion_list) * 2,
                       particle_number_per_plane_per_dp):
            for j in range(0, particle_number_per_plane_per_dp):
                pt = ppy_track[i + j]
                statistic.add(pt[0])
            outer_statistic.add((statistic.max() - statistic.min()) / MM / 2.0)
            statistic.clear()

        result_per_group.append(outer_statistic.var())
        outer_statistic.clear()

        result.append(result_per_group)

    np.savetxt('output.txt', np.array(result))


# 不和 COSY 对比，匹配方差和最大包络 输出 1 行
def analyze1127_and_output(ps: List[List[RunningParticle]], gantry_number: int, total_particle_number: int,
                           momentum_dispersion_list: List[float]) -> None:
    """
    分析运行结果，并写到 output.txt 中
    类似 1 2.9106590546670255 3.9272244111035284 1.9234584254384846 0.45806934921638964
    Parameters
    ----------
    ps 运行后的所有粒子
    gantry_number 机架数目 / 组数
    total_particle_number 总粒子数

    Returns 无
    -------

    """
    if gantry_number != len(ps):
        raise ValueError(f"数据错误，gantry_number{gantry_number}！=len(ps){len(ps)}")
    if int(total_particle_number) != int(len(ps[0]) * gantry_number):
        raise ValueError(
            f"数据错误，total_particle_number({total_particle_number})！=len(ps[0])*gantry_number({len(ps[0]) * gantry_number})")

    particle_number_per_plane_per_dp: int = total_particle_number // len(momentum_dispersion_list) // 2 // gantry_number

    result: List[List[float]] = []

    # 对于每个机架 / 组
    for gid in range(gantry_number):
        # 第一列存访机架编号
        result_per_group: List[float] = [gid + 1]

        # 这组机架 track 得到的 List[RunningParticle]
        particle_group = ps[gid]

        # 转到 pp
        pp_group = PhaseSpaceParticle.create_from_running_particles(
            IP_ISOC, IP_ISOC.get_natural_coordinate_system(), particle_group)

        # 映射到 x p 平面
        ppx_track = PhaseSpaceParticle.phase_space_particles_project_to_xxp_plane(pp_group)
        ppy_track = PhaseSpaceParticle.phase_space_particles_project_to_yyp_plane(pp_group)
        # 对于 x 平面，List[RunningParticle]中前一半

        statistic = Statistic()
        outer_statistic = Statistic()

        # 对于 x 平面
        for i in range(0, particle_number_per_plane_per_dp * len(momentum_dispersion_list),
                       particle_number_per_plane_per_dp):
            for j in range(0, particle_number_per_plane_per_dp):
                pt = ppx_track[i + j]
                statistic.add(pt[0])
            outer_statistic.add((statistic.max() - statistic.min()) / MM / 2.0)
            statistic.clear()

        # y 平面
        for i in range(particle_number_per_plane_per_dp * len(momentum_dispersion_list),
                       particle_number_per_plane_per_dp * len(momentum_dispersion_list) * 2,
                       particle_number_per_plane_per_dp):
            for j in range(0, particle_number_per_plane_per_dp):
                pt = ppy_track[i + j]
                statistic.add(pt[0])
            outer_statistic.add((statistic.max() - statistic.min()) / MM / 2.0)
            statistic.clear()

        result_per_group.append(outer_statistic.var())
        result_per_group.append(outer_statistic.max())
        outer_statistic.clear()

        result.append(result_per_group)

    np.savetxt('output.txt', np.array(result))


if __name__ == '__main__':
    data = np.array(
        [7.214325904846191, -0.0726986676454544, -0.0025632025208324194, 458844.28125, -174522352.0, -62833.609375,
         7.219380855560303, -0.03869308531284332, -0.0011728801764547825, 414757.90625, -180155952.0, -46914.82421875,
         7.217695236206055, -0.0725758746266365, -0.00033737538615241647, -614740.1875, -174515872.0, 501747.40625,
         7.216037273406982, -0.037836577743291855, -0.0003209249989595264, -1225645.375, -180144496.0, 659188.1875,
         7.215950965881348, -0.07231692224740982, -0.0025141446385532618, 522280.5625, -174522560.0, -56366.87890625,
         7.220852851867676, -0.03953986242413521, -0.00117855507414788, 570043.5625, -180156128.0, -46090.08984375,
         7.217308044433594, -0.07213743776082993, 0.0006527822115458548, -672383.1875, -174515872.0, 517166.75,
         7.215423583984375, -0.03856635093688965, 0.0006339816027320921, -1207285.25, -180144672.0, 677566.9375])

    rps = cuda_particle_data_to_running_particle(data, 2, 4)
    Stream(rps).peek(lambda ps: print(len(ps))).foreach(lambda ps: Stream(ps).foreach_println())

    print(read_param())

    ps = cuda_data_running_particles_at_second_part_entry(
        momentum_dispersion_list=[0.0, 0.05], particle_number=5, run_distance=7.104727865682728)

    Stream(ps).foreach_println()
