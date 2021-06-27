"""
CCT 建模优化代码
粒子类

作者：赵润晓
日期：2021年4月29日
"""

import multiprocessing  # since v0.1.1 多线程计算
import time  # since v0.1.1 统计计算时长
from typing import Callable, Dict, Generic, Iterable, List, NoReturn, Optional, Tuple, TypeVar, Union
import matplotlib.pyplot as plt
import math
import random  # since v0.1.1 随机数
import sys
import os  # since v0.1.1 查看CPU核心数
import numpy
from scipy.integrate import solve_ivp  # since v0.1.1 ODE45
import warnings  # since v0.1.1 提醒方法过时
from packages.point import *
from packages.constants import *
from packages.base_utils import BaseUtils
from packages.local_coordinate_system import LocalCoordinateSystem
from packages.line2s import *
from packages.trajectory import Trajectory
from packages.magnets import Magnet


class Protons:
    """
    质子相关常量和计算
    """

    # 静止质量
    STATIC_MASS_KG = 1.672621898e-27

    # 静止能量 = m0 * c ^ 2 单位焦耳
    STATIC_ENERGY_J = STATIC_MASS_KG * LIGHT_SPEED * LIGHT_SPEED

    # 静止能量 eV 为单位
    STATIC_ENERGY_eV = STATIC_ENERGY_J / eV

    # 静止能量 MeV 为单位，应该是 STATIC_ENERGY_J / MeV。但是写成字面量
    STATIC_ENERGY_MeV = 938.2720813

    # 电荷量 库伦
    CHARGE_QUANTITY = 1.6021766208e-19

    @classmethod
    def get_total_energy_MeV(cls, kinetic_energy_MeV: float) -> float:
        """
        计算总能量 MeV = 静止能量 + 动能
        Parameters
        ----------
        kinetic_energy_MeV 动能 MeV 一般为 250 Mev

        Returns 总能量 MeV
        -------

        """
        return cls.STATIC_ENERGY_MeV + kinetic_energy_MeV

    @classmethod
    def get_total_energy_J(cls, kinetic_energy_MeV: float) -> float:
        """
        计算总能量 焦耳
        Parameters
        ----------
        kinetic_energy_MeV 动能 MeV 一般为 250 Mev

        Returns 总能量 焦耳
        -------

        """
        return cls.get_total_energy_MeV(kinetic_energy_MeV) * MeV

    @classmethod
    def get_relativistic_mass(cls, kinetic_energy_MeV: float) -> float:
        """
        计算动质量 kg = 动能 / (c^2)
        Parameters
        ----------
        kinetic_energy_MeV 动能 MeV 一般为 250 Mev

        Returns 动质量 kg
        -------

        """
        return cls.get_total_energy_J(kinetic_energy_MeV) / LIGHT_SPEED / LIGHT_SPEED

    @classmethod
    def get_speed_m_per_s(cls, kinetic_energy_MeV: float) -> float:
        """
        计算速度 m/s = c * sqrt( 1 - (m0/m)^2 )
        Parameters
        ----------
        kinetic_energy_MeV 动能 MeV 一般为 250 Mev

        Returns 速度 m/s
        -------

        """
        return LIGHT_SPEED * math.sqrt(
            1
            - (cls.STATIC_MASS_KG / cls.get_relativistic_mass(kinetic_energy_MeV)) ** 2
        )

    @classmethod
    def get_momentum_kg_m_pre_s(cls, kinetic_energy_MeV: float) -> float:
        """
        动量 kg m/s
        Parameters
        ----------
        kinetic_energy_MeV 动能 MeV 一般为 250 Mev

        Returns 动量 kg m/s
        -------

        """
        return cls.get_relativistic_mass(kinetic_energy_MeV) * cls.get_speed_m_per_s(
            kinetic_energy_MeV
        )

    @classmethod
    def getMomentum_MeV_pre_c(cls, kinetic_energy_MeV: float) -> float:
        """
        动量 MeV/c
        Parameters 动能 MeV 一般为 250 Mev
        ----------
        kinetic_energy_MeV

        Returns 动量 MeV/c
        -------

        """
        return cls.get_momentum_kg_m_pre_s(kinetic_energy_MeV) / MeV_PER_C

    @classmethod
    def get_magnetic_stiffness(cls, kinetic_energy_MeV: float) -> float:
        """
        磁钢度 T/m
        Parameters
        ----------
        kinetic_energy_MeV 动能 MeV 一般为 250 Mev

        Returns 磁钢度 T/m
        -------

        """
        return cls.get_momentum_kg_m_pre_s(kinetic_energy_MeV) / cls.CHARGE_QUANTITY

    # ------------------  动量分散相关  ----------------------
    @classmethod
    def get_kinetic_energy_MeV(cls, momentum_KG_M_PER_S: float) -> float:
        """
        质子动量 kg m/s 转动能 MeV
        print("质子动量 kg m/s 转动能 MeV，动量 3.896699309502749e-19 kg m/s",
            Protons.get_kinetic_energy_MeV(3.896699309502749e-19)) 
        # 249.99999982031446
        """
        # 速度
        speed = momentum_KG_M_PER_S / math.sqrt(
            cls.STATIC_MASS_KG ** 2 + (momentum_KG_M_PER_S / LIGHT_SPEED) ** 2
        )
        # 动质量
        relativistic_mass = cls.STATIC_MASS_KG / math.sqrt(
            1 - (speed / LIGHT_SPEED) ** 2
        )
        # 总能量 J
        total_energy_J = relativistic_mass * LIGHT_SPEED * LIGHT_SPEED
        # 动能 J
        k = total_energy_J - cls.STATIC_ENERGY_J

        return k / MeV

    # @classmethod
    # def get动量分散后的动能(cls, 原动能_MeV: float, 动量分散: float):
    #     """
    #     英文版见下
    #     Parameters
    #     ----------
    #     原动能_MeV
    #     动量分散
    #
    #     Returns 动量分散后的动能 MeV
    #     -------
    #
    #     """
    #     原动量 = cls.get_momentum_kg_m_pre_s(原动能_MeV)
    #
    #     新动量 = 原动量 * (1 + 动量分散)
    #
    #     新动能 = cls.get_kinetic_energy_MeV(新动量)
    #
    #     return 新动能

    @classmethod
    def get_kinetic_energy_MeV_after_momentum_dispersion(
            cls, old_kinetic_energy_MeV: float, momentum_dispersion: float
    ) -> float:
        """
        中文版见上
        Parameters
        ----------
        old_kinetic_energy_MeV 原动能_MeV
        momentum_dispersion 动量分散

        Returns 动量分散后的动能 MeV
        -------

        """
        momentum0 = cls.get_momentum_kg_m_pre_s(old_kinetic_energy_MeV)

        momentum = momentum0 * (1 + momentum_dispersion)

        kinetic_energy = cls.get_kinetic_energy_MeV(momentum)

        return kinetic_energy

    # @classmethod
    # def convert动量分散_TO_能量分散(cls, 动量分散: float, 动能_MeV: float) -> float:
    #     """
    #     下方法的中文版
    #     Parameters
    #     ----------
    #     动量分散
    #     动能_MeV
    #
    #     Returns convert动量分散_TO_能量分散
    #     -------
    #
    #     """
    #     k = (动能_MeV + cls.STATIC_ENERGY_MeV) / \
    #         (动能_MeV + 2 * cls.STATIC_ENERGY_MeV)
    #
    #     return 动量分散 / k

    @classmethod
    def convert_momentum_dispersion_to_energy_dispersion(
            cls, momentum_dispersion: float, kinetic_energy_MeV: float
    ) -> float:
        """
        上方法的英文版
        Parameters
        ----------
        momentum_dispersion 动量分散
        kinetic_energy_MeV 动能_MeV

        Returns convert动量分散_TO_能量分散
        -------

        """
        k = (kinetic_energy_MeV + cls.STATIC_ENERGY_MeV) / (
            kinetic_energy_MeV + 2 * cls.STATIC_ENERGY_MeV
        )

        return momentum_dispersion / k

    # @classmethod
    # def convert能量分散_TO_动量分散(cls, 能量分散: float, 动能_MeV: float) -> float:
    #     k = (动能_MeV + cls.STATIC_ENERGY_MeV) / \
    #         (动能_MeV + 2 * cls.STATIC_ENERGY_MeV)
    #     return 能量分散 * k

    @classmethod
    def convert_energy_dispersion_to_momentum_dispersion(
            cls, energyDispersion: float, kineticEnergy_MeV: float
    ) -> float:
        """
        上方法的英文版
        Parameters
        ----------
        energyDispersion 能量分散
        kineticEnergy_MeV 动能，典型值 250

        Returns 动量分散
        -------

        """
        k = (kineticEnergy_MeV + cls.STATIC_ENERGY_MeV) / (
            kineticEnergy_MeV + 2 * cls.STATIC_ENERGY_MeV
        )
        return energyDispersion * k


class RunningParticle:
    """
    在全局坐标系中运动的一个粒子
    position 位置，三维矢量，单位 [m, m, m]
    velocity 速度，三位矢量，单位 [m/s, m/s, m/s]
    relativistic_mass 相对论质量，又称为动质量，单位 kg， M=Mo/√(1-v^2/c^2)
    e 电荷量，单位 C 库伦
    speed 速率，单位 m/s
    distance 运动距离，单位 m
    """

    def __init__(
            self,
            position: P3,
            velocity: P3,
            relativistic_mass: float,
            e: float,
            speed: float,
            distance: float = 0.0,
    ):
        """
        在全局坐标系中运动的一个粒子
        Parameters
        ----------
        position 位置，三维矢量，单位 [m, m, m]
        velocity 速度，三位矢量，单位 [m/s, m/s, m/s]
        relativistic_mass 相对论质量，又称为动质量，单位 kg， M=Mo/√(1-v^2/c^2)
        e 电荷量，单位 C 库伦
        speed 速率，单位 m/s
        distance 运动距离，单位 m

        注意：不应使用这个构造器来创建粒子，因为内部不检查 velocity 和 speed 的一致性
        应该使用  ParticleFactory 来创建粒子，ParticleFactory 提供了丰富了创建粒子、粒子束的函数
        """
        self.position = position
        self.velocity = velocity
        self.relativistic_mass = relativistic_mass
        self.e = e
        self.speed = speed
        self.distance = distance

    def run_self_in_magnetic_field(
            self, magnetic_field: P3, footstep: float = 1 * MM
    ) -> None:
        """
        粒子在磁场 magnetic_field 中运动 footstep 长度
        Parameters
        ----------
        magnetic_field 磁场，看作恒定场
        footstep 步长，默认 1 MM

        Returns None
        -------
        """
        warnings.warn(
            "run_self_in_magnetic_field 函数已经废弃，因为没有使用 Runge-Kutta 数值积分方法，误差过大", DeprecationWarning)
        # 计算受力 qvb
        f = (self.velocity @ magnetic_field) * self.e
        # 计算加速度 a = f/m
        a = f / self.relativistic_mass
        # 计算运动时间
        t = footstep / self.speed
        # 位置变化
        self.position += self.velocity * t
        # 速度变化
        self.velocity += t * a
        # 运动距离
        self.distance += footstep

    def copy(self) -> "RunningParticle":
        """
        深拷贝粒子
        Returns 深拷贝粒子
        -------

        """
        return RunningParticle(
            self.position.copy(),
            self.velocity.copy(),
            self.relativistic_mass,
            self.e,
            self.speed,
            self.distance,
        )

    def compute_scalar_momentum(self) -> float:
        """
        获得标量动量
        Returns 标量动量
        -------

        """
        return self.speed * self.relativistic_mass

    def change_scalar_momentum(self, scalar_momentum: float) -> None:
        """
        改变粒子的标量动量。
        注意：真正改变的是粒子的速度和动质量
        这个方法用于生成一组动量分散的粒子

        scalar_momentum 标量动量
        Returns None
        -------

        """
        # 先求 静止质量
        m0 = self.relativistic_mass * math.sqrt(
            1 - (self.speed ** 2) / (LIGHT_SPEED ** 2)
        )
        # 求新的速率
        new_speed = scalar_momentum / math.sqrt(
            m0 ** 2 + (scalar_momentum / LIGHT_SPEED) ** 2
        )
        # 求新的动质量
        new_relativistic_mass = m0 / \
            math.sqrt(1 - (new_speed / LIGHT_SPEED) ** 2)
        # 求新的速度
        new_velocity: P3 = self.velocity.change_length(new_speed)

        # 写入
        self.relativistic_mass = new_relativistic_mass
        self.speed = new_speed
        self.velocity = new_velocity

        # 验证
        BaseUtils.equal(
            scalar_momentum,
            self.compute_scalar_momentum(),
            msg=f"RunningParticle::change_scalar_momentum异常，scalar_momentum{scalar_momentum}!=self.compute_scalar_momentum{self.compute_scalar_momentum}",
            err=1e-6,
        )

        BaseUtils.equal(
            self.speed,
            self.velocity.length(),
            msg=f"RunningParticle::change_scalar_momentum异常,self.speed{self.speed}!=Vectors.length(self.velocity){self.velocity.length()}",
        )

    def get_natural_coordinate_system(
            self, y_direction: P3 = P3.z_direct()
    ) -> LocalCoordinateSystem:
        """
        以粒子 self 构建坐标系，其中 z 轴方向即粒子速度方向，另外指定 y 轴方向
        x 轴方向由 y 轴和 z 轴方向确定
        这个方法只用于相空间和实际三维空间转换
        理想粒子的 natural_coordinate_system 即用来确定其他粒子的相空间坐标
        """
        return LocalCoordinateSystem.create_by_y_and_z_direction(
            self.position, y_direction, self.velocity
        )

    def __str__(self) -> str:
        return f"p={self.position},v={self.velocity},v0={self.speed}"

    def __repr__(self) -> str:
        return self.__str__()

    def to_numpy_array_data(self, numpy_dtype=numpy.float64) -> numpy.ndarray:
        """
        RunningParticle 转为 numpy_array_data
        主要用于 GPU 加速
        numpy_array_data 是一个一维数组，分别是 
        (px0, py1, pz2, vx3, vy4, vz5, rm6, e7, speed8, distance9) len = 10

        since v0.1.1
        """
        data_list: List[float] = (
            self.position.to_list() +
            self.velocity.to_list() +
            [self.relativistic_mass, self.e, self.speed, self.distance]
        )
        return numpy.array(data_list, dtype=numpy_dtype)

    def detailed_info(self) -> str:
        return f"Particle[p={self.position}, v={self.velocity}], rm={self.relativistic_mass}, e={self.e}, speed={self.speed}, distance={self.distance}]"

    @staticmethod
    def from_numpy_array_data(numpy_array) -> 'RunningParticle':
        """
        上函数的逆函数
        see to_numpy_array_data
        since v0.1.1
        """
        pos = P3(numpy_array[0], numpy_array[1], numpy_array[2])
        vel = P3(numpy_array[3], numpy_array[4], numpy_array[5])

        return RunningParticle(
            position=pos,
            velocity=vel,
            relativistic_mass=numpy_array[6],
            e=numpy_array[7],
            speed=numpy_array[8],
            distance=numpy_array[9]
        )

    def populate(self, other: 'RunningParticle') -> None:
        """
        将 other 的值赋到 self 中
        since v0.1.1
        """
        self.position.populate(other.position)
        self.velocity.populate(other.velocity)
        self.relativistic_mass = other.relativistic_mass
        self.e = other.e
        self.speed = other.speed
        self.distance = other.distance

    def __sub__(self, other: 'RunningParticle') -> "RunningParticle":
        """
        粒子"减法" 只用来显示两个粒子的差异
        一般用于 debug
        since v0.1.1
        """
        return RunningParticle(
            position=self.position - other.position,
            velocity=self.velocity - other.velocity,
            relativistic_mass=self.relativistic_mass - other.relativistic_mass,
            e=self.e - other.e,
            speed=self.speed - other.speed,
            distance=self.distance - other.distance,
        )


class ParticleRunner:
    """
    粒子运动工具类
    """

    @staticmethod
    def __callback_for_runge_kutta4(particle: RunningParticle, magnet: Magnet) -> Callable[
            [float, numpy.ndarray], numpy.ndarray]:
        """
        将二阶微分方程转为一阶
        Y = [p,v]
        Y'= [v,a]
        see BaseUtils.runge_kutta4()
        since v0.1.1
        """
        k: float = particle.e / particle.relativistic_mass

        def callback(t: float, Y: numpy.ndarray) -> numpy.ndarray:
            # 闭包
            # nonlocal k, magnet
            position: P3 = Y[0]
            velocity: P3 = Y[1]

            accelerate: P3 = k * \
                (velocity @ magnet.magnetic_field_at(position))

            return numpy.array([velocity, accelerate])

        return callback

    @staticmethod
    def __callback_for_solve_ode(particle: RunningParticle, magnet: Magnet) -> Callable[
            [float, numpy.ndarray], numpy.ndarray]:
        """
        see BaseUtils.solve_ode()
        since v0.1.1
        """
        k: float = particle.e / particle.relativistic_mass

        def callback(t: float, Y: numpy.ndarray) -> numpy.ndarray:
            # 闭包
            # nonlocal k, magnet
            position: P3 = P3(Y[0], Y[1], Y[2])
            velocity: P3 = P3(Y[3], Y[4], Y[5])

            accelerate: P3 = k * \
                (velocity @ magnet.magnetic_field_at(position))

            return numpy.array([velocity.x, velocity.y, velocity.z, accelerate.x, accelerate.y, accelerate.z])

        return callback

    @staticmethod
    def run_only(
            p: Union[RunningParticle, List[RunningParticle]],
            m: Magnet,
            length: float,
            footstep: float = 20 * MM,
            concurrency_level: int = 1,
            report: bool = True
    ) -> Union[RunningParticle, List[RunningParticle]]:
        """
        让粒子 p 在磁场 m 中运动 length 距离，步长 footstep
        Parameters
        ----------
        p 粒子
        m 磁场
        length 运动长度
        footstep 步长
        concurrency_level 并行度
        report 是否打印并行任务计划

        Returns None
        -------
        refactor v0.1.1 使用 runge kutta 和 加入多进程支持
        """
        if isinstance(p, RunningParticle):
            dt = footstep / p.speed
            t_end = length / p.speed
            Y0 = numpy.array([p.position, p.velocity])
            func = ParticleRunner.__callback_for_runge_kutta4(
                particle=p, magnet=m)
            Y1 = BaseUtils.runge_kutta4(
                t0=0.0, t_end=t_end, Y0=Y0, y_derived_function=func, dt=dt, record=False)
            p.position = Y1[0]
            p.velocity = Y1[1]
            p.distance += length
            return p
        elif concurrency_level == 1:
            particle_number = len(p)
            print(f"track {particle_number} particles")
            print("当前使用单线程进行粒子跟踪，如果函数支持多线程并行，推荐使用多线程")
            particle_index = 0
            start_time = time.time()
            for this_p in p:
                ParticleRunner.run_only(this_p, m, length, footstep)
                particle_index += 1
                if particle_index == 1:
                    time_run_one_particle = time.time() - start_time
                    print(
                        f"运行一个粒子需要{time_run_one_particle:.5f}秒，估计总耗时{time_run_one_particle * particle_number:.5f}秒")
                print(
                    '\b'*8 + f'{(particle_index / particle_number * 100):>6.2f}% ', end='', flush=True)
            print(' finished')
            print(f"实际用时{(time.time()-start_time):.5f}秒")
            return p
        else:
            results: List[RunningParticle] = BaseUtils.submit_process_task(
                task=ParticleRunner.run_only,
                param_list=[
                    [this_p, m, length, footstep] for this_p in p
                ],
                concurrency_level=concurrency_level,
                report=report
            )
            particle_number = len(p)
            for i in range(particle_number):
                p[i].position = results[i].position
                p[i].velocity = results[i].velocity
                p[i].distance = results[i].distance

            return p

    @staticmethod
    def run_only_ode(
            p: Union[RunningParticle, List[RunningParticle]],
            m: Magnet,
            length: float,
            footstep: float = 20 * MM,
            absolute_tolerance: float = 1e-8,
            relative_tolerance: float = 1e-8
    ) -> None:
        """
        让粒子 p 在磁场 m 中运动 length 距离，步长 footstep
        使用 scipy 提供的 ode 法
        Parameters
        ----------
        p 粒子
        m 磁场
        length 运动长度
        footstep 步长

        Returns None
        -------
        refactor v0.1.1 ode45
        """
        if isinstance(p, RunningParticle):
            dt = footstep / p.speed
            t_end = length / p.speed
            Y0 = numpy.array([p.position.x, p.position.y, p.position.z,
                              p.velocity.x, p.velocity.y, p.velocity.z])
            func = ParticleRunner.__callback_for_solve_ode(
                particle=p, magnet=m)
            Y1 = BaseUtils.solve_ode(
                t0=0.0, t_end=t_end, Y0=Y0, y_derived_function=func, dt=dt, record=False,
                absolute_tolerance=absolute_tolerance, relative_tolerance=relative_tolerance)
            p.position = P3(Y1[0][-1], Y1[1][-1], Y1[2][-1])
            p.velocity = P3(Y1[3][-1], Y1[4][-1], Y1[5][-1])
            p.distance += length
            return None
        else:
            particle_number = len(p)
            print(f"track {particle_number} particles")
            for this_p in p:
                print('▇', end='', flush=True)
                ParticleRunner.run_only_ode(this_p, m, length, footstep,
                                            absolute_tolerance, relative_tolerance)
            print(' finished')

    @staticmethod
    def run_get_trajectory(
            p: RunningParticle, m: Magnet, length: float, footstep: float = 20 * MM
    ) -> List[P3]:
        """
        让粒子 p 在磁场 m 中运动 length 距离，步长 footstep
        获得粒子的轨迹
        Parameters
        ----------
        p 粒子
        m 磁场
        length 运动长度
        footstep 步长

        Returns 轨迹 np.ndarray，是三维点的数组
        -------
        refactor v0.1.1 runge kutta
        """
        dt = footstep / p.speed
        t_end = length / p.speed
        Y0 = numpy.array([p.position, p.velocity])
        func = ParticleRunner.__callback_for_runge_kutta4(
            particle=p, magnet=m)
        _, Ys = BaseUtils.runge_kutta4(
            t0=0.0, t_end=t_end, Y0=Y0, y_derived_function=func, dt=dt, record=True)
        p.distance += length

        trajectory: List[P3] = []
        for y in Ys:
            trajectory.append(y[0])

        return trajectory

    @staticmethod
    def run_get_all_info(
            p: Union[RunningParticle, List[RunningParticle]],
            m: Magnet, length: float, footstep: float = 1 * MM,
            concurrency_level: Optional[int] = None, report: bool = False
    ) -> Union[List[RunningParticle], List[List[RunningParticle]]]:
        """
        让粒子 p 在磁场 m 中运动 length 距离，步长 footstep
        获得粒子全部信息
        Parameters
        ----------
        p 粒子
        m 磁场
        length 运动长度
        footstep 步长

        Returns 每一步处的粒子全部信息 List[RunningParticle]
        -------
        refactor v0.1.1 runge kutta
        """
        if isinstance(p, RunningParticle):
            distance0 = p.distance

            dt = footstep / p.speed
            t_end = length / p.speed
            Y0 = numpy.array([p.position, p.velocity])
            func = ParticleRunner.__callback_for_runge_kutta4(
                particle=p, magnet=m)
            ts, Ys = BaseUtils.runge_kutta4(
                t0=0.0, t_end=t_end, Y0=Y0, y_derived_function=func, dt=dt, record=True)
            p.distance += length

            all_info: List[RunningParticle] = []

            for i in range(len(ts)):
                t: float = ts[i]
                pos: P3 = Ys[i][0]
                vel: P3 = Ys[i][1]

                this_p = p.copy()
                this_p.position = pos
                this_p.velocity = vel
                this_p.distance = distance0 + t * this_p.speed

                all_info.append(this_p)

            return all_info
        else:  # p is list[p]
            if concurrency_level is None:
                concurrency_level = os.cpu_count()
            if concurrency_level == 1:
                returns: List[List[RunningParticle]] = []
                for each_p in p:
                    returns.append(ParticleRunner.run_get_all_info(
                        p=each_p, m=m, length=length, footstep=footstep
                    ))
                return returns
            else:
                return BaseUtils.submit_process_task(
                    task=ParticleRunner.run_get_all_info,
                    param_list=[
                        [this_p, m, length, footstep] for this_p in p
                    ],
                    concurrency_level=concurrency_level,
                    report=report
                )

    @staticmethod
    def run_only_deprecated(
            p: RunningParticle, m: Magnet, length: float, footstep: float = 1 * MM
    ) -> None:
        """
        让粒子 p 在磁场 m 中运动 length 距离，步长 footstep
        Parameters
        ----------
        p 粒子
        m 磁场
        length 运动长度
        footstep 步长

        Returns None
        -------
        refactor v0.1.1 保存过时方法
        """
        warnings.warn(
            "run_only_deprecated 已过时，因为没有使用 Runge-Kutta 数值积分方法，误差过大", category=DeprecationWarning)
        distance = 0.0
        while distance < length:
            p.run_self_in_magnetic_field(
                m.magnetic_field_at(p.position), footstep=footstep
            )
            distance += footstep

    @staticmethod
    def run_get_trajectory_deprecated(
            p: RunningParticle, m: Magnet, length: float, footstep: float = 1 * MM
    ) -> List[P3]:
        """
        让粒子 p 在磁场 m 中运动 length 距离，步长 footstep
        获得粒子的轨迹
        Parameters
        ----------
        p 粒子
        m 磁场
        length 运动长度
        footstep 步长

        Returns 轨迹 np.ndarray，是三维点的数组
        -------
        refactor v0.1.1 保存过时方法
        """
        warnings.warn(
            "run_get_trajectory_deprecated 已过时，因为没有使用 Runge-Kutta 数值积分方法，误差过大", category=DeprecationWarning)
        trajectory: List[P3] = [p.position.copy()]

        i = 1
        distance = 0.0
        while distance < length:
            p.run_self_in_magnetic_field(
                m.magnetic_field_at(p.position), footstep=footstep
            )
            distance += footstep
            trajectory.append(p.position.copy())
            i += 1

        return trajectory

    @staticmethod
    def run_get_all_info_deprecated(
            p: RunningParticle, m: Magnet, length: float, footstep: float = 1 * MM
    ) -> List[RunningParticle]:
        """
        让粒子 p 在磁场 m 中运动 length 距离，步长 footstep
        获得粒子全部信息
        Parameters
        ----------
        p 粒子
        m 磁场
        length 运动长度
        footstep 步长

        Returns 每一步处的粒子全部信息 List[RunningParticle]
        -------
        refactor v0.1.1 保存过时方法
        """
        warnings.warn(
            "run_get_all_info_deprecated 已过时，因为没有使用 Runge-Kutta 数值积分方法，误差过大", category=DeprecationWarning)

        all_info: List[RunningParticle] = [p.copy()]
        distance = 0.0
        while distance < length:
            p.run_self_in_magnetic_field(
                m.magnetic_field_at(p.position), footstep=footstep
            )
            distance += footstep
            all_info.append(p.copy())

        return all_info


class PhaseSpaceParticle:
    XXP_PLANE = 1
    YYP_PLANE = 2

    """
    相空间中的粒子，6个坐标 x xp y yp z delta
    """

    def __init__(
            self, x: float = 0.0, xp: float = 0.0, y: float = 0.0, yp: float = 0.0, z: float = 0.0, delta: float = 0.0
    ):
        self.x = x
        self.xp = xp
        self.y = y
        self.yp = yp
        self.z = z
        self.delta = delta

    def project_to_xxp_plane(self, convert_to_mm: bool = False) -> P2:
        """
        投影到 x-xp 平面
        Returns [self.x, self.xp]

        refactor v0.1.2 convert_to_mm 单位转换
        -------

        """
        return P2(self.x * (1000 if convert_to_mm else 1), self.xp * (1000 if convert_to_mm else 1))

    def project_to_yyp_plane(self, convert_to_mm: bool = False) -> P2:
        """
        投影到 y-yp 平面
        Returns [self.y, self.yp]

        refactor v0.1.2 convert_to_mm 单位转换
        -------

        """
        return P2(self.y * (1000 if convert_to_mm else 1), self.yp * (1000 if convert_to_mm else 1))

    def project_to_plane(self, plane_id: int, convert_to_mm: bool = False) -> P2:
        """
        refactor v0.1.2 convert_to_mm 单位转换
        """
        if plane_id == PhaseSpaceParticle.XXP_PLANE:
            return self.project_to_xxp_plane(convert_to_mm=convert_to_mm)
        elif plane_id == PhaseSpaceParticle.YYP_PLANE:
            return self.project_to_yyp_plane(convert_to_mm=convert_to_mm)
        else:
            raise ValueError(f"没有处理plane_id({plane_id})的方法")

    @staticmethod
    def phase_space_particles_along_positive_ellipse_in_xxp_plane(
            xMax: float, xpMax: float, delta: float, number: int
    ) -> List["PhaseSpaceParticle"]:
        """
        获取分布于 x xp 平面上 正相椭圆上的 PhaseSpaceParticles
        注意是 正相椭圆
        Parameters
        ----------
        xMax 相椭圆参数 x 最大值
        xpMax 相椭圆参数 xp 最大值
        delta 动量分散
        number 粒子数目

        Returns 分布于 x xp 平面上 正相椭圆上的 PhaseSpaceParticles
        -------

        """
        A: float = 1 / (xMax ** 2)
        B: float = 0
        C: float = 1 / (xpMax ** 2)
        D: float = 1

        return [
            PhaseSpaceParticle(p.x, p.y, 0, 0, 0, delta)
            for p in BaseUtils.Ellipse(
                A, B, C, D
            ).uniform_distribution_points_along_edge(number)
        ]

    @staticmethod
    def phase_space_particles_along_positive_ellipse_in_yyp_plane(
            yMax: float, ypMax: float, delta: float, number: int
    ) -> List["PhaseSpaceParticle"]:
        """
        获取分布于 y yp 平面上 正相椭圆上的 PhaseSpaceParticles
        注意是 正相椭圆
        Parameters
        ----------
        yMax 相椭圆参数 y 最大值
        ypMax 相椭圆参数 yp 最大值
        delta 动量分散
        number 粒子数目

        Returns 分布于 y yp 平面上 正相椭圆上的 PhaseSpaceParticles
        -------

        """
        A: float = 1 / (yMax ** 2)
        B: float = 0
        C: float = 1 / (ypMax ** 2)
        D: float = 1

        return [
            PhaseSpaceParticle(0, 0, p.x, p.y, 0, delta)
            for p in BaseUtils.Ellipse(
                A, B, C, D
            ).uniform_distribution_points_along_edge(number)
        ]

    @staticmethod
    def phase_space_particles_along_positive_ellipse_in_plane(
            plane_id: int, xMax: float, xpMax: float, delta: float, number: int
    ) -> List["PhaseSpaceParticle"]:
        """
        获取分布于 x xp 平面上或 y yp 平面上的，正相椭圆上的 PhaseSpaceParticles
        Parameters
        ----------
        xxPlane x 平面或 y 平面，true：x 平面，false:y 平面
        xMax 相椭圆参数 x/y 最大值
        xpMax 相椭圆参数 xp/yp 最大值
        delta 动量分散
        number 粒子数目

        Returns 分布于 x xp 平面上或 y yp 平面上的，正相椭圆上的 PhaseSpaceParticles
        -------

        """
        if plane_id == PhaseSpaceParticle.XXP_PLANE:
            return PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_xxp_plane(
                xMax, xpMax, delta, number
            )
        elif plane_id == PhaseSpaceParticle.YYP_PLANE:
            return PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_yyp_plane(
                xMax, xpMax, delta, number
            )
        else:
            raise ValueError(f"没有处理plane_id({plane_id})的方法")

    @staticmethod
    def phase_space_particles_project_to_xxp_plane(
            phase_space_particles: List, convert_to_mm: bool = False
    ) -> List[P2]:
        """
        相空间粒子群投影到 x 平面
        Parameters
        ----------
        phase_space_particles 相空间粒子群

        Returns 相空间粒子群投影到 x 平面 [[x1,xp1], [x2,xp2] .. ]

        refactor v0.1.2 convert_to_mm 单位转换
        -------

        """
        return [p.project_to_xxp_plane(convert_to_mm=convert_to_mm) for p in phase_space_particles]

    @staticmethod
    def phase_space_particles_project_to_yyp_plane(
            phase_space_particles: List, convert_to_mm: bool = False
    ) -> List[P2]:
        """
        相空间粒子群投影到 y 平面
        Parameters
        ----------
        phase_space_particles 相空间粒子群

        Returns 相空间粒子群投影到 y 平面 [[y1,yp1], [y2,yp2] .. ]

        refactor v0.1.2 convert_to_mm 单位转换
        -------

        """
        return [p.project_to_yyp_plane(convert_to_mm=convert_to_mm) for p in phase_space_particles]

    @staticmethod
    def phase_space_particles_project_to_plane(
            phase_space_particles: List, plane_id: int, convert_to_mm: bool = False
    ) -> List[P2]:
        """
        相空间粒子群投影到 x/y 平面
        Parameters
        ----------
        phase_space_particles 相空间粒子群
        plane_id 投影到 x 或 y 平面

        Returns 相空间粒子群投影到 x/y 平面

        refactor v0.1.2 convert_to_mm 单位转换
        -------

        """
        if plane_id == PhaseSpaceParticle.XXP_PLANE:
            return PhaseSpaceParticle.phase_space_particles_project_to_xxp_plane(
                phase_space_particles, convert_to_mm=convert_to_mm
            )
        elif plane_id == PhaseSpaceParticle.YYP_PLANE:
            return PhaseSpaceParticle.phase_space_particles_project_to_yyp_plane(
                phase_space_particles, convert_to_mm=convert_to_mm
            )
        else:
            raise ValueError(f"没有处理plane_id({plane_id})的方法")

    @staticmethod
    def create_from_running_particle(
            ideal_particle: RunningParticle,
            coordinate_system: LocalCoordinateSystem,
            running_particle: RunningParticle,
    ) -> "PhaseSpaceParticle":
        """
        将实际粒子 running_particle 映射为 PhaseSpaceParticle
        这需要一个理想粒子/参考粒子 ideal_particle
        和一个参考粒子的自然坐标系 coordinate_system
        """
        # x y z
        relative_position = running_particle.position - ideal_particle.position
        x = coordinate_system.XI * relative_position
        y = coordinate_system.YI * relative_position
        z = coordinate_system.ZI * relative_position

        # xp yp
        relative_velocity = running_particle.velocity - ideal_particle.velocity
        # xp = (coordinate_system.XI * relative_velocity) / ideal_particle.speed
        # yp = (coordinate_system.YI * relative_velocity) / ideal_particle.speed

        # xp yp 就是求角度，所以修改代码如下
        # 修改于 2021年5月1日
        try:
            xp = (coordinate_system.XI * relative_velocity) / (
                math.sqrt(running_particle.speed**2 -
                          (coordinate_system.XI * relative_velocity)**2)
            )
            yp = (coordinate_system.YI * relative_velocity) / (
                math.sqrt(running_particle.speed**2 -
                          (coordinate_system.YI * relative_velocity)**2)
            )
        except Exception as e:
            print(f"异常{e}")
            print(f"ip={ideal_particle}")
            print(f"rp={running_particle}")
            print(f"lcs={coordinate_system}")

        # delta
        rm = running_particle.compute_scalar_momentum()
        im = ideal_particle.compute_scalar_momentum()
        delta = (rm - im) / im

        return PhaseSpaceParticle(x, xp, y, yp, z, delta)

    @staticmethod
    def create_from_running_particles(
            ideal_particle: RunningParticle,
            coordinate_system: LocalCoordinateSystem,
            running_particles: List[RunningParticle],
    ) -> List["PhaseSpaceParticle"]:
        """
        将多个实际粒子 running_particles 映射为 PhaseSpaceParticles
        参数意义见上函数 create_from_running_particle
        """
        return [
            PhaseSpaceParticle.create_from_running_particle(
                ideal_particle, coordinate_system, rp
            )
            for rp in running_particles
        ]

    @staticmethod
    def convert_delta_from_momentum_dispersion_to_energy_dispersion(
            phaseSpaceParticle: "PhaseSpaceParticle", centerKineticEnergy_MeV
    ) -> "PhaseSpaceParticle":
        """
        动量分散改动能分散
        Parameters
        ----------
        phaseSpaceParticle 原粒子
        centerKineticEnergy_MeV 中心动能，如 250

        Returns 动量分散改动能分散后的粒子
        -------

        """
        copied: PhaseSpaceParticle = phaseSpaceParticle.copy()
        deltaMomentumDispersion = copied.delta
        deltaEnergyDispersion = (
            Protons.convert_momentum_dispersion_to_energy_dispersion(
                deltaMomentumDispersion, centerKineticEnergy_MeV
            )
        )

        copied.delta = deltaEnergyDispersion

        return copied

    @staticmethod
    def convert_delta_from_momentum_dispersion_to_energy_dispersion_for_list(
            phaseSpaceParticles: List["PhaseSpaceParticle"], centerKineticEnergy_MeV
    ) -> List["PhaseSpaceParticle"]:
        """
        动量分散改动能分散，见上方法 convert_delta_from_momentum_dispersion_to_energy_dispersion
        Parameters
        ----------
        phaseSpaceParticles
        centerKineticEnergy_MeV

        Returns
        -------

        """
        return [
            PhaseSpaceParticle.convert_delta_from_momentum_dispersion_to_energy_dispersion(
                pp, centerKineticEnergy_MeV
            )
            for pp in phaseSpaceParticles
        ]

    @staticmethod
    def convert_delta_from_energy_dispersion_to_momentum_dispersion(
            phaseSpaceParticle: "PhaseSpaceParticle", centerKineticEnergy_MeV: float
    ) -> "PhaseSpaceParticle":
        """
        将相空间粒子 phaseSpaceParticle 中 delta 从能量分散转为动量分散
        centerKineticEnergy_MeV 中心动能
        """
        copied = phaseSpaceParticle.copy()

        EnergyDispersion = copied.getDelta()

        MomentumDispersion = Protons.convert_energy_dispersion_to_momentum_dispersion(
            EnergyDispersion, centerKineticEnergy_MeV
        )

        copied.delta = MomentumDispersion

        return copied

    @staticmethod
    def convert_delta_from_energy_dispersion_to_momentum_dispersion_for_list(
            phaseSpaceParticles: List["PhaseSpaceParticle"], centerKineticEnergy_MeV: float
    ) -> List["PhaseSpaceParticle"]:
        """
        将多个相空间粒子 phaseSpaceParticles 中 delta 从能量分散转为动量分散
        centerKineticEnergy_MeV 中心动能
        """
        return [
            PhaseSpaceParticle.convert_delta_from_energy_dispersion_to_momentum_dispersion(
                pp, centerKineticEnergy_MeV
            )
            for pp in phaseSpaceParticles
        ]

    def __str__(self) -> str:
        return (
            f"x={self.x},xp={self.xp},y={self.y},yp={self.yp},z={self.z},d={self.delta}"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self) -> "PhaseSpaceParticle":
        """
        PhaseSpaceParticle 深拷贝
        """
        return PhaseSpaceParticle(self.x, self.xp, self.y, self.yp, self.z, self.delta)

    def getDelta(self) -> float:
        """
        返回动量/能量分散 delta

        since v0.1.1
        """
        return self.delta

    def get_length(self) -> float:
        """
        求像空间中，粒子到原点（理想粒子的距离）

        since 2021年6月27日
        """
        return math.sqrt(
            self.x**2 + 
            self.xp**2 + 
            self.y**2 + 
            self.yp**2 + 
            self.z**2 + 
            self.delta**2 
        )

    def dominate(self, other: 'PhaseSpaceParticle') -> bool:
        """
        p1.dominate(p2) 判断 p1 是否支配 p2
        支配的含义是 p1 的每个值都不小于 p2

        since 2021年6月27日
        """


        count = 0

        if self.x > other.x :
            count += 1
        elif self.x < other.x:
            return False

        if self.xp > other.xp :
            count += 1
        elif self.xp < other.xp:
            return False

        if self.y > other.y :
            count += 1
        elif self.y < other.y:
            return False

        if self.yp > other.yp :
            count += 1
        elif self.yp < other.yp:
            return False

        if self.z > other.z :
            count += 1
        elif self.z < other.z:
            return False

        if self.delta > other.delta :
            count += 1
        elif self.delta < other.delta:
            return False

        return count != 0


        

        # 错误写法
        # return (
        #     self.x >= other.x and 
        #     self.xp >= other.xp and 
        #     self.y >= other.y and 
        #     self.yp >= other.yp and 
        #     self.z >= other.z and 
        #     self.delta >= other.delta
        # )


class ParticleFactory:
    """
    质子工厂
    提供了方便的构造质子/质子群的函数
    """

    @staticmethod
    def create_proton(
            position: P3, direct: P3, kinetic_MeV: float = 250
    ) -> RunningParticle:
        """
        生成一个质子（即 RunningParticle 对象），
        位置为 position，
        运动方向为 direct，
        动能为 kinetic_MeV
        """
        # 速率
        speed = LIGHT_SPEED * math.sqrt(
            1.0
            - (Protons.STATIC_ENERGY_MeV /
               (Protons.STATIC_ENERGY_MeV + kinetic_MeV))
            ** 2
        )

        # mass kg
        relativistic_mass = Protons.STATIC_MASS_KG / math.sqrt(
            1.0 - (speed ** 2) / (LIGHT_SPEED ** 2)
        )

        return RunningParticle(
            position,
            direct.copy().change_length(speed),
            relativistic_mass,
            Protons.CHARGE_QUANTITY,
            speed,
        )

    @staticmethod
    def create_proton_by_position_and_velocity(
            position: P3, velocity: P3
    ) -> RunningParticle:
        """
        生成一个质子，
        位置为 position，
        速度为 velocity
        """
        speed = velocity.length()

        relativistic_mass = 0.0

        try:
            relativistic_mass = Protons.STATIC_MASS_KG / math.sqrt(
                1.0 - (speed ** 2) / (LIGHT_SPEED ** 2)
            )
        except RuntimeWarning as e:
            print(
                f"ParticleFactory::create_proton_by_position_and_velocity 莫名其妙的异常 speed={speed} LIGHT_SPEED={LIGHT_SPEED} e={e}"
            )

        return RunningParticle(
            position, velocity, relativistic_mass, Protons.CHARGE_QUANTITY, speed
        )

    @staticmethod
    def create_proton_along(
            trajectory: Line2, s: float = 0.0, kinetic_MeV: float = 250
    ) -> RunningParticle:
        """
        生成一个沿着设计轨道 trajectory 的质子，
        位于轨道 s 位置，
        动能为 kinetic_MeV。
        这个函数一般用于生成参考粒子
        """
        return ParticleFactory.create_proton(
            trajectory.point_at(s).to_p3(),
            trajectory.direct_at(s).to_p3(),
            kinetic_MeV=kinetic_MeV,
        )

    @staticmethod
    def create_from_phase_space_particle(
            ideal_particle: RunningParticle,
            coordinate_system: LocalCoordinateSystem,
            phase_space_particle: PhaseSpaceParticle,
    ) -> RunningParticle:
        """
        将相空间粒子 phase_space_particle 映射为实际粒子（质子）
        Parameters
        ----------
        ideal_particle 理想粒子
        coordinate_system 相空间坐标系
        phase_space_particle 相空间粒子

        Returns 通过理想粒子，相空间坐标系 和 相空间粒子，来创造粒子
        -------

        """
        x = phase_space_particle.x
        xp = phase_space_particle.xp
        y = phase_space_particle.y
        yp = phase_space_particle.yp
        z = phase_space_particle.z
        delta = phase_space_particle.delta

        p = ideal_particle.copy()
        # 知道 LocalCoordinateSystem 的用处了吧
        p.position += coordinate_system.XI * x
        p.position += coordinate_system.YI * y
        p.position += coordinate_system.ZI * z

        if delta != 0.0:
            scalar_momentum = p.compute_scalar_momentum() * (1.0 + delta)
            p.change_scalar_momentum(scalar_momentum)  # 这个方法就是为了修改动量而写的

        p.velocity += coordinate_system.XI * (xp * p.speed)
        p.velocity += coordinate_system.YI * (yp * p.speed)

        p.velocity = p.velocity.change_length(p.speed)

        return p

    @staticmethod
    def create_from_phase_space_particles(
            ideal_particle: RunningParticle,
            coordinate_system: LocalCoordinateSystem,
            phase_space_particles: List[PhaseSpaceParticle],
    ) -> List[RunningParticle]:
        """
        将多个相空间粒子 phase_space_particle 映射为实际粒子（质子）
        详见上函数 create_from_phase_space_particle
        """
        return [
            ParticleFactory.create_from_phase_space_particle(
                ideal_particle, coordinate_system, p
            )
            for p in phase_space_particles
        ]

    DISTRIBUTION_AREA_EDGE = 1
    DISTRIBUTION_AREA_FULL = 2

    DISTRIBUTION_TYPE_GAUSS = "gauss"
    DISTRIBUTION_TYPE_UNIFORM = "uniform"

    @classmethod
    def distributed_particles(cls, x: float, xp: float, y: float, yp: float, delta: float, number: int,
                              distribution_area: int,
                              x_distributed: bool = False, xp_distributed: bool = False,
                              y_distributed: bool = False, yp_distributed: bool = False,
                              delta_distributed: bool = False,
                              distribution_type="uniform") -> List[PhaseSpaceParticle]:
        """
        随机产生某种分布的质子集合，即 PhaseSpaceParticle 数组

        仅支持正相椭圆/正相椭球分布，即不支持有倾斜角的相椭圆/相椭球

        相椭圆参数由 5 个轴给出，即 x xp y yp delta，例如 3.5mm 7.5mr 3.5mm 7.5mr 0.08

        number 指定生成的粒子数目

        distribution_area 指定粒子的分布区域，有边缘分布 DISTRIBUTION_AREA_EDGE 和全分布 DISTRIBUTION_AREA_FULL 两种
            边缘分布指的是，粒子位于相椭圆的圆周 或者 位于相椭球的表面
              全分布指的是，粒子位于相椭圆的内部 或者 位于相椭球的内部

        *_distributed 是一个布尔量，指定变量是否参于分布。默认不参与
            例如 x=xp=1，x_distributed=true，xp_distributed=false时，表示 x 参与分布，xp 不参与，
            即生成的粒子类似 (x=0.13,xp=1), (x=-0.79,xp=1), (x=0.45,xp=1)...
            再例如 x=xp=y=yp=delta=1，且 y_distributed，yp_distributed，delta_distributed 三个设为 true,
            则生成的粒子类似 (x=1,xp=1,y=0.3,yp=-0.5,delta=0.1) ...

            不参与分布的变量，输出即原值

        distribution_type 表示分布类型，当前仅支持均匀分布 uniform

        2021年2月26日 新增 gauss 高斯分布支持，gauss 下仅仅支持 DISTRIBUTION_AREA_FULL

        ----------------------
        使用示例
        束流参数为 x=y=3.5mm，xp,yp=7.5mr，dp=8%。生成粒子数目20

        1. 生成x/xp相椭圆圆周上，动量分散为0的粒子，
            ps = ParticleFactory.distributed_particles(
                3.5*MM, 7.5*MRAD, 3.5*MM, 7.5*MRAD, 0.0, 20,
                ParticleFactory.DISTRIBUTION_AREA_EDGE,
                x_distributed=True, xp_distributed=True
            )
        2. 生成y/yp相椭圆内部，动量分散均为0.05的粒子
            ps = ParticleFactory.distributed_particles(
                3.5*MM, 7.5*MRAD, 3.5*MM, 7.5*MRAD, 0.0, 20,
                ParticleFactory.DISTRIBUTION_AREA_FULL,
                y_distributed=True, yp_distributed=True
            )
        3. 生成 x/xp/delta 三维相椭球球面的粒子
            ps = ParticleFactory.distributed_particles(
                3.5*MM, 7.5*MRAD, 3.5*MM, 7.5*MRAD, 0.08, 20,
                ParticleFactory.DISTRIBUTION_AREA_EDGE,
                x_distributed=True, xp_distributed=True, delta_distributed=True
            )

        since v0.1.4
        """
        params = [x, xp, y, yp, delta]  # 全部参数
        distributed = [x_distributed, xp_distributed, y_distributed,
                       yp_distributed, delta_distributed]  # 是否参与分布
        variables = [params[i]
                     for i in range(len(params)) if distributed[i]]  # 参与分布的变量
        dim = len(variables)  # 分布维度

        if dim == 0:
            print("没有变量参与分布")
            return [PhaseSpaceParticle(x, xp, y, yp, 0, delta) for ignore in range(number)]

        distribution: List[List[float]] = None

        if distribution_type == cls.DISTRIBUTION_TYPE_UNIFORM:
            if distribution_area == cls.DISTRIBUTION_AREA_EDGE:
                # 边缘分布
                if dim == 1:
                    raise ValueError("一维下无边缘分布")
                elif dim == 2:
                    # 椭圆圆周
                    distribution = [
                        BaseUtils.Random.uniformly_distributed_along_elliptic_circumference(
                            variables[0], variables[1]).to_list()
                        for ignore in range(number)
                    ]
                elif dim == 3:
                    # 椭球表面
                    distribution = [
                        BaseUtils.Random.uniformly_distributed_at_ellipsoidal_surface(
                            variables[0], variables[1], variables[2]).to_list()
                        for ignore in range(number)
                    ]
                else:
                    # 超椭球表面
                    distribution = [
                        BaseUtils.Random.uniformly_distributed_at_hypereellipsoidal_surface(
                            variables)
                        for ignore in range(number)
                    ]
            elif distribution_area == cls.DISTRIBUTION_AREA_FULL:
                if dim == 1:
                    # 一维均匀分布
                    distribution = [
                        [random.uniform(-variables[0], variables[0])] for ignore in range(number)]
                elif dim == 2:
                    # 椭圆内分布
                    distribution = [
                        BaseUtils.Random.uniformly_distributed_in_ellipse(
                            variables[0], variables[1]).to_list()
                        for ignore in range(number)
                    ]
                elif dim == 3:
                    # 椭球内分布
                    distribution = [
                        BaseUtils.Random.uniformly_distributed_in_ellipsoid(
                            variables[0], variables[1], variables[2]).to_list()
                        for ignore in range(number)
                    ]
                else:
                    # 超椭球内分布
                    distribution = [
                        BaseUtils.Random.uniformly_distributed_in_hypereellipsoid(
                            variables)
                        for ignore in range(number)
                    ]
            else:
                raise ValueError("分布区域仅支持边缘分布和全分布")
        elif distribution_type == cls.DISTRIBUTION_TYPE_GAUSS:
            if distribution_area == cls.DISTRIBUTION_AREA_EDGE:
                raise ValueError("高斯分布下不支持边缘分布")
            elif distribution_area == cls.DISTRIBUTION_AREA_FULL:
                distribution = [
                    BaseUtils.Random.gauss_multi_dimension(
                        [0.0]*dim, variables)
                    for ignore in range(number)
                ]
            else:
                raise ValueError("分布区域仅支持边缘分布和全分布")
        else:
            raise ValueError("当前仅支持均匀分布(uniform)和高斯分布(gauss)")

        # distribution
        ps: List[PhaseSpaceParticle] = []
        for i in range(number):
            cur_params: List[float] = []
            distribution_index: int = 0
            for j in range(len(params)):
                if distributed[j]:
                    # 变量
                    cur_params.append(distribution[i][distribution_index])
                    distribution_index = distribution_index+1
                else:
                    # 常量
                    cur_params.append(params[j])
            ps.append(PhaseSpaceParticle(
                cur_params[0], cur_params[1], cur_params[2], cur_params[3], 0.0, cur_params[4]))

        return ps
