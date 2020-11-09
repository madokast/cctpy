"""
粒子和粒子跟踪
"""
from typing import List, Tuple

import numpy as np

from cctpy.abstract_classes import Plotable, Magnet, LocalCoordinateSystem
from cctpy.baseutils import Vectors, Equal, Stream, Ellipse
from cctpy.constant import MM, LIGHT_SPEED, Protons, ZI


class RunningParticle(Plotable):
    """
    在全局坐标系中运动的一个粒子
    position 位置，三维矢量，单位 [m, m, m]
    velocity 速度，三位矢量，单位 [m/s, m/s, m/s]
    relativistic_mass 相对论质量，又称为动质量，单位 kg， M=Mo/√(1-v^2/c^2)
    e 电荷量，单位 C 库伦
    speed 速率，单位 m/s
    distance 运动距离，单位 m
    """

    def __init__(self, position: np.ndarray, velocity: np.ndarray,
                 relativistic_mass: float, e: float, speed: float, distance: float = 0.0):
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
        """
        self.position = position
        self.velocity = velocity
        self.relativistic_mass = relativistic_mass
        self.e = e
        self.speed = speed
        self.distance = distance

    def run_self_in_magnetic_field(self, magnetic_field: np.ndarray, footstep: float = 1 * MM) -> None:
        """
        粒子在磁场 magnetic_field 中运动 footstep 长度
        Parameters
        ----------
        magnetic_field 磁场，看作恒定场
        footstep 步长，默认 1 MM

        Returns None
        -------
        """
        # 计算受力 qvb
        f = self.e * (np.cross(self.velocity, magnetic_field))
        # 计算加速度 a = f/m
        a = f / self.relativistic_mass
        # 计算运动时间
        t = footstep / self.speed
        # 位置变化
        self.position += t * self.velocity
        # 速度变化
        self.velocity += t * a
        # 运动距离
        self.distance += footstep

    def copy(self):
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

    def line_and_color(self, describe: str = 'r.') -> List[Tuple[np.ndarray, str]]:
        return [
            ([self.position], describe)
        ]

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
        m0 = self.relativistic_mass * np.sqrt(1 - (self.speed ** 2) / (LIGHT_SPEED ** 2))
        # 求新的速率
        new_speed = scalar_momentum / np.sqrt(m0 ** 2 + (scalar_momentum / LIGHT_SPEED) ** 2)
        # 求新的动质量
        new_relativistic_mass = m0 / np.sqrt(1 - (new_speed / LIGHT_SPEED) ** 2)
        # 求新的速度
        new_velocity = Vectors.update_length(self.velocity, new_speed)

        # 写入
        self.relativistic_mass = new_relativistic_mass
        self.speed = new_speed
        self.velocity = new_velocity

        # 验证
        Equal.require_float_equal(
            scalar_momentum, self.compute_scalar_momentum(),
            f"RunningParticle::change_scalar_momentum异常，scalar_momentum{scalar_momentum}!=self.compute_scalar_momentum{self.compute_scalar_momentum}"
        )

        Equal.require_float_equal(
            self.speed, Vectors.length(self.velocity),
            f"RunningParticle::change_scalar_momentum异常,self.speed{self.speed}!=Vectors.length(self.velocity){Vectors.length(self.velocity)}"
        )

    def get_natural_coordinate_system(self, y_direction: np.ndarray = ZI) -> LocalCoordinateSystem:
        return LocalCoordinateSystem.create_by_y_and_z_direction(self.position, y_direction, self.velocity)

    def __str__(self) -> str:
        return f"p={self.position},v={self.velocity},v0={self.speed}"


class ParticleRunner:
    """
    粒子运动工具类
    """

    @staticmethod
    def run_only(p: RunningParticle, m: Magnet, length: float, footstep: float = 1 * MM) -> None:
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

        """
        distance = 0.0
        while distance < length:
            p.run_self_in_magnetic_field(m.magnetic_field_at(p.position), footstep=footstep)
            distance += footstep

    @staticmethod
    def run_get_trajectory(p: RunningParticle, m: Magnet, length: float, footstep: float = 1 * MM) -> np.ndarray:
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

        """
        trajectory = np.empty((int(length / footstep) + 1, 3))
        trajectory[0, :] = p.position.copy()

        i = 1
        distance = 0.0
        while distance < length:
            p.run_self_in_magnetic_field(m.magnetic_field_at(p.position), footstep=footstep)
            distance += footstep
            trajectory[i, :] = p.position.copy()
            i += 1

        return trajectory[0:i, :]

    @staticmethod
    def run_get_all_info(p: RunningParticle, m: Magnet, length: float, footstep: float = 1 * MM) \
            -> List[RunningParticle]:
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

        """
        all_info: List[RunningParticle] = [p.copy()]
        distance = 0.0
        while distance < length:
            p.run_self_in_magnetic_field(m.magnetic_field_at(p.position), footstep=footstep)
            distance += footstep
            all_info.append(p.copy())

        return all_info

    @staticmethod
    def run_ps_only_cpu0(ps: List[RunningParticle], m: Magnet, length: float, footstep: float = 1 * MM) -> None:
        """
        让粒子群 ps 在磁场 m 中运动 length 距离，步长 footstep
        CPU 计算 单线程
        Parameters
        ----------
        ps 一群粒子
        m 磁场
        length 运动长度
        footstep 步长


        Returns None
        -------

        """
        for p in ps:
            ParticleRunner.run_only(p, m, length, footstep)


class PhaseSpaceParticle:
    XXP_PLANE = 1
    YYP_PLANE = 2

    """
    相空间中的粒子，6个坐标 x xp y yp z delta
    """

    def __init__(self, x: float, xp: float, y: float, yp: float, z: float, delta: float):
        self.x = x
        self.xp = xp
        self.y = y
        self.yp = yp
        self.z = z
        self.delta = delta

    def project_to_xxp_plane(self) -> np.ndarray:
        """
        投影到 x-xp 平面
        Returns [self.x, self.xp]
        -------

        """
        return np.array([self.x, self.xp])

    def project_to_yyp_plane(self) -> np.ndarray:
        """
        投影到 y-yp 平面
        Returns [self.y, self.yp]
        -------

        """
        return np.array([self.y, self.yp])

    def project_to_plane(self, plane_id: int) -> np.ndarray:
        if plane_id == PhaseSpaceParticle.XXP_PLANE:
            return self.project_to_xxp_plane()
        elif plane_id == PhaseSpaceParticle.YYP_PLANE:
            return self.project_to_yyp_plane()
        else:
            raise ValueError(f"没有处理plane_id({plane_id})的方法")

    @staticmethod
    def phase_space_particles_along_positive_ellipse_in_xxp_plane(
            xMax: float, xpMax: float, delta: float, number: int
    ) -> List:
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

        return Stream(Ellipse(A, B, C, D).uniform_distribution_points_along_edge(number).tolist()).map(
            lambda p: PhaseSpaceParticle(p[0], p[1], 0, 0, 0, delta)).to_list()

    @staticmethod
    def phase_space_particles_along_positive_ellipse_in_yyp_plane(
            yMax: float, ypMax: float, delta: float, number: int
    ) -> List:
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

        return Stream(Ellipse(A, B, C, D).uniform_distribution_points_along_edge(number).tolist()).map(
            lambda p: PhaseSpaceParticle(0, 0, p[0], p[1], 0, delta)
        ).to_list()

    @staticmethod
    def phase_space_particles_along_positive_ellipse_in_plane(
            plane_id: int, xMax: float, xpMax: float, delta: float, number: int
    ) -> List:
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
    def phase_space_particles_project_to_xxp_plane(phase_space_particles: List) -> np.ndarray:
        """
        相空间粒子群投影到 x 平面
        Parameters
        ----------
        phase_space_particles 相空间粒子群

        Returns 相空间粒子群投影到 x 平面 [[x1,xp1], [x2,xp2] .. ]
        -------

        """
        return Stream(phase_space_particles).map(
            lambda p: p.project_to_xxp_plane()
        ).to_vector()

    @staticmethod
    def phase_space_particles_project_to_yyp_plane(phase_space_particles: List) -> np.ndarray:
        """
        相空间粒子群投影到 y 平面
        Parameters
        ----------
        phase_space_particles 相空间粒子群

        Returns 相空间粒子群投影到 y 平面 [[y1,yp1], [y2,yp2] .. ]
        -------

        """
        return Stream(phase_space_particles).map(
            lambda p: p.project_to_yyp_plane()
        ).to_vector()

    @staticmethod
    def phase_space_particles_project_to_plane(phase_space_particles: List, plane_id: int) -> np.ndarray:
        """
        相空间粒子群投影到 x/y 平面
        Parameters
        ----------
        phase_space_particles 相空间粒子群
        plane_id 投影到 x 或 y 平面

        Returns 相空间粒子群投影到 x/y 平面
        -------

        """
        if plane_id == PhaseSpaceParticle.XXP_PLANE:
            return PhaseSpaceParticle.phase_space_particles_project_to_xxp_plane(phase_space_particles)
        elif plane_id == PhaseSpaceParticle.YYP_PLANE:
            return PhaseSpaceParticle.phase_space_particles_project_to_yyp_plane(phase_space_particles)
        else:
            raise ValueError(f"没有处理plane_id({plane_id})的方法")

    @staticmethod
    def create_from_running_particle(ideal_particle: RunningParticle,
                                     coordinate_system: LocalCoordinateSystem,
                                     running_particle: RunningParticle):
        # x y z
        relative_position = running_particle.position - ideal_particle.position
        x = np.inner(coordinate_system.XI, relative_position)
        y = np.inner(coordinate_system.YI, relative_position)
        z = np.inner(coordinate_system.ZI, relative_position)

        # xp yp
        relative_velocity = running_particle.velocity - ideal_particle.velocity
        xp = np.inner(coordinate_system.XI, relative_velocity) / ideal_particle.speed
        yp = np.inner(coordinate_system.YI, relative_velocity) / ideal_particle.speed

        # delta
        rm = running_particle.compute_scalar_momentum()
        im = ideal_particle.compute_scalar_momentum()
        delta = (rm - im) / im

        return PhaseSpaceParticle(x, xp, y, yp, z, delta)

    @staticmethod
    def create_from_running_particles(ideal_particle: RunningParticle,
                                      coordinate_system: LocalCoordinateSystem,
                                      running_particles: List[RunningParticle]) -> List:
        return Stream(running_particles).map(
            lambda rp: PhaseSpaceParticle.create_from_running_particle(
                ideal_particle, coordinate_system, rp)
        ).to_list()

    def __str__(self) -> str:
        return f"x={self.x},xp={self.xp},y={self.y},yp={self.yp},z={self.z},d={self.delta}"


class ParticleFactory:
    """
    质子工厂
    """

    @staticmethod
    def create_proton(position: np.ndarray, direct: np.ndarray, kinetic_MeV: float = 250) -> RunningParticle:
        # 速率
        speed = LIGHT_SPEED * np.sqrt(
            1 - (Protons.STATIC_ENERGY_MeV / (Protons.STATIC_ENERGY_MeV + kinetic_MeV)) ** 2
        )

        # mass kg
        relativistic_mass = Protons.STATIC_MASS_KG / np.sqrt(
            1 - (speed ** 2) / (LIGHT_SPEED ** 2)
        )

        return RunningParticle(position, Vectors.update_length(direct.copy(), speed), relativistic_mass,
                               Protons.CHARGE_QUANTITY, speed)

    @staticmethod
    def create_from_phase_space_particle(
            ideal_particle: RunningParticle,
            coordinate_system: LocalCoordinateSystem,
            phase_space_particle: PhaseSpaceParticle) -> RunningParticle:
        """
        通过理想粒子，相空间坐标系 和 相空间粒子，来创造粒子
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
            scalar_momentum = p.compute_scalar_momentum() * (1. + delta)
            p.change_scalar_momentum(scalar_momentum)  # 这个方法就是为了修改动量而写的

        p.velocity += coordinate_system.XI * (xp * p.speed)
        p.velocity += coordinate_system.YI * (yp * p.speed)

        return p

    @staticmethod
    def create_from_phase_space_particles(
            ideal_particle: RunningParticle,
            coordinate_system: LocalCoordinateSystem,
            phase_space_particles: List[PhaseSpaceParticle]) -> List[RunningParticle]:
        return Stream(phase_space_particles).map(
            lambda p: ParticleFactory.create_from_phase_space_particle(ideal_particle, coordinate_system, p)
        ).to_list()
