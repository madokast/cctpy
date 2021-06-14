"""
CCT 建模优化代码
CCT

作者：赵润晓
日期：2021年5月1日
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
from packages.line3s import *
from packages.trajectory import Trajectory
from packages.particles import *
from packages.magnets import *


class CCT(Magnet, ApertureObject):
    """
    表示一层弯曲 CCT 线圈
    """

    def __init__(
            self,
            # CCT 局部坐标系
            local_coordinate_system: LocalCoordinateSystem,
            # 大半径：偏转半径
            big_r: float,
            # 小半径（孔径/2）
            small_r: float,
            # 偏转角度，即 phi0*winding_number，典型值 67.5
            bending_angle: float,  # 必须为正
            # 各极倾斜角，典型值 [30,90,90,90]
            tilt_angles: List[float],
            # 匝数
            winding_number: int,
            # 电流
            current: float,
            # CCT 路径在二维 ξ-φ 坐标系中的起点
            starting_point_in_ksi_phi_coordinate: P2,
            # CCT 路径在二维 ξ-φ 坐标系中的终点
            end_point_in_ksi_phi_coordinate: P2,
            # 每匝线圈离散电流元数目，数字越大计算精度越高
            disperse_number_per_winding: int = 120,
    ):
        if bending_angle < 0:
            print(f"CCT 偏转角度应为正数，不能是 {bending_angle}，需要反向偏转的 CCT，" +
                  "应通过 starting_point_in_ksi_phi_coordinate，和 end_point_in_ksi_phi_coordinate 控制偏转方向"
                  )
        self.local_coordinate_system = local_coordinate_system
        self.big_r = float(big_r)
        self.small_r = float(small_r)
        self.bending_angle = float(bending_angle)
        self.tilt_angles = [float(e) for e in tilt_angles]
        self.winding_number = int(winding_number)
        self.current = float(current)
        self.starting_point_in_ksi_phi_coordinate = starting_point_in_ksi_phi_coordinate
        self.end_point_in_ksi_phi_coordinate = end_point_in_ksi_phi_coordinate
        self.disperse_number_per_winding = int(disperse_number_per_winding)

        # 弯转角度，弧度制
        self.bending_radian = BaseUtils.angle_to_radian(self.bending_angle)

        # 倾斜角，弧度制
        self.tilt_radians = BaseUtils.angle_to_radian(self.tilt_angles)

        # 每绕制一匝，φ 方向前进长度
        self.phi0 = self.bending_radian / self.winding_number

        # 极点 a
        self.a = math.sqrt(self.big_r ** 2 - self.small_r ** 2)

        # 双极坐标系另一个常量 η
        self.eta = 0.5 * \
            math.log((self.big_r + self.a) / (self.big_r - self.a))

        # 建立 ξ-φ 坐标到三维 xyz 坐标的转换器
        self.bipolar_toroidal_coordinate_system = CCT.BipolarToroidalCoordinateSystem(
            self.a, self.eta, self.big_r, self.small_r
        )

        # CCT 路径的在 ξ-φ 坐标的表示 函数 φ(ξ)
        def phi_ksi_function(ksi): return self.phi_ksi_function(ksi)

        # CCT 路径的在 ξ-φ 坐标的表示 函数 P(ξ)=(ξ,φ(ξ))
        def p2_function(ksi): return P2(ksi, phi_ksi_function(ksi))

        # CCT 路径的在 xyz 坐标的表示 函数 P(ξ)=P(x(ξ),y(ξ),z(ξ))
        def p3_function(ksi): return self.bipolar_toroidal_coordinate_system.convert(
            p2_function(ksi)
        )

        # self.phi_ksi_function = phi_ksi_function
        # self.p2_function = p2_function
        # self.p3_function = p3_function

        # 总分段数目 / 电流元数目
        self.total_disperse_number = self.winding_number * self.disperse_number_per_winding

        dispersed_path2: List[List[float]] = [
            p2_function(ksi).to_list()
            for ksi in BaseUtils.linspace(
                self.starting_point_in_ksi_phi_coordinate.x,
                self.end_point_in_ksi_phi_coordinate.x,
                self.total_disperse_number + 1,
            )  # +1 为了满足分段正确性，即匝数 m，需要用 m+1 个点
        ]

        self.dispersed_path3_points: List[P3] = [
            p3_function(ksi)
            for ksi in BaseUtils.linspace(
                self.starting_point_in_ksi_phi_coordinate.x,
                self.end_point_in_ksi_phi_coordinate.x,
                self.total_disperse_number + 1,
            )  # +1 为了满足分段正确性，见上
        ]

        dispersed_path3: List[List[float]] = [
            p.to_list() for p in self.dispersed_path3_points
        ]

        # 为了速度，转为 numpy
        self.dispersed_path2: numpy.ndarray = numpy.array(dispersed_path2)
        self.dispersed_path3: numpy.ndarray = numpy.array(dispersed_path3)

        # 电流元 (miu0/4pi) * current * (p[i+1] - p[i])
        # refactor v0.1.1
        # 语法分析：示例
        # a = array([1, 2, 3, 4])
        # a[1:] = array([2, 3, 4])
        # a[:-1] = array([1, 2, 3])
        self.elementary_current = 1e-7 * current * (
            self.dispersed_path3[1:] - self.dispersed_path3[:-1]
        )

        # 电流元的位置 (p[i+1]+p[i])/2
        self.elementary_current_position = 0.5 * (
            self.dispersed_path3[1:] + self.dispersed_path3[:-1]
        )

    def phi_ksi_function(self, ksi: float) -> float:
        """
        完成 ξ 到 φ 的映射
        """
        x1 = self.starting_point_in_ksi_phi_coordinate.x
        y1 = self.starting_point_in_ksi_phi_coordinate.y
        x2 = self.end_point_in_ksi_phi_coordinate.x
        y2 = self.end_point_in_ksi_phi_coordinate.y

        k = (y2 - y1) / (x2 - x1)
        b = -k * x1 + y1

        phi = k * ksi + b
        for i in range(len(self.tilt_radians)):
            if BaseUtils.equal(self.tilt_angles[i], 90.0):
                continue
            phi += (
                (1 / math.tan(self.tilt_radians[i]))
                / ((i + 1) * math.sinh(self.eta))
                * math.sin((i + 1) * ksi)
            )
        return phi

    class BipolarToroidalCoordinateSystem:
        """
        双极点坐标系
        """

        def __init__(self, a: float, eta: float, big_r: float, small_r: float):
            self.a = a
            self.eta = eta
            self.big_r = big_r
            self.small_r = small_r

            BaseUtils.equal(
                big_r,
                math.sqrt(a * a / (1 - 1 / math.pow(math.cosh(eta), 2))),
                msg=f"BipolarToroidalCoordinateSystem:init 错误1 a({a})eta({eta})R({big_r})r({small_r})",
            )

            BaseUtils.equal(
                small_r,
                big_r / math.cosh(eta),
                msg=f"BipolarToroidalCoordinateSystem:init 错误2 a({a})eta({eta})R({big_r})r({small_r})",
            )

        def convert(self, p: P2) -> P3:
            """
            将二维坐标 (ξ,φ) 转为三维坐标 (x,y,z)
            """
            ksi = p.x
            phi = p.y
            temp = self.a / (math.cosh(self.eta) - math.cos(ksi))
            return P3(
                temp * math.sinh(self.eta) * math.cos(phi),
                temp * math.sinh(self.eta) * math.sin(phi),
                temp * math.sin(ksi),
            )

        def main_normal_direction_at(self, p: P2) -> P3:
            """
            返回二维坐标 (ξ,φ) 映射到的三维坐标 (x,y,z) 点，
            它在圆环面上的法向量
            即返回值 P3 在这点 (x,y,z) 垂直于圆环面

            注意：已正则归一化
            """
            phi = p.y

            center = P3(self.big_r * math.cos(phi),
                        self.big_r * math.sin(phi), 0)

            face_point = self.convert(p)

            return (face_point - center).normalize()

        def __str__(self):
            return f"BipolarToroidalCoordinateSystem a({self.a})eta({self.eta})R({self.big_r})r({self.small_r})"

        def __repr__(self) -> str:
            return self.__str__()

    def magnetic_field_at(self, point: P3) -> P3:
        """
        计算 CCT 在全局坐标系点 P3 参数的磁场
        为了计算效率，使用 numpy
        """
        if BaseUtils.equal(self.current, 0, err=1e-6):
            return P3.zeros()

        # point 转为局部坐标，并变成 numpy 向量
        p = numpy.array(
            self.local_coordinate_system.point_to_local_coordinate(
                point).to_list()
        )

        # 点 p 到电流元中点
        r = p - self.elementary_current_position

        # 点 p 到电流元中点的距离的三次方
        rr = (numpy.linalg.norm(r, ord=2, axis=1)
              ** (-3)).reshape((r.shape[0], 1))

        # 计算每个电流元在 p 点产生的磁场 (此时还没有乘系数 μ0/4π )
        dB = numpy.cross(self.elementary_current, r) * rr

        # 求和，即得到磁场，
        # (不用乘乘以系数 μ0/4π = 1e-7)
        # refactor v0.1.1
        B = numpy.sum(dB, axis=0)

        # 转回 P3
        B_P3: P3 = P3.from_numpy_ndarry(B)

        # 从局部坐标转回全局坐标
        B_P3: P3 = self.local_coordinate_system.vector_to_global_coordinate(
            B_P3)

        return B_P3

    # from ApertureObject
    def is_out_of_aperture(self, point: P3) -> bool:
        """
        判断点 point 是在 CCT 的孔径内还是孔径外
        只有当粒子轴向投影在元件内部时，才会进行判断，
        否则即时粒子距离轴线很远，也认为粒子没有超出孔径，
        这是因为粒子不在元件内时，很可能处于另一个大孔径元件中，这样会造成误判。

        point 为全局坐标系点
        """
        # 转为局部坐标
        local_point = self.local_coordinate_system.point_to_local_coordinate(
            point)
        local_point_p2 = local_point.to_p2()

        # 查看偏转方向
        clockwise = self.end_point_in_ksi_phi_coordinate.y < 0

        # 映射到 cct 所在圆环轴上
        phi = local_point_p2.angle_to_x_axis()

        # 查看是否在 cct 轴上
        if clockwise:
            # phi 应大于 2pi-bending_radian 小于 2pi
            if phi > (2 * math.pi - self.bending_radian):
                return (
                    abs(local_point.z) > self.small_r
                    or local_point_p2.length() > (self.big_r + self.small_r)
                    or local_point_p2.length() < (self.big_r - self.small_r)
                )
            else:
                return False
        else:
            if phi < self.bending_radian:
                return (
                    abs(local_point.z) > self.small_r
                    or local_point_p2.length() > (self.big_r + self.small_r)
                    or local_point_p2.length() < (self.big_r - self.small_r)
                )
            else:
                return False

    def __str__(self):
        return (
            f"CCT: local_coordinate_system({self.local_coordinate_system})big_r({self.big_r})small_r({self.small_r})"
            + f"bending_angle({self.bending_angle})tilt_angles({self.tilt_angles})winding_number({self.winding_number})"
            + f"current({self.current})starting_point_in_ksi_phi_coordinate({self.starting_point_in_ksi_phi_coordinate})"
            + f"end_point_in_ksi_phi_coordinate({self.end_point_in_ksi_phi_coordinate})"
            + f"disperse_number_per_winding({self.disperse_number_per_winding})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def create_cct_along(
            # 设计轨道
            trajectory: Line2,
            # 设计轨道上该 CCT 起点
            s: float,
            # 大半径：偏转半径
            big_r: float,
            # 小半径（孔径/2）
            small_r: float,
            # 偏转角度，即 phi0*winding_number，典型值 67.5
            bending_angle: float,
            # 各极倾斜角，典型值 [30,90,90,90]
            tilt_angles: List[float],
            # 匝数
            winding_number: int,
            # 电流
            current: float,
            # CCT 路径在二维 ξ-φ 坐标系中的起点
            starting_point_in_ksi_phi_coordinate: P2,
            # CCT 路径在二维 ξ-φ 坐标系中的终点
            end_point_in_ksi_phi_coordinate: P2,
            # 每匝线圈离散电流元数目，数字越大计算精度越高
            disperse_number_per_winding: int = 120,
    ) -> "CCT":
        """
        按照设计轨迹 trajectory 上 s 位置处创建 CCT
        """
        start_point: P2 = trajectory.point_at(s)
        arc_length: float = big_r * BaseUtils.angle_to_radian(bending_angle)
        end_point: P2 = trajectory.point_at(
            s + arc_length)  # 2021年1月15日 bug fixed

        midpoint0: P2 = trajectory.point_at(s + arc_length / 3 * 1)
        midpoint1: P2 = trajectory.point_at(s + arc_length / 3 * 2)

        c1, r1 = BaseUtils.circle_center_and_radius(
            start_point, midpoint0, midpoint1)
        c2, r2 = BaseUtils.circle_center_and_radius(
            midpoint0, midpoint1, end_point)
        BaseUtils.equal(
            c1, c2, msg=f"构建 CCT 存在异常，通过设计轨道判断 CCT 圆心不一致，c1{c1}，c2{c2}")
        BaseUtils.equal(
            r1, r2, msg=f"构建 CCT 存在异常，通过设计轨道判断 CCT 半径不一致，r1{r1}，r2{r2}")
        center: P2 = (c1 + c2) * 0.5

        start_direct: P2 = trajectory.direct_at(s)
        pos: int = StraightLine2(
            # position_of 求点 p 相对于直线段的方位
            # 返回值：
            # 1  在右侧    # -1 在左侧    # 0  在直线段所在直线上
            1.0, start_direct, start_point).position_of(center)

        lcs = None
        if pos == 0:
            raise ValueError(f"错误：圆心{center}在设计轨道{trajectory}上")
        elif pos == 1:  # center 在 (start_direct, start_point) 右侧，顺时针
            lcs = LocalCoordinateSystem.create_by_y_and_z_direction(
                location=center.to_p3(),
                y_direction=-start_direct.to_p3(),  # diff
                z_direction=P3.z_direct(),
            )
        # pos = -1  # center 在 (start_direct, start_point) 左侧，逆时针时针
        else:
            lcs = LocalCoordinateSystem.create_by_y_and_z_direction(
                location=center.to_p3(),
                y_direction=start_direct.to_p3(),  # diff
                z_direction=P3.z_direct(),
            )
        return CCT(
            local_coordinate_system=lcs,
            big_r=big_r,
            small_r=small_r,
            bending_angle=bending_angle,
            tilt_angles=tilt_angles,
            winding_number=winding_number,
            current=current,
            starting_point_in_ksi_phi_coordinate=starting_point_in_ksi_phi_coordinate,
            end_point_in_ksi_phi_coordinate=end_point_in_ksi_phi_coordinate,
            disperse_number_per_winding=disperse_number_per_winding,
        )

    def global_path3(self) -> List[P3]:
        """
        获取 CCT 路径点，以全局坐标系的形式
        主要目的是为了 CUDA 计算
        since v0.1.1
        """
        return [
            self.local_coordinate_system.point_to_global_coordinate(p)
            for p in self.dispersed_path3_points
        ]

    def global_current_elements_and_elementary_current_positions(self, numpy_dtype=numpy.float64) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        获取全局坐标系下的
        电流元 (miu0/4pi) * current * (p[i+1] - p[i])
        和
        电流元的位置 (p[i+1]+p[i])/2
        主要目的是为了 CUDA 计算

        since v0.1.1
        """
        global_path3: List[P3] = self.global_path3()

        global_path3_numpy_array = numpy.array(
            [p.to_list() for p in global_path3], dtype=numpy_dtype)

        global_current_elements = 1e-7 * self.current * \
            (global_path3_numpy_array[1:] - global_path3_numpy_array[:-1])

        global_elementary_current_positions = 0.5 * \
            (global_path3_numpy_array[1:] + global_path3_numpy_array[:-1])
        return (
            global_current_elements.flatten(),
            global_elementary_current_positions.flatten()
        )

    def p2_function(self, ksi) -> P2:
        """
        二维坐标系点 (ksi, phi)

        since v0.1.1
        """
        return P2(ksi, self.phi_ksi_function(ksi))

    def p3_function(self, ksi) -> P3:
        """
        局部坐标系下路径方程

        since v0.1.1
        """
        return self.bipolar_toroidal_coordinate_system.convert(self.p2_function(ksi))

    def conductor_length(self, line_number: int = 2*7, disperse_number_per_winding: int = 360) -> float:
        """
        计算导线长度
        line_number 导线数目

        since v0.1.1
        """
        ksi0 = self.starting_point_in_ksi_phi_coordinate.x
        ksi1 = self.end_point_in_ksi_phi_coordinate.x
        ksi_list = BaseUtils.linspace(
            ksi0, ksi1, disperse_number_per_winding*self.winding_number+1)
        length: float = 0.0
        for i in range(len(ksi_list)-1):
            p0 = self.p3_function(ksi_list[i])
            p1 = self.p3_function(ksi_list[i+1])
            length += (p1-p0).length()
        return length*line_number

    def as_cct(anything) -> 'CCT':
        """
        仿佛是类型转换
        实际啥也没做
        但是 IDE 就能根据返回值做代码提示了

        常用在将 Magnet 转成 CCT
        例如从 Beamline 中取出的 magnets，然后按照真是类型转过去

        since v0.1.3
        """
        return anything

    @staticmethod
    def calculate_a(big_r: float, small_r: float):
        """
        计算极点 a 值的小方法
        """

        return math.sqrt(big_r ** 2 - small_r ** 2)

    @staticmethod
    def calculate_eta(big_r: float, small_r: float):
        """
        计算 eta 值的小方法
        """

        return 0.5 * math.log((big_r + CCT.calculate_a(big_r, small_r))
                              / (big_r - CCT.calculate_a(big_r, small_r)))

    @staticmethod
    def calculate_cheta(big_r: float, small_r: float):
        """
        计算 ch(eta) 值的小方法
        """

        return math.cosh(CCT.calculate_eta(big_r, small_r))

    @staticmethod
    def calculate_sheta(big_r: float, small_r: float):
        """
        计算 sh(eta) 值的小方法
        """

        return math.sinh(CCT.calculate_eta(big_r, small_r))


class Wire(Magnet):
    """
    任意空间三维导线
    """

    def __init__(self, function_line3: FunctionLine3, current: float, delta_length: float = 1*MM) -> None:
        """
        function_line3 导线路径
        current 电流
        delta_length 导线分割成电流元时，每个电流元的长度
        """
        self.function_line3 = function_line3
        self.current = current
        self.start = function_line3.get_start()
        self.end = function_line3.get_end()

        # 分段数目
        part_number = int(math.ceil(
            (self.end-self.start)/delta_length
        ))

        self.dispersed_s = numpy.array(
            BaseUtils.linspace(self.start, self.end, part_number+1))

        self.dispersed_path3 = numpy.array([
            self.function_line3.point_at_p3_function(s).to_numpy_ndarry3() for s in self.dispersed_s
        ])

        # 电流元 (miu0/4pi) * current * (p[i+1] - p[i])
        # refactor v0.1.1
        # 语法分析：示例
        # a = array([1, 2, 3, 4])
        # a[1:] = array([2, 3, 4])
        # a[:-1] = array([1, 2, 3])
        self.elementary_current = 1e-7 * self.current * (
            self.dispersed_path3[1:] - self.dispersed_path3[:-1]
        )

        # 电流元的位置 (p[i+1]+p[i])/2
        self.elementary_current_position = 0.5 * (
            self.dispersed_path3[1:] + self.dispersed_path3[:-1]
        )

    def magnetic_field_at(self, point: P3) -> P3:
        """
        计算磁场，全局坐标
        """
        if BaseUtils.equal(self.current, 0, err=1e-6):
            return P3.zeros()

        if BaseUtils.equal(self.start, self.end, err=1e-6):
            return P3.zeros()

        p = point.to_numpy_ndarry3()

        # 点 p 到电流元中点
        r = p - self.elementary_current_position

        # 点 p 到电流元中点的距离的三次方
        rr = (numpy.linalg.norm(r, ord=2, axis=1)
              ** (-3)).reshape((r.shape[0], 1))

        # 计算每个电流元在 p 点产生的磁场 (此时还没有乘系数 μ0/4π )
        dB = numpy.cross(self.elementary_current, r) * rr

        # 求和，即得到磁场，
        # (不用乘乘以系数 μ0/4π = 1e-7)
        # refactor v0.1.1
        B = numpy.sum(dB, axis=0)

        # 转回 P3
        B_P3: P3 = P3.from_numpy_ndarry(B)

        return B_P3

    def magnetic_field_on_wire(self, s: float, delta_length: float,
                               local_coordinate_point: LocalCoordinateSystem,
                               other_magnet: Magnet = Magnet.no_magnet()
                               ) -> Tuple[P3, P3]:
        """
        计算线圈上磁场，
        s 线圈位置，即要计算磁场的位置
        delta_length 分段长度
        local_coordinate_point 局部坐标系
        这个局部坐标系仅仅将计算的磁场转为局部坐标系中

        返回值 位置点+磁场
        """
        # 所在点（全局坐标系）
        p = self.function_line3.point_at_p3_function(s)

        # 所在点之前的导线
        pre_line3 = FunctionLine3(
            p3_function=self.function_line3.get_p3_function(),
            start=self.start,
            end=s - delta_length/2
        )

        # 所在点之后的导线
        post_line3 = FunctionLine3(
            p3_function=self.function_line3.get_p3_function(),
            start=s + delta_length/2,
            end=self.function_line3.get_end()
        )

        # 所在点之前的导线
        pre_wire = Wire(
            function_line3=pre_line3,
            current=self.current,
            delta_length=delta_length
        )

        # 所在点之后的导线
        post_wire = Wire(
            function_line3=post_line3,
            current=self.current,
            delta_length=delta_length
        )

        # 磁场
        B = pre_wire.magnetic_field_at(
            p) + post_wire.magnetic_field_at(p) + other_magnet.magnetic_field_at(p)

        # 返回点 + 磁场
        return p, local_coordinate_point.vector_to_local_coordinate(B)

    def lorentz_force_on_wire(self, s: float, delta_length: float,
                              local_coordinate_point: LocalCoordinateSystem,
                              other_magnet: Magnet = Magnet.no_magnet()
                              ) -> Tuple[P3, P3]:
        """
        计算线圈上 s 位置处洛伦兹力
        delta_length 线圈分段长度（s位置的这一段线圈不参与计算）
        local_coordinate_point 洛伦兹力坐标系
        这个局部坐标系仅仅将计算的洛伦兹力转为局部坐标系中
        """
        p, b = self.magnetic_field_on_wire(
            s=s,
            delta_length=delta_length,
            local_coordinate_point=LocalCoordinateSystem.global_coordinate_system(),
            other_magnet=other_magnet
        )

        direct = self.function_line3.direct_at_p3_function(s)

        F = self.current * delta_length * (direct@b)

        return p, local_coordinate_point.vector_to_local_coordinate(F)

    def pressure_on_wire_MPa(self, s: float, delta_length: float,
                             local_coordinate_point: LocalCoordinateSystem,
                             channel_width: float, channel_depth: float,
                             other_magnet: Magnet = Magnet.no_magnet(),
                             ) -> Tuple[P3, P3]:
        """
        计算压强，默认为 CCT

        默认在 local_coordinate_point 坐标系下计算的洛伦兹力 F

        Fx 绕线方向（应该是 0 ）
        Fy rib 方向 / 副法线方向
        Fz 径向

        返回值点 P 和三个方向的压强Pr

        Pr.x 绕线方向压强 Fx / (channel_width * channel_depth)
        Pr.y rib 方向压强 Fy / (绕线方向长度 * channel_depth)
        Pr.z 径向压强 Fz / (绕线方向长度 * channel_width)

        关于《绕线方向长度》 == len(point_at(s + delta_length/2)-point_at(s - delta_length/2))

        返回值为 点P和 三方向压强/兆帕
        """
        p, f = self.lorentz_force_on_wire(
            s, delta_length, local_coordinate_point, other_magnet
        )

        winding_direct_length: float = (self.function_line3.point_at_p3_function(s+delta_length/2) -
                                        self.function_line3.point_at_p3_function(s-delta_length/2)).length()

        pressure: P3 = P3(
            x=f.x/(channel_width*channel_depth),
            y=f.y/(winding_direct_length*channel_depth),
            z=f.z/(winding_direct_length*channel_width)
        )/1e6

        return p, pressure

    @staticmethod
    def create_by_cct(cct: CCT) -> 'Wire':
        """
        由 CCT 创建 wire
        """
        def p3f(ksi):
            return cct.bipolar_toroidal_coordinate_system.convert(
                P2(ksi, cct.phi_ksi_function(ksi))
            )

        return Wire(
            function_line3=FunctionLine3(
                p3_function=p3f,
                start=cct.starting_point_in_ksi_phi_coordinate.x,
                end=cct.end_point_in_ksi_phi_coordinate.x
            ),
            current=cct.current,
            delta_length=cct.small_r *
            BaseUtils.angle_to_radian(360/cct.winding_number)
        )

class AGCCT_CONNECTOR(Magnet):
    """
    agcct 连接段的构建，尽可能的自动化
    注意，存在重大 bug，即认为两个连接点处的切向，方向相反，实际上是错的！
    在当前一般配置的 CCT 中，切向方向大约相差 0.5 度
    这是什么概念呢？把 CCT 一匝分为 720 段，才能达到相接处 0.5 度变化。
    鉴于 0.5 度似乎可以忽略，暂时就不解决这个 bug 了（因为连接过程很复杂）
    """
    def __init__(self, agcct1: CCT, agcct2: CCT, step: float = 1*MM) -> None:
        # fields
        # self.current
        # self.local_coordinate_system
        # self.length
        # self.dispersed_path3

        # 必要的验证
        if agcct1.local_coordinate_system != agcct2.local_coordinate_system:
            raise ValueError(
                f"需要连接的cct不属于同一坐标系，无法连接。agccct1={agcct1},agcct2={agcct2}")
        BaseUtils.equal(agcct1.big_r, agcct2.big_r,
                        msg=f"需要连接的 cct big_r 不同，无法连接。agccct1={agcct1},agcct2={agcct2}")
        BaseUtils.equal(agcct1.small_r, agcct2.small_r,
                        msg=f"需要连接的 cct small_r 不同，无法连接。agccct1={agcct1},agcct2={agcct2}")
        BaseUtils.equal(agcct1.current, agcct2.current,
                        msg=f"需要连接的 cct 电流不同，无法连接。agccct1={agcct1},agcct2={agcct2}")
        self.current = agcct1.current
        # 坐标系
        self.local_coordinate_system = agcct1.local_coordinate_system

        # 前 cct 的终点
        pre_cct_end_point_in_ksi_phi_coordinate = agcct1.end_point_in_ksi_phi_coordinate
        # 后 cct 的起点
        next_cct_starting_point_in_ksi_phi_coordinate = agcct2.starting_point_in_ksi_phi_coordinate
        BaseUtils.equal(pre_cct_end_point_in_ksi_phi_coordinate.x, next_cct_starting_point_in_ksi_phi_coordinate.x,
                        msg=f"需要连接的前段 cct 终点 ξ 不等于后段 cct 的起点 ξ，无法连接。agccct1={agcct1},agcct2={agcct2}")

        # 前 cct 的终点，在 φR-ξr 坐标系中的点 a
        # 后 cct 的起点  b
        # 以及切向 va vb
        a = P2(x=pre_cct_end_point_in_ksi_phi_coordinate.y*agcct1.big_r, y=0.0)
        b = P2(x=next_cct_starting_point_in_ksi_phi_coordinate.y*agcct2.big_r, y=0.0)
        va = BaseUtils.derivative(lambda ksi: P2(
            x=agcct1.p2_function(ksi).y*agcct1.big_r,
            y=agcct1.p2_function(ksi).x*agcct1.small_r))(
            pre_cct_end_point_in_ksi_phi_coordinate.x
            # 下行的乘法很重要，因为自变量不一定是随着绕线增大，如果反向绕线就是减小 fixed 2021年1月8日
        ) *(-1 if agcct1.starting_point_in_ksi_phi_coordinate.x > agcct1.end_point_in_ksi_phi_coordinate.x else 1)
        vb = BaseUtils.derivative(lambda ksi: P2(
            x=agcct2.p2_function(ksi).y*agcct2.big_r,
            y=agcct2.p2_function(ksi).x*agcct2.small_r))(
            next_cct_starting_point_in_ksi_phi_coordinate.x
        )*(-1 if agcct2.starting_point_in_ksi_phi_coordinate.x > agcct2.end_point_in_ksi_phi_coordinate.x else 1)
        print(f"a={a}, b={b}")
        print(f"va={va}, vb={vb}")

        # 开始连接
        # 首先是 z-Ξ 坐标系
        ip, ka, kb = StraightLine2.intersecting_point(
            pa=a,
            va=va.copy().rotate(BaseUtils.angle_to_radian(90)),
            pb=b,
            vb=vb
        )
        if kb > 0:
            ip, ka, kb = StraightLine2.intersecting_point(
                pa=a,
                va=va,
                pb=b,
                vb=vb.copy().rotate(BaseUtils.angle_to_radian(90))
            )
            assert ka >= 0, 'ka 应该非负'
            ca = a + ka*va

            connector_line_in_z_Θ = (
                Trajectory
                .set_start_point(start_point=a)
                .first_line(
                    direct=va, length=abs(ka*va.length()))
                .add_arc_line(
                    radius=(ca-b).length()/2,
                    clockwise=StraightLine2.is_on_right(ca, va, b) == 1,
                    angle_deg=180.0
                )
            )
        else:  # kb<=0:
            cb = b + kb*vb
            connector_line_in_z_Θ = (
                Trajectory
                .set_start_point(start_point=a)
                .first_line(direct=va, length=0.0).add_arc_line(
                    radius=(a-cb).length()/2,
                    clockwise=StraightLine2.is_on_right(a, va, b) == 1,
                    angle_deg=180.0
                ).add_strait_line(length=abs(kb*vb.length()))
            )

        self.connector_line_in_z_Θ = connector_line_in_z_Θ
        self.length_in_z_Θ = connector_line_in_z_Θ.get_length()
        self.p2_function = lambda t: P2(
            x=self.connector_line_in_z_Θ.point_at(t).y/agcct1.small_r,
            y=self.connector_line_in_z_Θ.point_at(t).x/agcct1.big_r
        )
        self.p2_function_start = 0
        self.p2_function_end = self.length_in_z_Θ

        # z-Ξ 转到 ksi - phi 坐标系
        # z = φR
        # Ξ = ξr
        # 再转到三维坐标系 xyz
        dispersed_path3 = [agcct1.bipolar_toroidal_coordinate_system.convert(P2(
            x=p2.y/agcct1.small_r,  # p2.y 为 Ξ，ξ = Ξ/r
            y=p2.x/agcct1.big_r  # p2.x 为 z，φ = z/R
        )).to_list() for p2 in self.connector_line_in_z_Θ.disperse2d(step=step)]

        # 转为 numpy 数组
        self.dispersed_path3: numpy.ndarray = numpy.array(dispersed_path3)

        self.elementary_current = 1e-7 * self.current * (
            self.dispersed_path3[1:] - self.dispersed_path3[:-1]
        )

        # 电流元的位置 (p[i+1]+p[i])/2
        self.elementary_current_position = 0.5 * (
            self.dispersed_path3[1:] + self.dispersed_path3[:-1]
        )

    def magnetic_field_at(self, point: P3) -> P3:
        if BaseUtils.equal(self.current, 0, err=1e-6):
            return P3.zeros()

        # point 转为局部坐标，并变成 numpy 向量
        p = numpy.array(
            self.local_coordinate_system.point_to_local_coordinate(
                point).to_list()
        )

        # 点 p 到电流元中点
        r = p - self.elementary_current_position

        # 点 p 到电流元中点的距离的三次方
        rr = (numpy.linalg.norm(r, ord=2, axis=1)
              ** (-3)).reshape((r.shape[0], 1))

        # 计算每个电流元在 p 点产生的磁场 (此时还没有乘系数 μ0/4π )
        dB = numpy.cross(self.elementary_current, r) * rr

        # 求和，即得到磁场，
        # (不用乘乘以系数 μ0/4π = 1e-7)
        # refactor v0.1.1
        B = numpy.sum(dB, axis=0)

        # 转回 P3
        B_P3: P3 = P3.from_numpy_ndarry(B)

        # 从局部坐标转回全局坐标
        B_P3: P3 = self.local_coordinate_system.vector_to_global_coordinate(
            B_P3)

        return B_P3