"""
CCT 建模优化代码
磁铁类

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
from packages.lines import *
from packages.trajectory import Trajectory


class Magnet:
    """
    表示一个可以求磁场的对象，如 CCT 、 QS 磁铁
    所有实现此接口的类，可以计算出它在某一点的磁场

    本类（接口）只有一个接口方法 magnetic_field_at
    """

    def magnetic_field_at(self, point: P3) -> P3:
        """
        获得本对象 self 在点 point 处产生的磁场
        这个方法需要在子类中实现/重写
        ----------
        point 三维笛卡尔坐标系中的点，即一个三维矢量，如 [0,0,0]

        Returns 本对象 self 在点 point 处的磁场，用三维矢量表示
        -------
        """
        raise NotImplementedError

    def magnetic_field_along(
            self,
            line2: Line2,
            p2_t0_p3: Callable[[P2], P3] = lambda p2: P3(p2.x, p2.y, 0.0),
            step: float = 1 * MM,
    ) -> List[ValueWithDistance[P3]]:
        """
        计算本对象在二维曲线 line2 上的磁场分布
        p2_t0_p3 是一个函数，用于把 line2 上的二维点转为三维，默认转为 z=0 的三维点
        step 表示 line2 分段长度
        -------
        """
        length = line2.get_length()
        distances = BaseUtils.linspace(0, length, int(length / step) + 1)
        return [
            ValueWithDistance(
                self.magnetic_field_at(
                    line2.point_at(d).to_p3(transformation=p2_t0_p3)
                ),
                d,
            )
            for d in distances
        ]

    def magnetic_field_bz_along(
            self,
            line2: Line2,
            p2_t0_p3: Callable[[P2], P3] = lambda p2: P3(p2.x, p2.y, 0.0),
            step: float = 1 * MM,
    ) -> List[P2]:
        """
        计算本对象在二维曲线 line 上的磁场 Z 方向分量的分布
        因为磁铁一般放置在 XY 平面，所以 Bz 一般可以看作自然坐标系下 By，也就是二级场大小
        p2_t0_p3 是一个函数，用于把 line2 上的二维点转为三维，默认转为 z=0 的三维点
        step 表示 line2 分段长度

        返回 P2 的数组，P2 中 x 表示曲线 line2 上距离 s，y 表示前述距离对应的点的磁场 bz
        """
        ms: List[ValueWithDistance[P3]] = self.magnetic_field_along(
            line2, p2_t0_p3=p2_t0_p3, step=step
        )
        return [P2(p3d.distance, p3d.value.z) for p3d in ms]

    def graident_field_along(
            self,
            line2: Line2,
            good_field_area_width: float = 10 * MM,
            step: float = 1 * MM,
            point_number: int = 4,
    ) -> List[P2]:
        """
        计算本对象在二维曲线 line2 上的磁场梯度的分布
        每一点的梯度，采用这点水平垂线上 Bz 的多项式拟合得到
        line2 二维曲线，看作 z=0 的三维曲线。
        good_field_area_width：水平垂线的长度，注意应小于等于好场区范围
        step：line2 上取点间距
        point_number：水平垂线上取点数目，越多则拟合越精确
        """
        # 拟合阶数
        fit_order: int = 1

        # 自变量
        xs: List[float] = BaseUtils.linspace(
            -good_field_area_width / 2, good_field_area_width / 2, point_number
        )

        # line2 长度
        length = line2.get_length()

        # 离散距离
        distances = BaseUtils.linspace(0, length, int(length / step) + 1)

        # 返回值
        ret: List[P2] = []

        for s in distances:
            right_hand_point: P3 = line2.right_hand_side_point(
                s=s, d=good_field_area_width / 2
            ).to_p3()
            left_hand_point: P3 = line2.left_hand_side_point(
                s=s, d=good_field_area_width / 2
            ).to_p3()

            points: List[P3] = BaseUtils.linspace(
                right_hand_point, left_hand_point, point_number
            )

            # 磁场 bz
            ys: List[float] = [self.magnetic_field_at(p).z for p in points]

            # 拟合
            gradient: float = BaseUtils.polynomial_fitting(xs, ys, fit_order)[
                1]

            ret.append(P2(s, gradient))
        return ret

    def second_graident_field_along(
            self,
            line2: Line2,
            good_field_area_width: float = 10 * MM,
            step: float = 1 * MM,
            point_number: int = 4,
    ) -> List[P2]:
        """
        计算本对象在二维曲线 line2 上的磁场二阶梯度的分布（六极场）
        每一点的梯度，采用这点水平垂线上 Bz 的多项式拟合得到
        line2 二维曲线，看作 z=0 的三维曲线。
        good_field_area_width：水平垂线的长度，注意应小于等于好场区范围
        step：line2 上取点间距
        point_number：水平垂线上取点数目，越多则拟合越精确
        since v0.1.1
        """
        # 拟合阶数
        fit_order: int = 2

        # 自变量
        xs: List[float] = BaseUtils.linspace(
            -good_field_area_width / 2, good_field_area_width / 2, point_number
        )

        # line2 长度
        length = line2.get_length()

        # 离散距离
        distances = BaseUtils.linspace(0, length, int(length / step) + 1)

        # 返回值
        ret: List[P2] = []

        for s in distances:
            right_hand_point: P3 = line2.right_hand_side_point(
                s=s, d=good_field_area_width / 2
            ).to_p3()
            left_hand_point: P3 = line2.left_hand_side_point(
                s=s, d=good_field_area_width / 2
            ).to_p3()

            points: List[P3] = BaseUtils.linspace(
                right_hand_point, left_hand_point, point_number
            )

            # 磁场 bz
            ys: List[float] = [self.magnetic_field_at(p).z for p in points]

            # 拟合
            gradient: float = BaseUtils.polynomial_fitting(xs, ys, fit_order)[
                2] * 2.0  # 2021年5月1日 乘上 2.0

            ret.append(P2(s, gradient))
        return ret

    def multipole_field_along(
            self,
            line2: Line2,
            order: int,
            good_field_area_width: float = 10 * MM,
            step: float = 1 * MM,
            point_number: int = 10,
    ) -> List[ValueWithDistance[List[float]]]:
        """
        计算本对象在二维曲线 line2 上的各极谐波磁场分布
        line2 二维曲线，看作 z=0 的三维曲线
        order 谐波阶数，0阶求二级场，1阶求二四极场，2阶求二四六极场，以此类推
        good_field_area_width 好场区范围，在水平垂向取点时，从这个范围选取
        step 二维曲线 line2 步长，步长越多，返回的数组长度越长
        point_number 水平垂向取点数目，只有取点数目多于 order，才能正确拟合高次谐波

        实现于 2021年5月1日
        """
        # 自变量
        xs: List[float] = BaseUtils.linspace(
            -good_field_area_width / 2, good_field_area_width / 2, point_number
        )

        # line2 长度
        length = line2.get_length()

        # 离散距离
        distances = BaseUtils.linspace(0, length, int(length / step) + 1)

        # 返回值
        ret: List[ValueWithDistance[List[float]]] = []

        for s in distances:
            right_hand_point: P3 = line2.right_hand_side_point(
                s=s, d=good_field_area_width / 2
            ).to_p3()
            left_hand_point: P3 = line2.left_hand_side_point(
                s=s, d=good_field_area_width / 2
            ).to_p3()

            points: List[P3] = BaseUtils.linspace(
                right_hand_point, left_hand_point, point_number
            )

            # 磁场 bz
            ys: List[float] = [self.magnetic_field_at(p).z for p in points]

            # 拟合
            gradients: List[float] = BaseUtils.polynomial_fitting(
                xs, ys, order)

            # 乘系数，i次项乘上 i!
            for i in range(2, len(gradients)):
                gradients[i] = gradients[i] * math.factorial(i)

            ret.append(ValueWithDistance(value=gradients, distance=s))

        return ret

    def integration_field(
            self,
            line2: Line2,
            p2_t0_p3: Callable[[P2], P3] = lambda p2: P3(p2.x, p2.y, 0.0),
            step: float = 1 * MM,
    ) -> float:
        """
        计算本对象在二维曲线 line2 上的积分场
        line2     二维曲线
        p2_t0_p3 是一个函数，用于把 line2 上的二维点转为三维，默认转为 z=0 的三维点
        step     取点和积分的步长
        """
        fields = self.magnetic_field_bz_along(line2, p2_t0_p3, step)

        ret = 0.0

        for i in range(len(fields)-1):
            pre = fields[i].y
            post = fields[i+1].y

            ret += (pre+post)/2*step

        return ret

    # 此方法将放在 COSY 包中
    # def slice_to_cosy_script(
    #         self,
    #         Bp: float,
    #         aperture_radius: float,
    #         line2: Line2,
    #         good_field_area_width: float,
    #         min_step_length: float,
    #         tolerance: float,
    # ) -> str:
    #     """
    #     将本对象在由二维曲线 line2 切成 COSY 切片
    #     """
    #     raise NotImplementedError

    @staticmethod
    def no_magnet() -> 'Magnet':
        """
        返回一个不产生磁场的 Magnet
        实现代码完备性
        """
        return UniformMagnet(P3.zeros())

    @staticmethod
    def uniform_magnet(magnetic_field: P3 = P3.zeros()) -> 'Magnet':
        """
        返回一个产生匀强磁场的 Magnet
        """
        return UniformMagnet(magnetic_field)


class UniformMagnet(Magnet):
    """
    产生均匀磁场的磁铁
    """

    def __init__(self, magnetic_field: P3 = P3.zeros()) -> None:
        """
        构造器
        输入磁场 magnetic_field
        """
        super().__init__()
        self.magnetic_field = magnetic_field

    def magnetic_field_at(self, point: P3) -> P3:
        """
        任意点均产生相同磁场
        """
        return self.magnetic_field.copy()


class ApertureObject:
    """
    表示具有孔径的一个对象
    可以判断点 point 是在这个对象的孔径内还是孔径外

    只有当粒子轴向投影在元件内部时，才会进行判断，
    否则即时粒子距离轴线很远，也认为粒子没有超出孔径，
    这是因为粒子不在元件内时，很可能处于另一个大孔径元件中，这样会造成误判。
    """

    def is_out_of_aperture(self, point: P3) -> bool:
        """
        判断点 point 是在这个对象的孔径内还是孔径外
        只有当粒子轴向投影在元件内部时，才会进行判断，
        否则即时粒子距离轴线很远，也认为粒子没有超出孔径，
        这是因为粒子不在元件内时，很可能处于另一个大孔径元件中，这样会造成误判。
        """
        raise NotImplementedError


class QS(Magnet, ApertureObject):
    """
    硬边 QS 磁铁，由以下参数完全确定：

    length 磁铁长度 / m
    gradient 四极场梯度 / Tm-1
    second_gradient 六极场梯度 / Tm-2
    aperture_radius 孔径（半径） / m
    local_coordinate_system 局部坐标系
        根据局部坐标系确定 QS 磁铁的位置
         ③
         ↑
         |----------|
    -----①-------------->②
         |----------|
    ① QS 磁铁入口中心位置，是局部坐标系的原心
    ② 理想粒子运动方向，是局部坐标系 Z 方向
    ③ 相空间中 X 方向，由此可知垂直屏幕向外（向面部）是 Y 方向

    """

    def __init__(
            self,
            local_coordinate_system: LocalCoordinateSystem,
            length: float,
            gradient: float,
            second_gradient: float,
            aperture_radius: float,
    ):
        self.local_coordinate_system = local_coordinate_system
        self.length = float(length)
        self.gradient = float(gradient)
        self.second_gradient = float(second_gradient)
        self.aperture_radius = float(aperture_radius)

    def __str__(self) -> str:
        """
        since v0.1.1
        """
        return f"QS:local_coordinate_system={self.local_coordinate_system}, length={self.length}, gradient={self.gradient}, second_gradient={self.second_gradient}, aperture_radius={self.aperture_radius}"

    def __repr__(self) -> str:
        """
        since v0.1.1
        """
        return self.__str__()

    def magnetic_field_at(self, point: P3) -> P3:
        """
        qs 磁铁在点 point （全局坐标系点）处产生的磁场
        """
        # point 转为局部坐标
        p_local = self.local_coordinate_system.point_to_local_coordinate(point)
        x = p_local.x
        y = p_local.y
        z = p_local.z

        # z < 0 or z > self.length 表示点 point 位于磁铁外部
        if z < 0 or z > self.length:
            return P3.zeros()
        else:
            # 以下判断点 point 是不是在孔径外，前两个 or 是为了快速短路判断，避免不必要的开方计算
            if (
                    abs(x) > self.aperture_radius
                    or abs(y) > self.aperture_radius
                    or math.sqrt(x ** 2 + y ** 2) > self.aperture_radius
            ):
                return P3.zeros()
            else:
                # bx 和 by 分别是局部坐标系中 x 和 y 方向的磁场（局部坐标系中 z 方向是理想束流方向/中轴线反向，不会产生磁场）
                bx = self.gradient * y + self.second_gradient * (x * y)
                by = self.gradient * x + 0.5 * \
                    self.second_gradient * (x ** 2 - y ** 2)

                # 转移到全局坐标系中
                return (
                    self.local_coordinate_system.XI * bx
                    + self.local_coordinate_system.YI * by
                )

    # from ApertureObject
    def is_out_of_aperture(self, point: P3) -> bool:
        """
        判断点 point 是在 QS 的孔径内还是孔径外
        只有当粒子轴向投影在元件内部时，才会进行判断，
        否则即时粒子距离轴线很远，也认为粒子没有超出孔径，
        这是因为粒子不在元件内时，很可能处于另一个大孔径元件中，这样会造成误判。
        """
        # 转为局部坐标系
        local_point = self.local_coordinate_system.point_to_local_coordinate(
            point)

        if local_point.z >= 0 and local_point.z <= self.length:
            return (local_point.x ** 2 + local_point.y ** 2) > self.aperture_radius ** 2
        else:
            return False

    @staticmethod
    def create_qs_along(
            trajectory: Line2,
            s: float,
            length: float,
            gradient: float,
            second_gradient: float,
            aperture_radius: float,
    ) -> "QS":
        """
        按照设计轨迹 trajectory 的 s 处创建 QS 磁铁
        trajectory 二维设计轨道，因为轨道是二维的，处于 xy 平面，
            这也限制了 qs 磁铁的轴在 xy 平面上
            这样的限制影响不大，因为通常的束线设计，磁铁元件都会位于一个平面内
        s 确定 qs 磁铁位于设计轨道 trajectory 的位置，
            即沿着轨迹出发 s 距离处是 qs 磁铁的入口，同时此时轨迹的切向为 qs 磁铁的轴向
        length qs 磁铁的长度
        gradient 四极场梯度 / Tm-1
        second_gradient 六极场梯度 / Tm-2
        aperture_radius 孔径（半径） / m
        """
        origin: P2 = trajectory.point_at(s)
        z_direct: P2 = trajectory.direct_at(s)
        x_direct: P2 = z_direct.rotate(BaseUtils.angle_to_radian(90))

        lcs: LocalCoordinateSystem = LocalCoordinateSystem(
            location=origin.to_p3(),
            x_direction=x_direct.to_p3(),
            z_direction=z_direct.to_p3(),
        )

        return QS(
            local_coordinate_system=lcs,
            length=length,
            gradient=gradient,
            second_gradient=second_gradient,
            aperture_radius=aperture_radius,
        )
