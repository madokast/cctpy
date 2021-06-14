"""
CCT 建模优化代码
磁铁类

作者：赵润晓
日期：2021年4月29日
"""

import multiprocessing  # since v0.1.1 多线程计算
import time  # since v0.1.1 统计计算时长
from typing import Callable, Dict, Generic, Iterable, List, NoReturn, Optional, Sequence, Tuple, TypeVar, Union
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

        order = 0，表示0阶，即返回二极场。
        order = 1，表示1阶，即返回[二极场,四极场]。
        order = 2，表示2阶，即返回[二极场,四极场,六极场]。


        返回值为 List[ValueWithDistance[List[float]]]，
        是 ValueWithDistance 对象的一个数组，表示沿着 line2 分布的各极谐波，
        各极谐波采用数组表示，数组下标0表示二级场，1表示四极场，2表示六极场，以此类推

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

    def multipole_field_along_line3(
            self,
            line3: Line3,
            order: int,
            good_field_area_width: float = 10 * MM,
            step: float = 1 * MM,
            point_number: int = 10,
            plane_direct: P3 = P3.z_direct()
    ) -> List[ValueWithDistance[List[float]]]:
        """
        计算本对象在三维曲线 line3 上的各极谐波磁场分布
        line3 三维曲线
        order 谐波阶数，0阶求二级场，1阶求二四极场，2阶求二四六极场，以此类推
        good_field_area_width 好场区范围，在水平垂向取点时，从这个范围选取
        step 三维曲线 line2 步长，步长越多，返回的数组长度越长
        point_number 水平垂向取点数目，只有取点数目多于 order，才能正确拟合高次谐波
        plane_direct 确定研究平面，见 Line3.right_hand_side_point() 方法的注释

        这个方法主要用于 COSY 三维轨迹切片

        实现于 2021年6月4日
        """
        # 自变量
        xs: List[float] = BaseUtils.linspace(
            -good_field_area_width / 2, good_field_area_width / 2, point_number
        )

        # line2 长度
        length = line3.get_length()

        # 离散距离
        distances = BaseUtils.linspace(0, length, int(length / step) + 1)

        # 返回值
        ret: List[ValueWithDistance[List[float]]] = []

        for s in distances:
            right_hand_point: P3 = line3.right_hand_side_point(
                s=s, d=good_field_area_width / 2, plane_direct=plane_direct
            )
            left_hand_point: P3 = line3.left_hand_side_point(
                s=s, d=good_field_area_width / 2, plane_direct=plane_direct
            )

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

    @staticmethod
    def combine(*magnets) -> 'Magnet':
        """
        多个 magnet 组合
        """
        for m in magnets:
            if not isinstance(m, Magnet):
                raise ValueError(f"{m} 不是磁铁对象")
        return CombinedMagnet(*magnets)


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

    def __str__(self) -> str:
        return f"均匀磁铁，磁场{self.magnetic_field}"


class LocalUniformMagnet(Magnet, ApertureObject):
    """
    局部圆柱区域产生均匀磁场的磁铁，可以看作一个直线二极磁铁
    local_coordinate_system 局部坐标系
        局部坐标系的原点即二极磁铁的起点
        局部坐标系的 z 方向即二极磁铁延申方向
        局部坐标系的 y 方向即磁场方向
    length 磁铁长度
    aperture_radius 磁铁孔径，磁铁外部磁场为零
    magnetic_field 磁场大小，标量。磁场方向由 local_coordinate_system 确定
    """

    def __init__(
            self,
            local_coordinate_system: LocalCoordinateSystem,
            length: float,
            aperture_radius: float,
            magnetic_field: float) -> None:
        """
        构造器
        输入磁场 magnetic_field
        """
        super().__init__()
        self.local_coordinate_system: LocalCoordinateSystem = local_coordinate_system
        self.length: float = float(length)
        self.aperture_radius = float(aperture_radius)
        self.magnetic_field = float(magnetic_field)

        self.magnetic_field_vector: P3 = self.local_coordinate_system.YI * self.magnetic_field

    def magnetic_field_at(self, point: P3) -> P3:
        """
        point 全局坐标系的点
        """
        lp = self.local_coordinate_system.point_to_local_coordinate(point)
        if lp.z < 0 or lp.z > self.length:
            return P3.zeros()
        else:
            if (abs(lp.x) > self.aperture_radius
                        or abs(lp.y) > self.aperture_radius
                        or math.sqrt(lp.x ** 2 + lp.y ** 2) > self.aperture_radius
                    ):
                return P3.zeros()
            else:
                return self.magnetic_field_vector.copy()

    def is_out_of_aperture(self, point: P3) -> bool:
        """
        判断点 point 是在这个对象的孔径内还是孔径外
        只有当粒子轴向投影在元件内部时，才会进行判断，
        否则即时粒子距离轴线很远，也认为粒子没有超出孔径，
        这是因为粒子不在元件内时，很可能处于另一个大孔径元件中，这样会造成误判。
        """
        lp = self.local_coordinate_system.point_to_local_coordinate(point)
        if lp.z < 0 or lp.z > self.length:
            return True
        else:
            if (abs(lp.x) > self.aperture_radius
                        or abs(lp.y) > self.aperture_radius
                        or math.sqrt(lp.x ** 2 + lp.y ** 2) > self.aperture_radius
                    ):
                return True
            else:
                return False

    @staticmethod
    def create_local_uniform_magnet_along(
            trajectory: Line2,
            s: float,
            magnetic_field: float,
            length: float,
            aperture_radius: float,
    ) -> "LocalUniformMagnet":
        """
        在设计轨道上创建 LocalUniformMagnet
        trajectory 设计轨道
        s 磁铁起点位于设计轨道上的位置
        magnetic_field 磁场大小，标量，磁场垂直于设计轨道
        length 磁铁长度
        aperture_radius 磁铁孔径
        """
        origin: P2 = trajectory.point_at(s)
        z_direct: P2 = trajectory.direct_at(s)
        x_direct: P2 = z_direct.rotate(BaseUtils.angle_to_radian(90))

        lcs: LocalCoordinateSystem = LocalCoordinateSystem(
            location=origin.to_p3(),
            x_direction=x_direct.to_p3(),
            z_direction=z_direct.to_p3(),
        )

        return LocalUniformMagnet(
            local_coordinate_system=lcs,
            length=length,
            aperture_radius=aperture_radius,
            magnetic_field=magnetic_field
        )


class CombinedMagnet(Magnet):
    """
    多个磁铁组合
    """

    def __init__(self, *magnets) -> None:
        """
        构造器
        输入磁场 magnetic_field
        """
        super().__init__()
        for magnet in magnets:
            if not isinstance(magnet, Magnet):
                raise ValueError(f"{magnet}不是磁铁对象")
        self.__magnets: List[Magnet] = list(magnets)

    def add(self, magnet: Magnet) -> "CombinedMagnet":
        """
        添加一个磁铁
        """
        self.__magnets.append(magnet)
        return self

    def get_magnets(self) -> List[Magnet]:
        """
        暴露内部磁铁数组
        """
        return self.__magnets

    def magnetic_field_at(self, point: P3) -> P3:
        """
        任意点均产生相同磁场
        """
        B = P3.zeros()
        for m in self.__magnets:
            B += m.magnetic_field_at(point)
        return B

    def remove(self, m: Magnet) -> 'CombinedMagnet':
        """
        移除一个磁场/磁铁
        """
        self.__magnets.remove(m)
        return self


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
        # print(aperture_radius)
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

    def to_numpy_array(self, numpy_dtype=numpy.float64) -> numpy.ndarray:
        """
        将 qs 磁铁转为 numpy 数组，用于 cuda 计算
        数组格式如下：
        坐标轴原点 ox oy oz
        坐标轴XI方向  xx xy xz
        坐标轴YI方向  yx yy yz
        坐标轴ZI方向  zx zy zz
        qs 长度
        qs 四极场梯度
        qs 六极场梯度
        qs 半孔径

        数组长度一共为 16
        """
        return numpy.array(
            self.local_coordinate_system.location.to_list() +
            self.local_coordinate_system.XI.to_list() +
            self.local_coordinate_system.YI.to_list() +
            self.local_coordinate_system.ZI.to_list() +
            [self.length, self.gradient, self.second_gradient, self.aperture_radius],
            dtype=numpy_dtype
        )

    def as_qs(anything) -> 'QS':
        """
        仿佛是类型转换
        实际啥也没做
        但是 IDE 就能根据返回值做代码提示了

        常用在将 Magnet 转成 QS
        例如从 Beamline 中取出的 magnets，然后按照真是类型转过去
        """
        return anything


class Q(Magnet, ApertureObject):
    """
    Q 铁，见 QS
    """

    def __init__(
            self,
            local_coordinate_system: LocalCoordinateSystem,
            length: float,
            gradient: float,
            aperture_radius: float,
    ) -> None:
        super().__init__()
        self.local_coordinate_system = local_coordinate_system
        self.length = float(length)
        self.gradient = float(gradient)
        self.aperture_radius = float(aperture_radius)

        self.qs = QS(
            local_coordinate_system=local_coordinate_system,
            length=length,
            gradient=gradient,
            second_gradient=0.0,
            aperture_radius=aperture_radius,
        )

    def __str__(self) -> str:
        """
        since v0.1.1
        """
        return f"Q:local_coordinate_system={self.local_coordinate_system}, length={self.length}, gradient={self.gradient}, aperture_radius={self.aperture_radius}"

    def __repr__(self) -> str:
        """
        since v0.1.1
        """
        return self.__str__()

    def magnetic_field_at(self, point: P3) -> P3:
        return self.qs.magnetic_field_at(point)

    def is_out_of_aperture(self, point: P3) -> bool:
        return self.qs.is_out_of_aperture(point)

    @staticmethod
    def create_q_along(
            trajectory: Line2,
            s: float,
            length: float,
            gradient: float,
            aperture_radius: float,
    ) -> "Q":
        origin: P2 = trajectory.point_at(s)
        z_direct: P2 = trajectory.direct_at(s)
        x_direct: P2 = z_direct.rotate(BaseUtils.angle_to_radian(90))

        lcs: LocalCoordinateSystem = LocalCoordinateSystem(
            location=origin.to_p3(),
            x_direction=x_direct.to_p3(),
            z_direction=z_direct.to_p3(),
        )

        return Q(
            local_coordinate_system=lcs,
            length=length,
            gradient=gradient,
            aperture_radius=aperture_radius,
        )

    def as_q(anything) -> 'Q':
        return anything
