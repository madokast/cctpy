"""
CCT 建模优化代码
三维曲线段

作者：赵润晓
日期：2021年4月27日
"""


from typing import Callable, Dict, Generic, Iterable, List, NoReturn, Optional, Tuple, TypeVar, Union
import matplotlib.pyplot as plt
import math
import sys
import os  # since v0.1.1 查看CPU核心数
from packages.point import *
from packages.constants import *
from packages.base_utils import BaseUtils


class Line3:
    """
    三维一条有方向的连续曲线段
    本类包含 3 个抽象方法，需要实现：
    get_length 获得曲线长度
    point_at 从曲线起点出发，s 位置处的点
    direct_at 从曲线起点出发，s 位置处曲线方向

    说明：这个类用于
        一。CCT 线圈的描述，用于计算洛伦兹力、骨架压力等等。
        二。用于 COSY 切片。（COSY 切片分为两种，二维设计轨道切片，三维粒子轨迹切片）

    """

    def get_length(self) -> float:
        """
        获得曲线的长度
        Returns 曲线的长度
        -------

        """
        raise NotImplementedError

    def point_at(self, s: float) -> P3:
        """
        获得曲线 s 位置处的点 (x,y,z)
        即从曲线起点出发，运动 s 长度后的位置
        Parameters
        ----------
        s 长度量，曲线上 s 位置

        Returns 曲线 s 位置处的点 (x,y,z)
        -------

        """
        raise NotImplementedError

    def direct_at(self, s: float) -> P3:
        """
        获得 s 位置处，曲线的方向
        Parameters
        ----------
        s 长度量，曲线上 s 位置

        Returns s 位置处，曲线的方向

        refactor 添加粗略实现
        -------

        """
        delta = 1e-7
        p1 = self.point_at(s)
        p2 = self.point_at(s+delta)
        return (p2-p1).normalize()

    def right_hand_side_point(self, s: float, d: float, plane_direct: P3) -> P3:
        """
        和 Line2 的方法 right_hand_side_point 类似
        只不过这里的右手侧，在矢量 plane_direct 确定的平面内进行，
        即 s 点处的方向 direct_at(s) 和右手侧共同确定的平面，与 plane_direct 垂直
        一般情况下 plane_direct 是全局坐标系的 z 方向
        """
        ps = self.point_at(s)

        # 方向
        ds = self.direct_at(s)

        return ps + (ds@plane_direct).change_length(d)

    def left_hand_side_point(self, s: float, d: float) -> P3:
        """
        左手侧。见 right_hand_side_point
        """
        return self.right_hand_side_point(s, -d)

    def right_hand_side_line3(self, d: float, plane_direct: P3) -> 'Line3':
        """
        右手侧曲线，即 right_hand_side_point 从 0 到 length 得到的曲线
        """
        return RightHandSideLine3(self, d, plane_direct)

    def disperse3d(self, step: float = 1.0 * MM) -> List[P3]:
        """
        离散轨迹点
        Parameters
        ----------
        step 步长

        Returns 离散轨迹点
        -------

        """
        number: int = int(math.ceil(self.get_length() / step)
                          ) + 1  # 这里要加 1，调整于 2021年4月28日
        return [
            self.point_at(s) for s in BaseUtils.linspace(0, self.get_length(), number)
        ]

    def disperse3d_with_distance(
            self, step: float = 1.0 * MM
    ) -> List[ValueWithDistance[P3]]:
        """
        同方法 disperse3d
        每个离散点带有距离，返回值是 ValueWithDistance[P3] 的数组
        """
        number: int = int(math.ceil(self.get_length() / step)
                          ) + 1  # 这里要加 1，调整于 2021年4月28日
        return [
            ValueWithDistance(self.point_at(s), s)
            for s in BaseUtils.linspace(0, self.get_length(), number)
        ]


class RightHandSideLine3(Line3):
    """
    右手侧曲线，用于 Line3.right_hand_side_line3 方法
    origin 原曲线
    d 右侧距离
    plane_direct 垂直平面，含义见 Line3.right_hand_side_point
    """

    def __init__(self, origin: Line3, d: float, plane_direct: P3) -> None:
        super().__init__()
        self.origin = origin
        self.d = d
        self.plane_direct = plane_direct

    def get_length(self) -> float:
        """
        曲线长度即原曲线长度
        """
        return self.origin.get_length()

    def point_at(self, s: float) -> P3:
        """
        曲线上 s 处的点，即原曲线右侧 d 位置处
        """
        return self.origin.right_hand_side_point(s, self.d, self.plane_direct)

    def direct_at(self, s: float) -> P3:
        """
        方向不变
        """
        return self.origin.direct_at(s)


class FunctionLine3(Line3):
    """
    函数解析式表述的 Line3
    一般用于任意线圈的构建、CCT 洛伦兹力的分析等

    注意：FunctionLine3 中 point_at(s) 和 direct_at(s)
    中的 s 不再是曲线位置 s，而是内部函数解析式自变量 s
    """

    def __init__(self, p3_function: Callable[[float], P3], start: float, end: float,
                 direct_function: Optional[Callable[[float], P3]] = None,
                 delta_for_compute_direct_function: float = 0.1*MM) -> None:
        """
        p3_function 曲线方程  p = p(s)
        start 曲线起点对应的自变量 s 值
        end 曲线终点对应的自变量 s 值
        direct_function 曲线方向方程， d = p'(s)，可以为空，若为空则 d = (p(s+Δ) - p(s))/Δ 计算而得
        delta_for_compute_direct_function 计算曲线方向方程 d(s) 时 Δ 取值，默认 0.1 毫米。同时还用于去曲线长度计算
        """
        super().__init__()
        self.p3_function = p3_function
        self.start = start
        self.end = end
        self.delta_for_compute_direct_function = delta_for_compute_direct_function

        if direct_function is None:
            def direct_function(s) -> P3:
                return (
                    self.p3_function(s+delta_for_compute_direct_function) -
                    self.p3_function(s)
                )/delta_for_compute_direct_function

        self.direct_function = direct_function

    def get_length(self) -> float:
        """
        曲线长度，注意不是 self.end-self.start
        """
        print("FunctionLine3的长度不是精确值，采用积分方法计算")

        disperse_line3 = self.disperse3d(
            step=self.delta_for_compute_direct_function)

        length = 0.0
        for i in range(len(disperse_line3)-1):
            pre = disperse_line3[i]
            cur = disperse_line3[i+1]
            length += (pre-cur).length()

        return length

    def point_at(self, s: float) -> P3:
        """
        s 不再是曲线位置 s，而是内部函数解析式自变量 s
        """
        return self.p3_function(s)

    def direct_at(self, s: float) -> P3:
        """
        s 不再是曲线位置 s，而是内部函数解析式自变量 s
        """
        return self.direct_function(s)


class TwoPointLine3(Line3):
    """
    两点确定的直线段 Line3
    这是一个不可变类
    """

    def __init__(self, p0: P3, p1: P3) -> None:
        super().__init__()
        self.__p0 = p0
        self.__p1 = p1

        # 提前计算好长度和方向
        self.__length = (p1-p0).length()
        self.__direct = (p1-p0).normalize()

    def get_length(self) -> float:
        return self.__length

    def point_at(self, s: float) -> P3:
        return self.__p0 + self.__direct*s

    def direct_at(self, s: float) -> P3:
        return self.__direct


class DiscretePointLine3(Line3):
    """
    由离散点描述的 Line3，离散点一般是粒子跟踪的轨迹

    实现方法是将离散点看成 TwoPointLine3，一段段拼接

    这是一个不可变类
    """

    def __init__(self, track: List[P3]) -> None:
        super().__init__()

        self.__two_point_line3s: List[TwoPointLine3] = []

        length = 0.0
        for i in range(len(track)-1):
            pre = track[i]
            cur = track[i+1]
            two_point_line3 = TwoPointLine3(pre, cur)

            length += two_point_line3.get_length()
            self.__two_point_line3s.append(two_point_line3)

        # 提前计算好长度
        self.__length = length

        # 两段曲线段数目
        self.__two_point_line3s_length = len(self.__two_point_line3s)

    def get_length(self) -> float:
        return self.__length

    def point_at(self, s: float) -> P3:
        index: int = 0
        line3_part: TwoPointLine3 = None
        while True:
            if index == self.__two_point_line3s_length:
                break
            line3_part = self.__two_point_line3s[index]
            if s < line3_part.get_length():
                break

            s -= line3_part.get_length()
            index += 1

        return line3_part.point_at(s)

    def direct_at(self, s: float) -> P3:
        index: int = 0
        line3_part: TwoPointLine3 = None
        while True:
            if index == self.__two_point_line3s_length:
                break
            line3_part = self.__two_point_line3s[index]
            if s < line3_part.get_length():
                break

            s -= line3_part.get_length()
            index += 1

        return line3_part.direct_at(s)
