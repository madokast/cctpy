"""
CCT 建模优化代码
二维曲线段

作者：赵润晓
日期：2021年4月27日
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


class Line2:
    """
    二维 xy 平面的一条有方向的连续曲线段，可以是直线、圆弧
    本类包含 3 个抽象方法，需要实现：
    get_length 获得曲线长度
    point_at 从曲线起点出发，s 位置处的点
    direct_at 从曲线起点出发，s 位置处曲线方向

    说明：这个类主要用于构建 “理想轨道”，理想轨道的用处很多：
    1. 获取理想轨道上的理想粒子；
    2. 研究理想轨道上的磁场分布

    """

    def get_length(self) -> float:
        """
        获得曲线的长度
        Returns 曲线的长度
        -------

        """
        raise NotImplementedError

    def point_at(self, s: float) -> P2:
        """
        获得曲线 s 位置处的点 (x,y)
        即从曲线起点出发，运动 s 长度后的位置
        Parameters
        ----------
        s 长度量，曲线上 s 位置

        Returns 曲线 s 位置处的点 (x,y)
        -------

        """
        raise NotImplementedError

    def direct_at(self, s: float) -> P2:
        """
        获得 s 位置处，曲线的方向
        Parameters
        ----------
        s 长度量，曲线上 s 位置

        Returns s 位置处，曲线的方向

        refactor v0.1.3 添加粗略实现
        -------

        """
        delta = 1e-7
        p1 = self.point_at(s)
        p2 = self.point_at(s+delta)
        return (p2-p1).normalize()

    def right_hand_side_point(self, s: float, d: float) -> P2:
        """
        位于 s 处的点，它右手边 d 处的点

         1    5    10     15
         -----------------@------>
         |2
         |4               *
        如上图，一条直线，s=15，d=4 ,即点 @ 右手边 4 距离处的点 *

        说明：这个方法，主要用于四极场、六极场的计算，因为需要涉及轨道横向位置的磁场

        Parameters
        ----------
        s 长度量，曲线上 s 位置
        d 长度量，d 距离远处

        Returns 位于 s 处的点，它右手边 d 处的点
        -------

        """
        ps = self.point_at(s)

        # 方向
        ds = self.direct_at(s)

        return ps + ds.copy().rotate(-math.pi / 2).change_length(d)

    def left_hand_side_point(self, s: float, d: float) -> P2:
        """
        位于 s 处的点，它左手边 d 处的点
        说明见 right_hand_side_point 方法
        Parameters
        ----------
        s 长度量，曲线上 s 位置
        d 长度量，d 距离远处

        Returns 位于 s 处的点，它左手边 d 处的点
        -------

        """
        return self.right_hand_side_point(s, -d)

    # ------------------------------端点性质-------------------- #
    def point_at_start(self) -> P2:
        """
        获得曲线 line 起点位置
        """
        return self.point_at(0.0)

    def point_at_end(self) -> P2:
        """
        获得曲线 line 终点位置
        """
        return self.point_at(self.get_length())

    def direct_at_start(self) -> P2:
        """
        获得曲线 line 起点方向
        """
        return self.direct_at(0.0)

    def direct_at_end(self) -> P2:
        """
        获得曲线 line 终点方向
        """
        return self.direct_at(self.get_length())

    # ------------------------------平移-------------------- #
    def __add__(self, v2: P2) -> "Line2":
        """
        Line2 的平移， v2 表示移动的方向和距离
        Parameters
        ----------
        v2 二维向量

        Returns 平移后的 Line2
        -------

        """

        class MovedLine2(Line2):
            def __init__(self, hold):
                self.hold = hold

            def get_length(self) -> float:
                return self.hold.get_length()

            def point_at(self, s: float) -> P2:
                return self.hold.point_at(s) + v2

            def direct_at(self, s: float) -> P2:
                return self.hold.direct_at(s)

        return MovedLine2(self)

    # ------------------------------ 离散 ------------------------#
    def disperse2d(self, step: float = 1.0 * MM) -> List[P2]:
        """
        二维离散轨迹点
        Parameters
        ----------
        step 步长

        Returns 二维离散轨迹点
        -------

        """
        number: int = int(math.ceil(self.get_length() / step)) + 1 # 这里要加 1，调整于 2021年4月28日
        return [
            self.point_at(s) for s in BaseUtils.linspace(0, self.get_length(), number)
        ]

    def disperse2d_with_distance(
            self, step: float = 1.0 * MM
    ) -> List[ValueWithDistance[P2]]:
        """
        同方法 disperse2d
        每个离散点带有距离，返回值是 ValueWithDistance[P2] 的数组
        """
        number: int = int(math.ceil(self.get_length() / step)) + 1 # 这里要加 1，调整于 2021年4月28日
        return [
            ValueWithDistance(self.point_at(s), s)
            for s in BaseUtils.linspace(0, self.get_length(), number)
        ]

    def disperse3d(
            self,
            p2_t0_p3: Callable[[P2], P3] = lambda p2: P3(p2.x, p2.y, 0.0),
            step: float = 1.0 * MM,
    ) -> List[P3]:
        """
        三维离散轨迹点，其中第三维 z == 0.0
        Parameters
        ----------
        step 步长
        p2_t0_p3：二维点 P2 到三维点 P3 转换函数，默认 z=0

        Returns 三维离散轨迹点
        -------

        """
        return [p.to_p3(p2_t0_p3) for p in self.disperse2d(step=step)]

    def disperse3d_with_distance(
            self,
            p2_t0_p3: Callable[[P2], P3] = lambda p2: P3(p2.x, p2.y, 0.0),
            step: float = 1.0 * MM,
    ) -> List[ValueWithDistance[P3]]:
        """
        同 disperse3d
        每个离散点带有距离，返回值是 ValueWithDistance[P3] 的数组
        """
        return [
            ValueWithDistance(vp2.value.to_p3(p2_t0_p3), vp2.distance)
            for vp2 in self.disperse2d_with_distance(step=step)
        ]

    def __str__(self) -> str:
        return f"Line2[起点{self.point_at_start()}，长度{self.get_length()}]"


class StraightLine2(Line2):
    """
    二维有向直线段，包含三个参数：长度、方向、起点
    """

    def __init__(self, length: float, direct: P2, start_point: P2):
        self.length = float(length)
        self.direct = direct
        self.start_point = start_point

    def get_length(self) -> float:
        """
        二维有向直线段的长度
        """
        return self.length

    def point_at(self, s: float) -> P2:
        """
        二维有向直线段 s 位置点
        """
        return self.start_point + self.direct.copy().change_length(s)

    def direct_at(self, s: float) -> P2:
        """
        二维有向直线段 s 位置方向
        """
        return self.direct

    def __str__(self) -> str:
        return f"直线段[起点{self.start_point}，方向{self.direct}，长度{self.length}]"

    def __repr__(self) -> str:
        return self.__str__()

    def position_of(self, p: P2) -> int:
        """
        求点 p 相对于直线段的方位
        返回值：
            1  在右侧
            -1 在左侧
            0  在直线段所在直线上
        因为直线段 self 是有方向的，所以可以确定左侧还是右侧
        这个函数用于确定 trajectory 当前是左偏还是右偏 / 逆时针偏转还是顺时针偏转
            #
        --------------&---->
                $
        如上图，对于点 # ，在直线左侧，返回 -1
        对于点 & 在直线上，返回 0
        对于点 $，在直线右侧，返回 1
        """
        p0 = self.start_point  # 直线段起点
        d = self.direct  # 直线方向

        p0_t0_p: P2 = p - p0  # 点 p0 到 p
        k: float = d * p0_t0_p  # 投影
        project: P2 = k * d  # 投影点

        vertical_line: P2 = p0_t0_p - project  # 点 p 的垂线方向

        # 垂线长度 0，说明点 p 在直线上
        if vertical_line == P2.zeros():
            return 0

        # 归一化
        vertical_line = vertical_line.normalize()
        right_hand: P2 = d.rotate(
            BaseUtils.angle_to_radian(-90)).normalize()  # 右手侧

        if vertical_line == right_hand:
            return 1
        else:
            return -1

    def straight_line_equation(self) -> Tuple[float, float, float]:
        """
        返回直线的一般式方程 A B C
        Ax + By + C = 0
        注意结果不唯一，不能用于比较
        具体计算方法如下 from https://www.zybang.com/question/7699174d2637a60b3db85a4bc2e82c95.html

        当x1=x2时，直线方程为x-x1=0
        当y1=y2时，直线方程为y-y1=0
        当x1≠x2，y1≠y2时，
        直线的斜率k=(y2-y1)/(x2-x1)
        故直线方程为y-y1=(y2-y1)/(x2-x1)×(x-x1)
        即x2y-x1y-x2y1+x1y1=(y2-y1)x-x1(y2-y1)
        即为(y2-y1)x-(x2-x1)y-x1(y2-y1)+(x2-x1)y1=0
        即为(y2-y1)x-(x2-x1)y-x1y2+x2y1=0
        A = Y2 - Y1
        B = X1 - X2
        C = X2*Y1 - X1*Y2

        since v0.1.3
        """
        if BaseUtils.equal(self.direct.length(), 0, err=1e-10):
            raise ValueError(
                f"straight_line_equation 直线方向矢量 direct 长度为 0，无法计算一般式方程")

        x1 = self.start_point.x
        y1 = self.start_point.y
        x2 = (self.direct+self.start_point).x
        y2 = (self.direct+self.start_point).y

        if BaseUtils.equal(x1, x2, err=1e-10):
            return 1.0, 0.0, -x1
        if BaseUtils.equal(y1, y2, err=1e-10):
            return 0.0, 1.0, -y1

        return (y2-y1), (x1-x2), (x2*y1-x1*y2)

    @staticmethod
    def intersecting_point(pa: P2, va: P2, pb: P2, vb: P2) -> Tuple[P2, float, float]:
        """
        求两条直线 a 和 b 的交点
        pa 直线 a 上的一点
        va 直线 a 方向
        pb 直线 b 上的一点
        vb 直线 b 方向

        返回值为交点 cp，交点在直线 a 和 b 上的坐标 ka kb
        即 cp = pa + va * ka = pb + vb * kb

        since v0.1.3
        """
        # 方向矢量不能为 0
        if BaseUtils.equal(va.length(), 0.0, err=1e-10):
            raise ValueError(
                f"intersecting_point：方向矢量 va 长度为零。pa={pa},pb={pb},va={va},vb={vb}")
        if BaseUtils.equal(vb.length(), 0.0, err=1e-10):
            raise ValueError(
                f"intersecting_point：方向矢量 vb 长度为零。pa={pa},pb={pb},va={va},vb={vb}")
        # 判断是否平行
        if va.normalize() == vb.normalize() or (va.normalize()+vb.normalize()) == P2.origin():
            print(
                f"intersecting_point：两条直线平行，计算结果可能无意义。pa={pa},pb={pb},va={va},vb={vb}")

        # pa 和 pb 就是交点。不短路，也走流程
        # if pa==pb:
        #     return pa,0.0,0.0

        # 计算交点
        # 为了防止除数为 0 ，只能将直线转为一般式
        line_a = StraightLine2(length=1.0, direct=va, start_point=pa)
        line_b = StraightLine2(length=1.0, direct=vb, start_point=pb)
        A1, B1, C1 = line_a.straight_line_equation()
        A2, B2, C2 = line_b.straight_line_equation()

        cpy = (A1*C2-A2*C1)/(A2*B1-A1*B2)
        cpx = -(B1*cpy+C1)/A1 if A1 != 0 else -(B2*cpy+C2)/A2
        cp = P2(cpx, cpy)

        ka = (cp.x-pa.x)/va.x if va.x != 0 else (cp.y-pa.y)/va.y
        kb = (cp.x-pb.x)/vb.x if vb.x != 0 else (cp.y-pb.y)/vb.y

        return cp, ka, kb

    @staticmethod
    def is_on_right(view_point: P2, view_direct: P2, viewed_point: P2) -> int:
        """
        查看点 viewed_point 是不是在右边
        观察点为 view_point 观测方向为 view_direct

        返回值
        1  在右侧
        0  在正前方或者正后方
        -1 在左侧
        """
        right_direct = view_direct.copy().rotate(BaseUtils.angle_to_radian(-90))
        relative_position = viewed_point-view_point

        k = right_direct*relative_position

        if k > 0:
            return 1
        elif k < 0:
            return -1
        else:
            return 0

    @staticmethod
    def calculate_k_b(p1: P2, p2: P2) -> Tuple[float]:
        """
        求过两点的直线方程
        y = kx + d

        k 和 d 的值
        """
        k = (p2.y-p1.y)/(p2.x-p1.x)
        b = p2.y - k * p2.x

        return (k, b)


class ArcLine2(Line2):
    """
    二维有向圆弧段
    借助极坐标的思想来描述圆弧
    基础属性： 圆弧的半径 radius、圆弧的圆心 center
    起点描述：极坐标 phi 值
    弧长：len = radius * totalPhi

    起点start_point、圆心center、半径radius、旋转方向clockwise、角度totalPhi 五个自由度
    起点弧度值 starting_phi、起点处方向、半径radius、旋转方向clockwise、角度totalPhi 五个自由度

    如图： *1 表示起点方向，@ 是圆心，上箭头 ↑ 是起点处方向，旋转方向是顺时针，*5 是终点，因此角度大约是 80 deg
                *5
           *4
       *3
     *2
    *1     ↑       @

    """

    def __init__(
            self,
            starting_phi: float,
            center: P2,
            radius: float,
            total_phi: float,
            clockwise: bool,
    ):
        self.starting_phi = starting_phi
        self.center = center
        self.radius = radius
        self.total_phi = total_phi
        self.clockwise = clockwise
        self.length = radius * total_phi

    def get_length(self) -> float:
        """
        二维有向圆弧段的长度
        """
        return self.length

    def point_at(self, s: float) -> P2:
        """
        二维有向圆弧段的 s 位置点
        """
        phi = s / self.radius
        current_phi = (
            self.starting_phi - phi if self.clockwise else self.starting_phi + phi
        )

        uc = ArcLine2.unit_circle(current_phi)

        return uc.change_length(self.radius) + self.center

    def direct_at(self, s: float) -> P2:
        """
        二维有向圆弧段的 s 位置方向
        """
        phi = s / self.radius
        current_phi = (
            self.starting_phi - phi if self.clockwise else self.starting_phi + phi
        )

        uc = ArcLine2.unit_circle(current_phi)

        return uc.rotate(-math.pi / 2 if self.clockwise else math.pi / 2)

    @staticmethod
    def create(
            start_point: P2,
            start_direct: P2,
            radius: float,
            clockwise: bool,
            total_deg: float,
    ) -> "ArcLine2":
        """
        利用起点、起点方向、半径、偏转角度创建二维有向圆弧段
        """
        center: P2 = start_point + start_direct.copy().rotate(
            -math.pi / 2 if clockwise else math.pi / 2
        ).change_length(radius)

        starting_phi = (start_point - center).angle_to_x_axis()

        total_phi = BaseUtils.angle_to_radian(total_deg)

        return ArcLine2(starting_phi, center, radius, total_phi, clockwise)

    @staticmethod
    def unit_circle(phi: float) -> P2:
        """
        单位圆（极坐标）
        返回：极坐标(r=1.0,phi=phi)的点的直角坐标(x,y)
        Parameters
        ----------
        phi 极坐标phi

        Returns 单位圆上的一点
        -------

        """
        x = math.cos(phi)
        y = math.sin(phi)

        return P2(x, y)

    def __str__(self) -> str:
        clock_wise_str = "顺时针" if self.clockwise else "逆时针"
        return (
            f"弧线段[起点{self.point_at_start()}，"
            + f"方向{self.direct_at_start()}，{clock_wise_str}，半径{self.radius}，角度{self.total_phi}]"
        )

    def __repr__(self) -> str:
        return self.__str__()

