"""
CCT 建模优化代码
二维曲线段

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


class Trajectory(Line2):
    """
    二维设计轨迹，由直线+圆弧组成
    """

    def __init__(self, first_line: Line2):
        """
        构造器，传入第一条线 first_line
        Parameters
        ----------
        first_line 第一条线
        -------

        """
        self.__trajectoryList = [first_line]
        self.__length = first_line.get_length()
        self.__point_at_error_happen = False  # 是否发生 point_at 错误
        self.__aperture_objrcts = []  # 用于绘制孔径轮廓，since 0.1.1

    def add_line2(self, line2: Line2) -> "Trajectory":
        """
        尾接任意二维曲线
        不判断是否和当前轨迹相接、相切

        since v0.1.3
        """
        self.__trajectoryList.append(line2)
        self.__length += line2.get_length()

    def add_strait_line(self, length: float) -> "Trajectory":
        """
        尾接直线
        Parameters
        ----------
        length 直线长度

        Returns self
        -------

        """
        last_line = self.__trajectoryList[-1]
        sp = last_line.point_at_end()
        sd = last_line.direct_at_end()

        sl = StraightLine2(length, sd, sp)

        self.__trajectoryList.append(sl)
        self.__length += length

        return self

    def add_arc_line(
            self, radius: float, clockwise: bool, angle_deg: float
    ) -> "Trajectory":
        """
        尾接圆弧
        Parameters
        ----------
        radius 半径
        clockwise 顺时针？
        angle_deg 角度

        Returns self
        -------

        """
        last_line = self.__trajectoryList[-1]
        sp = last_line.point_at_end()
        sd = last_line.direct_at_end()

        al = ArcLine2.create(sp, sd, radius, clockwise, angle_deg)

        self.__trajectoryList.append(al)
        self.__length += al.get_length()

        return self

    def get_length(self) -> float:
        """
        二维设计轨迹的长度
        """
        return self.__length

    def point_at(self, s: float) -> P2:
        """
        二维设计轨迹的 s 位置点
        """
        s0 = s

        for line in self.__trajectoryList:
            if s <= line.get_length():
                return line.point_at(s)
            else:
                s -= line.get_length()

        last_line = self.__trajectoryList[-1]

        # 2020年4月2日
        # 解决了一个因为浮点数产生的巨大bug
        if abs(s) <= 1e-8:
            return last_line.point_at_end()

        if not self.__point_at_error_happen:
            self.__point_at_error_happen = True
            print(f"ERROR Trajectory::point_at{s0}")
            BaseUtils.print_traceback()

        return last_line.point_at(s)

    def direct_at(self, s: float) -> P2:
        """
        二维设计轨迹的 s 位置方向
        """
        s0 = s

        for line in self.__trajectoryList:
            if s <= line.get_length():
                return line.direct_at(s)
            else:
                s -= line.get_length()

        last_line = self.__trajectoryList[-1]

        # 2020年4月2日
        # 解决了一个因为浮点数产生的巨大bug
        if abs(s) <= 1e-8:
            return last_line.direct_at_end()

        if not self.__point_at_error_happen:
            self.__point_at_error_happen = True
            print(f"ERROR Trajectory::direct_at{s0}")
            BaseUtils.print_traceback()

        return last_line.direct_at(s)

    def __str__(self) -> str:

        details = ["# {:0>2d} ".format(i+1)+self.__trajectoryList[i].__str__()
                   for i in range(len(self.__trajectoryList))]
        details = "\t\n".join(details)
        return f"Trajectory:\t\n{details}"

    def __repr__(self) -> str:
        return self.__str__()

    class __TrajectoryBuilder:
        """
        Trajectory 使用 set_start_point 进行构造时的中间产物
        """

        def __init__(self, start_point: P2):
            self.start_point = start_point

        def first_line(self, direct: P2 = P2.x_direct(), length: float = 1.0) -> "Trajectory":
            """
            设置 Trajectory 第一条直线段
            注意：Trajectory 只能以直线开头，不能以圆弧开头
            """
            return Trajectory(StraightLine2(length, direct, self.start_point))

    @staticmethod
    def set_start_point(start_point: P2 = P2.origin()) -> 'Trajectory.__TrajectoryBuilder':
        """
        设置 Trajectory 起点
        """
        return Trajectory.__TrajectoryBuilder(start_point)

    def get_line2_list(self) -> List[Line2]:
        """
        暴露内部 line2 list
        主要用于画图

        since 0.1.1
        """
        return self.__trajectoryList

    def get_last_line2(self) -> Line2:
        """
        获取 Trajectory 中 line2 数组中最后一个
        """
        return self.get_line2_list()[-1]

    def as_aperture_objrct_on_last(self, aperture_radius: float) -> "Trajectory":
        """
        给 traj 轨迹中最后一段添加孔径信息
        这个信息只用于绘图，无其他意义
        使用示例：
        -----------------------------
        Trajectory.set_start_point(P2.origin()).first_line(P2.x_direct(), DL1)
            .add_arc_line(radius=0.95, clockwise=False, angle_deg=45/2)
            .add_strait_line(length=GAP1)
            .add_strait_line(length=qs1_length).as_aperture_objrct_on_last(20*MM)
            .add_strait_line(length=GAP2)
            .add_strait_line(length=qs2_length).as_aperture_objrct_on_last(20*MM)
            .add_strait_line(length=GAP2)
            .add_strait_line(length=qs1_length).as_aperture_objrct_on_last(20*MM)
            .add_strait_line(length=GAP1)
            .add_arc_line(radius=0.95, clockwise=False, angle_deg=45/2)
        -----------------------------

        since 0.1.1
        """
        if len(self.__trajectoryList) == 0:
            print(f"无法将traj最后一段添加孔径轮廓，因为traj{self}为空")
        else:
            last_line = self.__trajectoryList[-1]
            if isinstance(last_line, StraightLine2):
                length = last_line.length
                direct = last_line.direct
                start_point = last_line.start_point

                self.__aperture_objrcts.append(StraightLine2(
                    length=length,
                    direct=direct,
                    start_point=start_point +
                    direct.rotate(BaseUtils.angle_to_radian(
                        90)).change_length(aperture_radius)
                ))
                self.__aperture_objrcts.append(StraightLine2(
                    length=length,
                    direct=direct,
                    start_point=start_point -
                    direct.rotate(BaseUtils.angle_to_radian(
                        90)).change_length(aperture_radius)
                ))
                self.__aperture_objrcts.append(StraightLine2(
                    length=aperture_radius*2,
                    direct=direct.rotate(BaseUtils.angle_to_radian(90)),
                    start_point=start_point -
                    direct.rotate(BaseUtils.angle_to_radian(
                        90)).change_length(aperture_radius)
                ))
                self.__aperture_objrcts.append(StraightLine2(
                    length=aperture_radius*2,
                    direct=direct.rotate(BaseUtils.angle_to_radian(90)),
                    start_point=(start_point -
                                 direct.rotate(BaseUtils.angle_to_radian(90)).change_length(aperture_radius) +
                                 direct.change_length(length))
                ))
            elif isinstance(last_line, ArcLine2):
                starting_phi = last_line.starting_phi
                center = last_line.center
                radius = last_line.radius
                total_phi = last_line.total_phi
                clockwise = last_line.clockwise
                self.__aperture_objrcts.append(ArcLine2(
                    starting_phi=starting_phi,
                    center=center,
                    radius=radius+aperture_radius,
                    total_phi=total_phi,
                    clockwise=clockwise,
                ))
                self.__aperture_objrcts.append(ArcLine2(
                    starting_phi=starting_phi,
                    center=center,
                    radius=radius-aperture_radius,
                    total_phi=total_phi,
                    clockwise=clockwise,
                ))

                direct_start = last_line.direct_at_start()
                direct_end = last_line.direct_at_end()
                point_start = last_line.point_at_start()
                point_end = last_line.point_at_end()
                self.__aperture_objrcts.append(StraightLine2(
                    length=aperture_radius*2,
                    direct=direct_start.rotate(BaseUtils.angle_to_radian(90)),
                    start_point=point_start -
                    direct_start.rotate(BaseUtils.angle_to_radian(
                        90)).change_length(aperture_radius)
                ))
                self.__aperture_objrcts.append(StraightLine2(
                    length=aperture_radius*2,
                    direct=direct_end.rotate(BaseUtils.angle_to_radian(90)),
                    start_point=point_end -
                    direct_end.rotate(BaseUtils.angle_to_radian(
                        90)).change_length(aperture_radius)
                ))
            else:
                print(f"无法给未知对象{last_line}添加孔径轮廓")

        return self

    def get_aperture_objrcts(self) -> List[Line2]:
        """
        暴露 __aperture_objrcts
        用于画图

        since 0.1.1 
        """
        return self.__aperture_objrcts

    @classmethod
    def __cctpy__(cls) -> List[Line2]:
        """
        彩蛋
        """
        width = 40
        c_angle = 300
        r = 1 * MM
        c_r = 100
        C1 = (
            Trajectory.set_start_point(start_point=P2(176, 88))
            .first_line(
                direct=P2.x_direct().rotate(
                    BaseUtils.angle_to_radian(360 - c_angle) / 2
                ),
                length=width,
            )
            .add_arc_line(radius=r, clockwise=False, angle_deg=90)
            .add_arc_line(radius=c_r, clockwise=False, angle_deg=c_angle)
            .add_arc_line(radius=r, clockwise=False, angle_deg=90)
            .add_strait_line(length=width)
            .add_arc_line(radius=r, clockwise=False, angle_deg=90)
            .add_arc_line(radius=c_r - width, clockwise=True, angle_deg=c_angle)
        )
        C2 = C1 + P2(200, 0)

        t_width = 190
        t_height = 190

        T = (
            Trajectory.set_start_point(start_point=P2(430, 155))
            .first_line(direct=P2.x_direct(), length=t_width)
            .add_arc_line(radius=r, clockwise=True, angle_deg=90)
            .add_strait_line(length=width)
            .add_arc_line(radius=r, clockwise=True, angle_deg=90)
            .add_strait_line(length=(t_width / 2 - width / 2))
            .add_arc_line(radius=r, clockwise=False, angle_deg=90)
            .add_strait_line(length=t_height - width)
            .add_arc_line(radius=r, clockwise=True, angle_deg=90)
            .add_strait_line(length=width)
            .add_arc_line(radius=r, clockwise=True, angle_deg=90)
            .add_strait_line(length=t_height - width)
            .add_arc_line(radius=r, clockwise=False, angle_deg=90)
            .add_strait_line(length=(t_width / 2 - width / 2))
            .add_arc_line(radius=r, clockwise=True, angle_deg=90)
            .add_strait_line(length=width)
        ) + P2(0, -5)

        p_height = t_height
        p_r = 50
        width = 45

        P_out = (
            Trajectory.set_start_point(start_point=P2(655, 155))
            .first_line(direct=P2.x_direct(), length=2 * width)
            .add_arc_line(radius=p_r, clockwise=True, angle_deg=180)
            .add_strait_line(length=width)
            .add_arc_line(radius=r, clockwise=False, angle_deg=90)
            .add_strait_line(length=p_height - p_r * 2)
            .add_arc_line(radius=r, clockwise=True, angle_deg=90)
            .add_strait_line(length=width)
            .add_arc_line(radius=r, clockwise=True, angle_deg=90)
            .add_strait_line(length=p_height)
        ) + P2(0, -5)

        P_in = (
            Trajectory.set_start_point(
                start_point=P_out.point_at(width) - P2(0, width * 0.6)
            )
            .first_line(direct=P2.x_direct(), length=width)
            .add_arc_line(radius=p_r - width * 0.6, clockwise=True, angle_deg=180)
            .add_strait_line(length=width)
            .add_arc_line(radius=r, clockwise=True, angle_deg=90)
            .add_strait_line(length=(p_r - width * 0.6) * 2)
        )

        width = 40
        y_width = 50
        y_heigt = t_height
        y_tilt_len = 120

        Y = (
            Trajectory.set_start_point(start_point=P2(810, 155))
            .first_line(direct=P2.x_direct(), length=width)
            .add_arc_line(radius=r, clockwise=True, angle_deg=60)
            .add_strait_line(length=y_tilt_len)
            .add_arc_line(radius=r, clockwise=False, angle_deg=120)
            .add_strait_line(length=y_tilt_len)
            .add_arc_line(radius=r, clockwise=True, angle_deg=60)
            .add_strait_line(length=width)
            .add_arc_line(radius=r, clockwise=True, angle_deg=120)
            .add_strait_line(length=y_tilt_len * 1.3)
            .add_arc_line(radius=r, clockwise=False, angle_deg=30)
            .add_strait_line(length=t_height * 0.4)
            .add_arc_line(radius=r, clockwise=True, angle_deg=90)
            .add_strait_line(length=width * 1.1)
            .add_arc_line(radius=r, clockwise=True, angle_deg=90)
            .add_strait_line(length=t_height * 0.4)
            .add_arc_line(radius=r, clockwise=False, angle_deg=30)
            .add_strait_line(length=y_tilt_len * 1.3)
        )

        return [C1, C2, T, P_in, P_out, Y]
