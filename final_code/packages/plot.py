"""
CCT 建模优化代码
绘图代码

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
from packages.trajectory import Trajectory
from packages.particles import *
from packages.magnets import *
from packages.cct import CCT
from packages.beamline import Beamline


class Plot3:
    INIT: bool = False  # 是否初始化
    ax = None
    PLT = plt

    @staticmethod
    def __init():
        """
        初始化 Plot3
        自动检查，无需调用
        """
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
        plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

        fig = plt.figure()
        Plot3.ax = fig.gca(projection="3d")
        Plot3.ax.grid(False)

        Plot3.INIT = True

    @staticmethod
    def plot_xyz(x: float, y: float, z: float, describe="r.") -> None:
        """
        绘制点 (x,y,z)
        绘制图象时只有 plot_xyz 和 plot_xyz_array 访问底层，所以需要判断是否初始化
        """
        if not Plot3.INIT:
            Plot3.__init()

        Plot3.ax.plot(x, y, z, describe)

    @staticmethod
    def plot_xyz_array(
            xs: List[float], ys: List[float], zs: List[float], describe="r-"
    ) -> None:
        """
        绘制多个点
        按照 x y z 分别给值
        绘制图象时只有 plot_xyz 和 plot_xyz_array 访问底层，所以需要判断是否初始化
        """
        if not Plot3.INIT:
            Plot3.__init()

        Plot3.ax.plot(xs, ys, zs, describe)

    @staticmethod
    def plot_p3(p: P3, describe="r.") -> None:
        """
        绘制点 P3
        """
        Plot3.plot_xyz(p.x, p.y, p.z, describe)

    @staticmethod
    def plot_p3s(ps: List[P3], describe="r-") -> None:
        """
        绘制点 P3 数组，多个点
        """
        Plot3.plot_xyz_array(
            [p.x for p in ps], [p.y for p in ps], [p.z for p in ps], describe
        )

    @staticmethod
    def plot_line2(line2: Line2, step: float = 1 * MM, describe="r") -> None:
        """
        绘制 line2
        """
        Plot3.plot_p3s(line2.disperse3d(step=step), describe)

    @staticmethod
    def plot_line2s(
            line2s: List[Line2], steps: List[float] = [1 * MM], describes: List[str] = ["r"]
    ) -> None:
        """
        绘制多个 line2
        """
        length = len(line2s)
        for i in range(length):
            Plot3.plot_line2(
                line2=line2s[i],
                step=steps[i] if i < len(steps) else steps[-1],
                describe=describes[i] if i < len(describes) else describes[-1],
            )

    @staticmethod
    def plot_line3(line3: Line3, step: float = 1 * MM, describe="r") -> None:
        """
        绘制 line3
        """
        Plot3.plot_p3s(line3.disperse3d(step=step), describe)

    @staticmethod
    def plot_beamline(beamline: Beamline, describes=["r-"]) -> None:
        """
        绘制 beamline
        包括 beamline 上的磁铁和设计轨道
        """
        size = len(beamline.magnets)
        for i in range(1, size+1):
            b = beamline.magnets[i-1]
            d = describes[i] if i < len(describes) else describes[-1]
            if isinstance(b, QS):
                Plot3.plot_qs(b, d)
            elif isinstance(b, CCT):
                Plot3.plot_cct(b, d)
            elif isinstance(b, LocalUniformMagnet):
                Plot3.plot_local_uniform_magnet(b, d)
            else:
                print(f"无法绘制{b}")

        Plot3.plot_line2(beamline.trajectory, describe=describes[0])

    @staticmethod
    def plot_ndarry3ds(narray: numpy.ndarray, describe="r-") -> None:
        """
        绘制 numpy 数组
        """
        x = narray[:, 0]
        y = narray[:, 1]
        z = narray[:, 2]
        Plot3.plot_xyz_array(x, y, z, describe)

    @staticmethod
    def plot_cct(cct: CCT, describe="r-") -> None:
        """
        绘制 cct
        """
        cct_path3d: numpy.ndarray = cct.dispersed_path3
        cct_path3d_points: List[P3] = P3.from_numpy_ndarry(cct_path3d)
        cct_path3d_points: List[P3] = [
            cct.local_coordinate_system.point_to_global_coordinate(p)
            for p in cct_path3d_points
        ]
        Plot3.plot_p3s(cct_path3d_points, describe)

    @staticmethod
    def plot_qs(qs: QS, describe="r-") -> None:
        """
        绘制 qs
        """
        # 前中后三个圈
        front_circle_local = [
            P3(
                qs.aperture_radius * math.cos(i / 180 * numpy.pi),
                qs.aperture_radius * math.sin(i / 180 * numpy.pi),
                0.0,
            )
            for i in range(360)
        ]

        mid_circle_local = [p + P3(0, 0, qs.length / 2)
                            for p in front_circle_local]
        back_circle_local = [p + P3(0, 0, qs.length)
                             for p in front_circle_local]
        # 转到全局坐标系中
        front_circle = [
            qs.local_coordinate_system.point_to_global_coordinate(p)
            for p in front_circle_local
        ]
        mid_circle = [
            qs.local_coordinate_system.point_to_global_coordinate(p)
            for p in mid_circle_local
        ]
        back_circle = [
            qs.local_coordinate_system.point_to_global_coordinate(p)
            for p in back_circle_local
        ]

        Plot3.plot_p3s(front_circle, describe)
        Plot3.plot_p3s(mid_circle, describe)
        Plot3.plot_p3s(back_circle, describe)

        # 画轴线
        for i in range(0, 360, 10):
            Plot3.plot_p3s([front_circle[i], back_circle[i]], describe)

    @staticmethod
    def plot_local_uniform_magnet(local_uniform_magnet: LocalUniformMagnet, describe="r-") -> None:
        """
        绘制 LocalUniformMagnet
        """
        # 前中后三个圈
        front_circle_local = [
            P3(
                local_uniform_magnet.aperture_radius * math.cos(i / 180 * numpy.pi),
                local_uniform_magnet.aperture_radius * math.sin(i / 180 * numpy.pi),
                0.0,
            )
            for i in range(360)
        ]

        mid_circle_local = [p + P3(0, 0, local_uniform_magnet.length / 2)
                            for p in front_circle_local]
        back_circle_local = [p + P3(0, 0, local_uniform_magnet.length)
                             for p in front_circle_local]
        # 转到全局坐标系中
        front_circle = [
            local_uniform_magnet.local_coordinate_system.point_to_global_coordinate(p)
            for p in front_circle_local
        ]
        mid_circle = [
            local_uniform_magnet.local_coordinate_system.point_to_global_coordinate(p)
            for p in mid_circle_local
        ]
        back_circle = [
            local_uniform_magnet.local_coordinate_system.point_to_global_coordinate(p)
            for p in back_circle_local
        ]

        Plot3.plot_p3s(front_circle, describe)
        Plot3.plot_p3s(mid_circle, describe)
        Plot3.plot_p3s(back_circle, describe)

        # 画轴线
        for i in range(0, 360, 10):
            Plot3.plot_p3s([front_circle[i], back_circle[i]], describe)

    @staticmethod
    def plot_local_coordinate_system(
            local_coordinate_syste: LocalCoordinateSystem,
            axis_lengths: List[float] = [100 * MM] * 3,
            describe="r-",
    ) -> None:
        """
        绘制 local_coordinate_syste
        axis_lengths 各个轴的长度
        """
        origin = local_coordinate_syste.location
        xi = local_coordinate_syste.XI
        yi = local_coordinate_syste.YI
        zi = local_coordinate_syste.ZI

        Plot3.plot_p3s(ps=[origin, origin + xi *
                           axis_lengths[0]], describe=describe)
        Plot3.plot_p3s(ps=[origin, origin + yi *
                           axis_lengths[1]], describe=describe)
        Plot3.plot_p3s(ps=[origin, origin + zi *
                           axis_lengths[2]], describe=describe)

    @staticmethod
    def plot_running_particle(p:RunningParticle, describe="r.")->None:
        """
        绘制单个粒子，实际上绘制粒子的位置
        """
        Plot3.plot_p3(p.position,describe=describe)


    @staticmethod
    def plot_running_particles(ps:List[RunningParticle], describe="r.")->None:
        """
        绘制多个粒子，实际上绘制粒子的位置
        """
        Plot3.plot_p3s([p.position for p in ps],describe=describe)    


    @staticmethod
    def set_center(center: P3 = P3.origin(), cube_size: float = 1.0) -> None:
        """
        设置视界中心和范围
        因为范围是一个正方体，所以这个方法类似于 Plot2.equal()
        """
        p = P3(cube_size, cube_size, cube_size)
        Plot3.plot_p3(center - p, "w")
        Plot3.plot_p3(center + p, "w")

    @staticmethod
    def set_box(front_down_left: P3, back_top_right: P3) -> None:
        """
        设置视界范围
        按照立方体两个点设置
        """
        Plot3.plot_p3(front_down_left, "w")
        Plot3.plot_p3(back_top_right, "w")

    @staticmethod
    def off_axis() -> None:
        """
        去除坐标轴
        """
        Plot3.PLT.axis("off")

    @staticmethod
    def remove_background_color() -> None:
        """
        去除背景颜色
        """
        Plot3.ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        Plot3.ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        Plot3.ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    @staticmethod
    def show():
        """
        展示图象
        """
        if not Plot3.INIT:
            raise RuntimeError("Plot3::请在show前绘制图象")

        plt.show()

    @staticmethod
    def __logo__():
        """
        绘制 logo 并展示
        """
        LOGO = Trajectory.__cctpy__()
        Plot3.plot_line2s(LOGO, [1 * M], ["r-", "r-", "r-", "b-", "b-"])
        Plot3.plot_local_coordinate_system(
            LocalCoordinateSystem(location=P3(z=-0.5e-6)),
            axis_lengths=[1000, 200, 1e-6],
            describe="k-",
        )
        Plot3.off_axis()
        Plot3.remove_background_color()
        Plot3.ax.view_init(elev=20, azim=-79)
        Plot3.show()


class Plot2:
    INIT = False  # 是否初始化
    PLT = plt

    @staticmethod
    def __init():
        """
        初始化 Plot3
        自动检查，无需调用
        """
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
        plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

        Plot2.INIT = True

    @staticmethod
    def plot(*data, describe="r-") -> None:
        """
        绘制任意数据
        """
        param_length = len(data)
        if param_length == 1:
            param1 = data[0]
            if isinstance(param1, P2):
                Plot2.plot_p2(param1, describe=describe)
            elif isinstance(param1, P3):
                Plot2.plot_p3(param1, describe=describe)
            elif isinstance(param1, List):
                if isinstance(param1[0], P2):
                    Plot2.plot_p2s(param1, describe=describe)
                elif isinstance(param1[0], P3):
                    Plot2.plot_p3s(param1, describe=describe)
                else:
                    print(f"无法绘制{data}")
            elif isinstance(param1, numpy.ndarray):
                Plot2.plot_ndarry2ds(param1, describe=describe)
            elif isinstance(param1, CCT):
                Plot2.plot_cct_outline(param1, describe=describe)
            elif isinstance(param1, QS):
                Plot2.plot_qs(param1, describe=describe)
            elif isinstance(param1, LocalUniformMagnet):
                Plot2.plot_local_uniform_magnet(param1, describe=describe)
            elif isinstance(param1, Beamline):
                Plot2.plot_beamline(param1, describes=["k-", "r-"])
            elif isinstance(param1, Line2):
                Plot2.plot_line2(param1, describe=describe)
            elif isinstance(param1, BaseUtils.Ellipse):
                p2s = param1.uniform_distribution_points_along_edge(64)
                p2s.append(p2s[0])
                Plot2.plot(p2s, describe=describe)
            else:
                print(f"无法绘制{data}")

        elif param_length == 2:
            param1 = data[0]
            param2 = data[1]
            if (isinstance(param1, int) or isinstance(param1, float)) and (
                    isinstance(param2, int) or isinstance(param2, float)
            ):
                Plot2.plot_xy(param1, param2, describe=describe)
            elif isinstance(param1, List) and isinstance(param2, List):
                Plot2.plot_xy_array(param1, param2, describe=describe)
            else:
                Plot2.plot(param1, describe=describe)
                Plot2.plot(param2, describe=describe)
        else:
            for d in data:
                Plot2.plot(d, describe=describe)

    @staticmethod
    def plot_xy(x: float, y: float, describe="r.") -> None:
        """
        绘制点 (x,y)
        绘制图象时只有 plot_xy 和 plot_xy_array 访问底层，所以需要判断是否初始化
        """
        if not Plot2.INIT:
            Plot2.__init()

        if describe is None:
            plt.plot(x, y)
        else:
            plt.plot(x, y, describe)

        

    @staticmethod
    def plot_xy_array(xs: List[float], ys: List[float], describe="r-") -> None:
        """
        绘制多个点
        按照 x y 分别给值
        绘制图象时只有 plot_xy 和 plot_xy_array 访问底层，所以需要判断是否初始化
        """
        if not Plot2.INIT:
            Plot2.__init()
            
        if describe is None:
            plt.plot(xs, ys)
        else:
            plt.plot(xs, ys, describe)

    @staticmethod
    def plot_p2(p: P2, describe="r") -> None:
        """
        绘制点 P2
        """
        Plot2.plot_xy(p.x, p.y, describe)

    @staticmethod
    def plot_p3(
            p: P3, p3_to_p2: Callable = lambda p3: P2(p3.x, p3.y), describe="r"
    ) -> None:
        """
        绘制点 P3
        P3 按照策略 p3_to_p2z 转为 P2
        """
        Plot2.plot_p2(p3_to_p2(p), describe)

    @staticmethod
    def plot_p2s(ps: List[P2], describe="r-",circle:bool=False) -> None:
        """
        绘制点 P2 数组，多个点
        circle 是否画一个封闭的圆
        """
        ps_c = ps + [ps[0]] if circle else ps
        Plot2.plot_xy_array([p.x for p in ps_c], [p.y for p in ps_c], describe)

    @staticmethod
    def plot_p3s(
            ps: List[P3], p3_to_p2: Callable[[P3],P2] = lambda p3: P2(p3.x, p3.y), describe="r-"
    ) -> None:
        """
        绘制点 P3 数组，多个点
        P3 按照策略 p3_to_p2z 转为 P2
        """
        Plot2.plot_p2s([p3_to_p2(p) for p in ps], describe)

    @staticmethod
    def plot_ndarry2ds(narray: numpy.ndarray, describe="r-") -> None:
        """
        绘制 numpy 数组
        """
        x = narray[:, 0]
        y = narray[:, 1]
        Plot2.plot_xy_array(x, y, describe)

    @staticmethod
    def plot_cct_path2d(cct: CCT, describe="r-") -> None:
        """
        绘制 cct 二维图象，即 (ξ, φ)
        """
        cct_path2: numpy.ndarray = cct.dispersed_path2
        Plot2.plot_ndarry2ds(cct_path2, describe)

    @staticmethod
    def plot_cct_path3d_in_2d(cct: CCT, describe="r-") -> None:
        """
        绘制 cct
        仅仅将三维 CCT 路径映射到 xy 平面
        """
        cct_path3d: numpy.ndarray = cct.dispersed_path3
        cct_path3d_points: List[P3] = P3.from_numpy_ndarry(cct_path3d)
        cct_path3d_points: List[P3] = [
            cct.local_coordinate_system.point_to_global_coordinate(p)
            for p in cct_path3d_points
        ]
        cct_path2d_points: List[P2] = [p.to_p2() for p in cct_path3d_points]
        Plot2.plot_p2s(cct_path2d_points, describe)

    @staticmethod
    def plot_cct_outline(cct: CCT, describe="r-") -> None:
        R = cct.big_r
        r = cct.small_r
        lcs = cct.local_coordinate_system
        center = lcs.location

        phi0 = cct.starting_point_in_ksi_phi_coordinate.y
        phi1 = cct.end_point_in_ksi_phi_coordinate.y

        clockwise: bool = phi1 < phi0

        phi_length = abs(phi1 - phi0)

        arc = ArcLine2(
            starting_phi=phi0 + lcs.XI.to_p2().angle_to_x_axis(),  # 这个地方搞晕了
            center=center.to_p2(),
            radius=R,
            total_phi=phi_length,
            clockwise=clockwise,
        )

        left_points = []
        right_points = []

        for t in BaseUtils.linspace(0, arc.get_length(), 100):
            left_points.append(arc.left_hand_side_point(t, r))
            right_points.append(arc.right_hand_side_point(t, r))

        Plot2.plot_p2s(left_points, describe=describe)
        Plot2.plot_p2s(right_points, describe=describe)

        Plot2.plot_p2s([left_points[0], right_points[0]], describe=describe)
        Plot2.plot_p2s([left_points[-1], right_points[-1]], describe=describe)

    @staticmethod
    def plot_cct_outline_straight(location: float, cct: CCT, length: float, describe="r-") -> None:
        """
        直线版本，配合 plot_beamline_straight
        """
        start_point = P2(x=location)
        x = P2.x_direct()
        y = P2.y_direct()

        p1 = start_point + y*cct.small_r
        p4 = start_point - y*cct.small_r

        p2 = p1 + x*length
        p3 = p4 + x*length

        Plot2.plot_p2s([p1, p2, p3, p4, p1], describe=describe)

    @staticmethod
    def plot_qs(qs: QS, describe="r-") -> None:
        """
        绘制 qs
        """
        length = qs.length
        aper = qs.aperture_radius
        lsc = qs.local_coordinate_system
        origin = lsc.location

        outline = [
            origin,
            origin + lsc.XI * aper,
            origin + lsc.XI * aper + lsc.ZI * length,
            origin - lsc.XI * aper + lsc.ZI * length,
            origin - lsc.XI * aper,
            origin,
        ]

        outline_2d = [p.to_p2() for p in outline]
        Plot2.plot_p2s(outline_2d, describe)

    @staticmethod
    def plot_local_uniform_magnet(local_uniform_magnet: LocalUniformMagnet, describe="r-") -> None:
        """
        绘制 LocalUniformMagnet
        """
        length = local_uniform_magnet.length
        aper = local_uniform_magnet.aperture_radius
        lsc = local_uniform_magnet.local_coordinate_system
        origin = lsc.location

        outline = [
            origin,
            origin + lsc.XI * aper,
            origin + lsc.XI * aper + lsc.ZI * length,
            origin - lsc.XI * aper + lsc.ZI * length,
            origin - lsc.XI * aper,
            origin,
        ]


        outline_2d = [p.to_p2() for p in outline]
        Plot2.plot_p2s(outline_2d, describe)

    @staticmethod
    def plot_qs_straight(location: float, qs: QS, length: float, describe="k-") -> None:
        """
        绘制 qs
        轨道绘制为直线，配合 plot_beamline_straight
        """
        start_point = P2(x=location)
        x = P2.x_direct()
        y = None

        if qs.gradient >= 0:
            y = P2.y_direct()
        else:
            y = -P2.y_direct()

        p1 = start_point + x*length
        p2 = p1 + y*qs.aperture_radius
        p3 = start_point + y*qs.aperture_radius
        Plot2.plot_p2s([start_point, p1, p2, p3, start_point],
                       describe=describe)

    @staticmethod
    def plot_local_uniform_magnet_straight(location: float, local_uniform_magnet: LocalUniformMagnet, length: float, describe="k-") -> None:
        """
        绘制 local_uniform_magnet
        轨道绘制为直线，配合 plot_beamline_straight
        """
        start_point = P2(x=location)
        x = P2.x_direct()
        y = None

        if local_uniform_magnet.gradient >= 0:
            y = P2.y_direct()
        else:
            y = -P2.y_direct()

        p1 = start_point + x*length
        p2 = p1 + y*local_uniform_magnet.aperture_radius
        p3 = start_point + y*local_uniform_magnet.aperture_radius
        Plot2.plot_p2s([start_point, p1, p2, p3, start_point],
                       describe=describe)

    @staticmethod
    def plot_beamline(beamline: Beamline, describes=["r-"]) -> None:
        """
        绘制 beamline
        包括 beamline 上的磁铁和设计轨道
        注意：以轨道实际分布绘图
        """
        size = len(beamline.magnets)
        for i in range(size):
            b = beamline.magnets[i]
            d = describes[i + 1] if i < (len(describes) - 1) else describes[-1]
            if isinstance(b, QS):
                Plot2.plot_qs(b, d)
            elif isinstance(b, CCT):
                Plot2.plot_cct_outline(b, d)
            elif isinstance(b, LocalUniformMagnet):
                Plot2.plot_local_uniform_magnet(b, d)
            else:
                print(f"无法绘制{b}")
        Plot2.plot_line2(beamline.trajectory, describe=describes[0])

    @staticmethod
    def plot_beamline_straight(beamline: Beamline, describes=["k-"]) -> None:
        """
        绘制 beamline
        包括 beamline 上的磁铁和设计轨道
        注意：同上方法一致，但是将轨道绘制为直线
        CCT 磁铁绘制为二极铁形式
        QS 磁铁按照 Q 值是聚焦还是散焦，绘制为四极铁样式
        """
        size = len(beamline.elements)
        for i in range(size):
            loc = beamline.elements[i][0]
            b = beamline.elements[i][1]
            length = beamline.elements[i][2]

            d = describes[i + 1] if i < (len(describes) - 1) else describes[-1]
            if b == None:
                pass
            else:
                if isinstance(b, QS):
                    Plot2.plot_qs_straight(loc, b, length, describe=d)
                elif isinstance(b, CCT):
                    Plot2.plot_cct_outline_straight(loc, b, length, describe=d)
                elif isinstance(b, LocalUniformMagnet):
                    Plot2.plot_local_uniform_magnet_straight(loc, b, length, describe=d)
                else:
                    print(f"无法绘制{b}")
        Plot2.plot_p2s(
            [P2.origin(), P2(x=beamline.trajectory.get_length())], describe=describes[0])

    @staticmethod
    def plot_line2(line: Line2, step: float = 1 * MM, describe="r-") -> None:
        """
        绘制 line2

        refactor 0.1.1 分开绘制 Line2 和 Trajectory
        """
        if isinstance(line, Trajectory):
            Plot2.plot_trajectory(line, describes=describe)
        else:
            p2s = line.disperse2d(step)
            Plot2.plot_p2s(p2s, describe)

    @staticmethod
    def plot_trajectory(trajectory: Trajectory, describes=['r-', 'b-', 'k-']) -> None:
        """
        绘制 trajectory
        直线和弧线使用不同颜色

        since 0.1.1
        """
        line2_list = trajectory.get_line2_list()

        describe_straight_line = 'r-'
        describe_arc_line = 'b-'
        describe_aperture_objrcts = 'k-'

        if isinstance(describes, List):
            describe_straight_line = describes[0] if len(
                describes) >= 1 else 'r-'
            describe_arc_line = describes[1] if len(
                describes) >= 2 else describe_straight_line
            describe_aperture_objrcts = describes[2] if len(
                describes) >= 3 else describe_straight_line
        elif isinstance(describes, str):
            describe_straight_line = describes
            describe_arc_line = describes
            describe_aperture_objrcts = describes
        else:
            print(f"Plot2.plot_trajectory 参数describes异常{describes}")

        for l2 in line2_list:
            if isinstance(l2, StraightLine2):
                Plot2.plot_line2(l2, describe=describe_straight_line)
            elif isinstance(l2, ArcLine2):
                Plot2.plot_line2(l2, describe=describe_arc_line)
            elif isinstance(l2, Trajectory):
                Plot2.plot_trajectory(l2, describes)
            elif isinstance(l2, Line2):
                Plot2.plot_line2(l2, describe=describe_straight_line)
            else:
                print(f"无法绘制{l2}")

        # 绘制轮廓
        for a in trajectory.get_aperture_objrcts():
            Plot2.plot_line2(a, describe=describe_aperture_objrcts)

    @staticmethod
    def equal():
        """
        设置坐标轴比例相同
        """
        if not Plot2.INIT:
            Plot2.__init()
        plt.axis("equal")

    @staticmethod
    def xlim(x_min: float, x_max: float):
        """
        设置坐标轴范围

        since 0.1.4
        """
        if not Plot2.INIT:
            Plot2.__init()
        plt.xlim(x_min, x_max)

    @staticmethod
    def ylim(y_min: float, y_max: float):
        """
        设置坐标轴范围

        since 0.1.4
        """
        if not Plot2.INIT:
            Plot2.__init()
        plt.ylim(y_min, y_max)

    @staticmethod
    def info(
            x_label: str = "",
            y_label: str = "",
            title: str = "",
            font_size: int = 24,
            font_family: str = "Times New Roman",
    ) -> NoReturn:
        """
        设置文字标记
        """
        if not Plot2.INIT:
            Plot2.__init()

        font_label = {
            "family": font_family,
            "weight": "normal",
            "size": font_size,
        }
        plt.xlabel(xlabel=x_label, fontdict=font_label)
        plt.ylabel(ylabel=y_label, fontdict=font_label)
        plt.title(label=title, fontdict=font_label)

        plt.xticks(fontproperties=font_family, size=font_size)
        plt.yticks(fontproperties=font_family, size=font_size)

    @staticmethod
    def legend(*labels: Tuple, font_size: int = 24, font_family: str = "Times New Roman") -> NoReturn:
        """
        设置图例
        since v0.1.1
        """
        if not Plot2.INIT:
            Plot2.__init()

        font_label = {
            "family": font_family,
            "weight": "normal",
            "size": font_size,
        }

        plt.legend(labels=list(labels), prop=font_label)

    @staticmethod
    def show():
        """
        展示图象
        """
        if not Plot2.INIT:
            print("Plot2::请在show前调用plot")

        plt.show()

