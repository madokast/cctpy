"""
CCT 建模优化代码
束线

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
from packages.lines import *
from packages.trajectory import Trajectory
from packages.particles import *
from packages.magnets import *
from packages.cct import CCT


class Beamline(Line2, Magnet, ApertureObject):
    def __init__(self, trajectory: Trajectory) -> None:
        """
        不要直接调用构造器
        请使用 set_start_point
        """
        self.magnets: List[Magnet] = []
        self.trajectory: Trajectory = trajectory

        # 2021年3月18日 新增，表示元件。List 中每个元素表示一个元件
        # 元件由三部分组成，位置、元件自身、长度
        # 其中位置表示沿着 Beamline 的长度
        # 元件自身，使用 None 表示漂移段。
        self.elements: List[Tuple[float, Magnet, float]] = []

    def magnetic_field_at(self, point: P3) -> P3:
        """
        返回 Beamline 在全局坐标系点 P3 处产生的磁场
        """
        b: P3 = P3.zeros()
        for m in self.magnets:
            b += m.magnetic_field_at(point)
        return b

    # from Magnet
    def magnetic_field_along(
            self,
            line2: Optional[Line2] = None,
            p2_t0_p3: Callable[[P2], P3] = lambda p2: P3(p2.x, p2.y, 0.0),
            step: float = 1 * MM,
    ) -> List[ValueWithDistance[P3]]:
        """
        计算本对象在二维曲线 line2 上的磁场分布(line2 为 None 时，默认为 self.trajectory)
        p2_t0_p3 是一个函数，用于把 line2 上的二维点转为三维，默认转为 z=0 的三维点
        step 表示 line2 分段长度
        -------
        """
        if line2 is None:
            line2 = self.trajectory

        return super(Beamline, self).magnetic_field_along(
            line2=line2, p2_t0_p3=p2_t0_p3, step=step
        )

    def magnetic_field_bz_along(
            self,
            line2: Optional[Line2] = None,
            p2_t0_p3: Callable[[P2], P3] = lambda p2: P3(p2.x, p2.y, 0.0),
            step: float = 1 * MM,
    ) -> List[P2]:
        """
        计算本对象在二维曲线 line (line2 为 None 时，默认为 self.trajectory)上的磁场 Z 方向分量的分布
        因为磁铁一般放置在 XY 平面，所以 Bz 一般可以看作自然坐标系下 By，也就是二级场大小
        p2_t0_p3 是一个函数，用于把 line2 上的二维点转为三维，默认转为 z=0 的三维点
        step 表示 line2 分段长度

        返回 P2 的数组，P2 中 x 表示曲线 line2 上距离 s，y 表示前述距离对应的点的磁场 bz
        """
        if line2 is None:
            line2 = self.trajectory

        return super(Beamline, self).magnetic_field_bz_along(
            line2=line2, p2_t0_p3=p2_t0_p3, step=step
        )

    def graident_field_along(
            self,
            line2: Optional[Line2] = None,
            good_field_area_width: float = 10 * MM,
            step: float = 1 * MM,
            point_number: int = 4,
    ) -> List[P2]:
        """
        计算本对象在二维曲线 line2 (line2 为 None 时，默认为 self.trajectory)上的磁场梯度的分布
        每一点的梯度，采用这点水平垂线上 Bz 的多项式拟合得到
        good_field_area_width：水平垂线的长度，注意应小于等于好场区范围
        step：line2 上取点间距
        point_number：水平垂线上取点数目，越多则拟合越精确
        """
        if line2 is None:
            line2 = self.trajectory

        return super(Beamline, self).graident_field_along(
            line2=line2, good_field_area_width=good_field_area_width, step=step, point_number=point_number
        )

    def second_graident_field_along(
            self,
            line2: Optional[Line2] = None,
            good_field_area_width: float = 10 * MM,
            step: float = 1 * MM,
            point_number: int = 4,
    ) -> List[P2]:
        """
        计算本对象在二维曲线 line2 (line2 为 None 时，默认为 self.trajectory)上的磁场二阶梯度的分布（六极场）
        每一点的梯度，采用这点水平垂线上 Bz 的多项式拟合得到
        good_field_area_width：水平垂线的长度，注意应小于等于好场区范围
        step：line2 上取点间距
        point_number：水平垂线上取点数目，越多则拟合越精确
        """
        if line2 is None:
            line2 = self.trajectory

        return super(Beamline, self).second_graident_field_along(
            line2=line2, good_field_area_width=good_field_area_width, step=step, point_number=point_number
        )

    def track_ideal_particle(
            self,
            kinetic_MeV: float,
            s: float = 0.0,
            length: Optional[float] = None,
            footstep: float = 5 * MM,
    ) -> List[P3]:
        """
        束流跟踪，运行一个理想粒子，返回轨迹
        kinetic_MeV 粒子动能，单位 MeV
        s 起点位置
        length 粒子运行长度，默认运动到束线尾部
        footstep 粒子运动步长
        """
        if length is None:
            length = self.trajectory.get_length() - s
        ip = ParticleFactory.create_proton_along(
            self.trajectory, s, kinetic_MeV)
        return ParticleRunner.run_get_trajectory(ip, self, length, footstep)

    def track_phase_ellipse(
            self,
            x_sigma_mm: float,
            xp_sigma_mrad: float,
            y_sigma_mm: float,
            yp_sigma_mrad,
            delta: float,
            particle_number: int,
            kinetic_MeV: float,
            s: float = 0.0,
            length: Optional[float] = None,
            footstep: float = 10 * MM,
            concurrency_level: int = 1,
            report: bool = True
    ) -> Tuple[List[P2], List[P2]]:
        """
        束流跟踪，运行两个相椭圆边界上的粒子，
        返回一个长度 2 的元组，表示相空间 x-xp 平面和 y-yp 平面上粒子投影（单位 mm / mrad）
        两个相椭圆，一个位于 xxp 平面，参数为 σx 和 σxp ，动量分散为 delta
        另一个位于 xxp 平面，参数为 σx 和 σxp ，动量分散为 delta
        x_sigma_mm σx 单位 mm
        xp_sigma_mrad σxp 单位 mrad
        y_sigma_mm σy 单位 mm
        yp_sigma_mrad σyp 单位 mrad
        delta 动量分散 单位 1
        particle_number 粒子数目
        kinetic_MeV 动能 单位 MeV
        s 起点位置
        length 粒子运行长度，默认运行到束线尾部
        footstep 粒子运动步长
        concurrency_level 并发等级（使用多少个核心进行粒子跟踪）
        report 是否打印并行任务计划
        """
        if length is None:
            length = self.trajectory.get_length() - s
        ip_start = ParticleFactory.create_proton_along(
            self.trajectory, s, kinetic_MeV)
        ip_end = ParticleFactory.create_proton_along(
            self.trajectory, s + length, kinetic_MeV
        )

        pp_x = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_xxp_plane(
            xMax=x_sigma_mm * MM,
            xpMax=xp_sigma_mrad * MRAD,
            delta=delta,
            number=particle_number,
        )

        pp_y = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_yyp_plane(
            yMax=y_sigma_mm * MM,
            ypMax=yp_sigma_mrad * MRAD,
            delta=delta,
            number=particle_number,
        )

        rp_x = ParticleFactory.create_from_phase_space_particles(
            ideal_particle=ip_start,
            coordinate_system=ip_start.get_natural_coordinate_system(),
            phase_space_particles=pp_x,
        )

        rp_y = ParticleFactory.create_from_phase_space_particles(
            ideal_particle=ip_start,
            coordinate_system=ip_start.get_natural_coordinate_system(),
            phase_space_particles=pp_y,
        )

        # run
        # refactor v0.1.1 合并计算
        ParticleRunner.run_only(
            p=rp_x + rp_y, m=self, length=length, footstep=footstep, concurrency_level=concurrency_level,
            report=report
        )

        pp_x_end = PhaseSpaceParticle.create_from_running_particles(
            ideal_particle=ip_end,
            coordinate_system=ip_end.get_natural_coordinate_system(),
            running_particles=rp_x,
        )

        pp_y_end = PhaseSpaceParticle.create_from_running_particles(
            ideal_particle=ip_end,
            coordinate_system=ip_end.get_natural_coordinate_system(),
            running_particles=rp_y,
        )

        xs = [pp.project_to_xxp_plane() / MM for pp in pp_x_end]
        ys = [pp.project_to_yyp_plane() / MM for pp in pp_y_end]

        s = BaseUtils.Statistic()

        print(
            f"delta={delta}," +
            f"avg_size_x={s.clear().add_all(P2.extract(xs)[0]).helf_width()}mm," +
            f"avg_size_y={s.clear().add_all(P2.extract(ys)[0]).helf_width()}mm"
        )

        return (xs, ys)

    # from ApertureObject
    def is_out_of_aperture(self, point: P3) -> bool:
        """
        判断点 point 是否超出 Beamline 的任意一个元件的孔径
        只有当粒子轴向投影在元件内部时，才会进行判断，
        否则即时粒子距离轴线很远，也认为粒子没有超出孔径，
        这是因为粒子不在元件内时，很可能处于另一个大孔径元件中，这样会造成误判。

        注意：这个函数的效率极低！
        """
        for m in self.magnets:
            if isinstance(m, ApertureObject) and m.is_out_of_aperture(point):
                print(f"beamline在{m}位置超出孔径")
                return True

        return False

    def trace_is_out_of_aperture(
            self, trace_with_distance: List[ValueWithDistance[P3]]
    ) -> bool:
        """
        判断一条粒子轨迹是否超出孔径

        注意：这个函数的效率极低！
        """
        for pd in trace_with_distance:
            if self.is_out_of_aperture(pd.value):
                return True
        
        return False

    def get_length(self) -> float:
        """
        获得 Beamline 的长度
        """
        return self.trajectory.get_length()

    def point_at(self, s: float) -> P2:
        """
        获得 Beamline s 位置处的点 (x,y)
        -------

        """
        return self.trajectory.point_at(s)

    def direct_at(self, s: float) -> P2:
        """
        获得 Beamline s 位置处的方向
        """
        return self.trajectory.direct_at(s)

    class __BeamlineBuilder:
        """
        构建 Beamline 的中间产物
        """

        def __init__(self, start_point: P2) -> None:
            self.start_point = start_point

        def first_drift(self, direct: P2, length: float) -> "Beamline":
            """
            为 Beamline 添加第一个 drift
            正如 Trajectory 的第一个曲线段必须是是直线一样
            Beamline 中第一个元件必须是 drift
            """
            bl = Beamline(
                Trajectory.set_start_point(self.start_point).first_line(
                    direct=direct, length=length
                )
            )
            bl.elements.append((0, None, length))
            return bl

    @staticmethod
    def set_start_point(start_point: P2):  # -> "Beamline.__BeamlineBuilder"
        """
        设置束线起点
        """
        return Beamline.__BeamlineBuilder(start_point)

    def append_drift(self, length: float) -> "Beamline":
        """
        尾加漂移段
        length 漂移段长度
        """
        old_len = self.trajectory.get_length()
        self.trajectory.add_strait_line(length=length)
        self.elements.append((old_len, None, length))

        return self

    def append_qs(
            self,
            length: float,
            gradient: float,
            second_gradient: float,
            aperture_radius: float,
    ) -> "Beamline":
        """
        尾加 QS 磁铁

        length: float QS 磁铁长度
        gradient: float 梯度 T/m
        second_gradient: float 二阶梯度（六极场） T/m^2
        aperture_radius: float 半孔径 单位 m
        """
        old_length = self.trajectory.get_length()
        self.trajectory.add_strait_line(length=length)

        qs = QS.create_qs_along(
            trajectory=self.trajectory,
            s=old_length,
            length=length,
            gradient=gradient,
            second_gradient=second_gradient,
            aperture_radius=aperture_radius,
        )

        self.magnets.append(qs)
        self.elements.append((old_length, qs, length))

        return self

    def append_dipole_cct(
            self,
            big_r: float,
            small_r_inner: float,
            small_r_outer: float,
            bending_angle: float,
            tilt_angles: List[float],
            winding_number: int,
            current: float,
            disperse_number_per_winding: int = 120,
    ) -> "Beamline":
        """
        尾加二极CCT

        big_r: float 偏转半径
        small_r_inner: float 内层半孔径
        small_r_outer: float 外层半孔径
        bending_angle: float 偏转角度（正数表示逆时针、负数表示顺时针）
        tilt_angles: List[float] 各极倾斜角
        winding_number: int 匝数
        current: float 电流
        disperse_number_per_winding: int 每匝分段数目，越大计算越精确
        """
        old_length = self.trajectory.get_length()
        cct_length = big_r * abs(BaseUtils.angle_to_radian(bending_angle))
        self.trajectory.add_arc_line(
            radius=big_r, clockwise=bending_angle < 0, angle_deg=abs(bending_angle)
        )

        cct_inner = CCT.create_cct_along(
            trajectory=self.trajectory,
            s=old_length,
            big_r=big_r,
            small_r=small_r_inner,
            bending_angle=abs(bending_angle),
            tilt_angles=tilt_angles,
            winding_number=winding_number,
            current=current,
            starting_point_in_ksi_phi_coordinate=P2.origin(),
            end_point_in_ksi_phi_coordinate=P2(
                2 * math.pi * winding_number,
                BaseUtils.angle_to_radian(bending_angle),
            ),
            disperse_number_per_winding=disperse_number_per_winding,
        )
        self.magnets.append(cct_inner)
        self.elements.append((old_length, cct_inner, cct_length))

        cct_outer = CCT.create_cct_along(
            trajectory=self.trajectory,
            s=old_length,
            big_r=big_r,
            small_r=small_r_outer,
            bending_angle=abs(bending_angle),
            tilt_angles=BaseUtils.list_multiply(tilt_angles, -1),
            winding_number=winding_number,
            current=current,
            starting_point_in_ksi_phi_coordinate=P2.origin(),
            end_point_in_ksi_phi_coordinate=P2(
                -2 * math.pi * winding_number,
                BaseUtils.angle_to_radian(bending_angle),
            ),
            disperse_number_per_winding=disperse_number_per_winding,
        )
        self.magnets.append(cct_outer)
        self.elements.append((old_length, cct_outer, cct_length))

        return self

    def append_agcct(
            self,
            big_r: float,
            small_rs: List[float],
            bending_angles: List[float],
            tilt_angles: List[List[float]],
            winding_numbers: List[List[int]],
            currents: List[float],
            disperse_number_per_winding: int = 120,
    ) -> "Beamline":
        """
        尾加 agcct
        本质是两层二极 CCT 和两层交变四极 CCT

        big_r: float 偏转半径，单位 m
        small_rs: List[float] 各层 CCT 的孔径，一共四层，从大到小排列。分别是二极CCT外层、内层，四极CCT外层、内层
        bending_angles: List[float] 交变四极 CCT 每个 part 的偏转半径（正数表示逆时针、负数表示顺时针），要么全正数，要么全负数。不需要传入二极 CCT 偏转半径，因为它就是 sum(bending_angles)
        tilt_angles: List[List[float]] 二极 CCT 和四极 CCT 的倾斜角，典型值 [[30],[90,30]]，只有两个元素的二维数组
        winding_numbers: List[List[int]], 二极 CCT 和四极 CCT 的匝数，典型值 [[128],[21,50,50]] 表示二极 CCT 128匝，四极交变 CCT 为 21、50、50 匝
        currents: List[float] 二极 CCT 和四极 CCT 的电流，典型值 [8000,9000]
        disperse_number_per_winding: int 每匝分段数目，越大计算越精确

        添加 CCT 的顺序为：
        外层二极 CCT
        内层二极 CCT
        part1 四极 CCT 内层
        part1 四极 CCT 外层
        part2 四极 CCT 内层
        part2 四极 CCT 外层
        ... ... 
        """
        if len(small_rs) != 4:
            raise ValueError(
                f"small_rs({small_rs})，长度应为4，分别是二极CCT外层、内层，四极CCT外层、内层")
        if not BaseUtils.is_sorted(small_rs[::-1]):
            raise ValueError(
                f"small_rs({small_rs})，应从大到小排列，分别是二极CCT外层、内层，四极CCT外层、内层")

        total_bending_angle = sum(bending_angles)
        old_length = self.trajectory.get_length()
        cct_length = big_r * \
            abs(BaseUtils.angle_to_radian(total_bending_angle))
        self.trajectory.add_arc_line(
            radius=big_r,
            clockwise=total_bending_angle < 0,
            angle_deg=abs(total_bending_angle),
        )

        # 构建二极 CCT 外层
        cct2_outer = CCT.create_cct_along(
            trajectory=self.trajectory,
            s=old_length,
            big_r=big_r,
            small_r=small_rs[0],
            bending_angle=abs(total_bending_angle),
            tilt_angles=BaseUtils.list_multiply(tilt_angles[0], -1),
            winding_number=winding_numbers[0][0],
            current=currents[0],
            starting_point_in_ksi_phi_coordinate=P2.origin(),
            end_point_in_ksi_phi_coordinate=P2(
                -2 * math.pi * winding_numbers[0][0],
                BaseUtils.angle_to_radian(total_bending_angle),
            ),
            disperse_number_per_winding=disperse_number_per_winding,
        )
        self.magnets.append(cct2_outer)
        self.elements.append((old_length, cct2_outer, cct_length))

        # 构建二极 CCT 内层
        cct2_innter = CCT.create_cct_along(
            trajectory=self.trajectory,
            s=old_length,
            big_r=big_r,
            small_r=small_rs[1],
            bending_angle=abs(total_bending_angle),
            tilt_angles=tilt_angles[0],
            winding_number=winding_numbers[0][0],
            current=currents[0],
            starting_point_in_ksi_phi_coordinate=P2.origin(),
            end_point_in_ksi_phi_coordinate=P2(
                2 * math.pi * winding_numbers[0][0],
                BaseUtils.angle_to_radian(total_bending_angle),
            ),
            disperse_number_per_winding=disperse_number_per_winding,
        )
        self.magnets.append(cct2_innter)
        self.elements.append((old_length, cct2_innter, cct_length))

        # 构建内外侧四极交变 CCT
        # 提取参数
        agcct_small_r_out = small_rs[2]
        agcct_small_r_in = small_rs[3]
        agcct_winding_nums: List[int] = winding_numbers[1]
        agcct_bending_angles: List[float] = bending_angles
        agcct_bending_angles_rad: List[float] = BaseUtils.angle_to_radian(
            agcct_bending_angles
        )
        agcct_tilt_angles: List[float] = tilt_angles[1]
        agcct_current: float = currents[1]

        # 构建 part1
        agcct_index = 0
        agcct_start_in = P2.origin()
        agcct_start_out = P2.origin()
        agcct_end_in = P2(
            ((-1.0) ** agcct_index) * 2 * math.pi *
            agcct_winding_nums[agcct_index],
            agcct_bending_angles_rad[agcct_index],
        )
        agcct_end_out = P2(
            ((-1.0) ** (agcct_index + 1))
            * 2
            * math.pi
            * agcct_winding_nums[agcct_index],
            agcct_bending_angles_rad[agcct_index],
        )
        agcct_part1_inner = CCT.create_cct_along(
            trajectory=self.trajectory,
            s=old_length,
            big_r=big_r,
            small_r=agcct_small_r_in,
            bending_angle=abs(agcct_bending_angles[agcct_index]),
            tilt_angles=BaseUtils.list_multiply(agcct_tilt_angles, -1),
            winding_number=agcct_winding_nums[agcct_index],
            current=agcct_current,
            starting_point_in_ksi_phi_coordinate=agcct_start_in,
            end_point_in_ksi_phi_coordinate=agcct_end_in,
            disperse_number_per_winding=disperse_number_per_winding,
        )
        agcct_part1_length = big_r * \
            BaseUtils.angle_to_radian(abs(agcct_bending_angles[agcct_index]))
        self.magnets.append(agcct_part1_inner)
        self.elements.append(
            (old_length, agcct_part1_inner, agcct_part1_length))

        agcct_part1_outer = CCT.create_cct_along(
            trajectory=self.trajectory,
            s=old_length,
            big_r=big_r,
            small_r=agcct_small_r_out,
            bending_angle=abs(agcct_bending_angles[agcct_index]),
            tilt_angles=agcct_tilt_angles,
            winding_number=agcct_winding_nums[agcct_index],
            current=agcct_current,
            starting_point_in_ksi_phi_coordinate=agcct_start_out,
            end_point_in_ksi_phi_coordinate=agcct_end_out,
            disperse_number_per_winding=disperse_number_per_winding,
        )
        self.magnets.append(agcct_part1_outer)
        self.elements.append(
            (old_length, agcct_part1_outer, agcct_part1_length))

        old_length_i = old_length + agcct_part1_length
        # 构建 part2 和之后的 part
        for ignore in range(len(agcct_bending_angles) - 1):
            agcct_index += 1
            agcct_start_in = agcct_end_in + P2(
                0,
                agcct_bending_angles_rad[agcct_index - 1]
                / agcct_winding_nums[agcct_index - 1],
            )
            agcct_start_out = agcct_end_out + P2(
                0,
                agcct_bending_angles_rad[agcct_index - 1]
                / agcct_winding_nums[agcct_index - 1],
            )
            agcct_end_in = agcct_start_in + P2(
                ((-1) ** agcct_index) * 2 * math.pi *
                agcct_winding_nums[agcct_index],
                agcct_bending_angles_rad[agcct_index],
            )
            agcct_end_out = agcct_start_out + P2(
                ((-1) ** (agcct_index + 1))
                * 2
                * math.pi
                * agcct_winding_nums[agcct_index],
                agcct_bending_angles_rad[agcct_index],
            )
            agcct_parti_inner = CCT.create_cct_along(
                trajectory=self.trajectory,
                s=old_length,
                big_r=big_r,
                small_r=agcct_small_r_in,
                bending_angle=abs(agcct_bending_angles[agcct_index]),
                tilt_angles=BaseUtils.list_multiply(agcct_tilt_angles, -1),
                winding_number=agcct_winding_nums[agcct_index],
                current=agcct_current,
                starting_point_in_ksi_phi_coordinate=agcct_start_in,
                end_point_in_ksi_phi_coordinate=agcct_end_in,
                disperse_number_per_winding=disperse_number_per_winding,
            )
            agcct_parti_length = big_r * \
                BaseUtils.angle_to_radian(
                    abs(agcct_bending_angles[agcct_index]))
            self.magnets.append(agcct_parti_inner)
            self.elements.append(
                (old_length_i, agcct_parti_inner, agcct_parti_length))

            agcct_parti_outer = CCT.create_cct_along(
                trajectory=self.trajectory,
                s=old_length,
                big_r=big_r,
                small_r=agcct_small_r_out,
                bending_angle=abs(agcct_bending_angles[agcct_index]),
                tilt_angles=agcct_tilt_angles,
                winding_number=agcct_winding_nums[agcct_index],
                current=agcct_current,
                starting_point_in_ksi_phi_coordinate=agcct_start_out,
                end_point_in_ksi_phi_coordinate=agcct_end_out,
                disperse_number_per_winding=disperse_number_per_winding,
            )
            self.magnets.append(agcct_parti_outer)
            self.elements.append(
                (old_length_i, agcct_parti_outer, agcct_parti_length))

            old_length_i += agcct_parti_length

        return self

    def __str__(self) -> str:
        return f"beamline(magnet_size={len(self.magnets)}, traj_len={self.trajectory.get_length()})"

