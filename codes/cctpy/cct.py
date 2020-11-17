"""
构建 CCT 相关类
"""
from abc import ABC
from typing import List, Tuple

import numpy as np

from cctpy.abstract_classes import Magnet, LocalCoordinateSystem, Plotable


class SoleLayerCct(Magnet, Plotable):
    """
    单层 CCT
    由离散的路径 winding_path，电流 current 和 局部坐标系 local_coordinate_system 组成

    实际上这个类广义的多，可以表示空间中的任意导线
    """

    def __init__(self, winding_path: np.ndarray, current: float, local_coordinate_system: LocalCoordinateSystem):
        self.winding_path = winding_path
        self.current = current
        self.local_coordinate_system = local_coordinate_system

        # 电流元 current * (w[i+1] - w[i])
        self.elementary_current = current * (winding_path[1:] - winding_path[:-1])

        # 电流元的位置 (w[i+1]+w[i])/2
        self.elementary_current_position = 0.5 * (winding_path[1:] + winding_path[:-1])

    def magnetic_field_at(self, point: np.ndarray) -> np.ndarray:
        """
        单层 CCT 在点 point 处产生的磁场
        Parameters
        ----------
        point 空间任意一点（全局坐标系）

        Returns 这一点的磁场
        -------

        """
        # point 转为局部坐标
        p = self.local_coordinate_system.point_to_local_coordinate(point)

        # 点 p 到电流元中点
        r = p - self.elementary_current_position

        # 点 p 到电流元中点的距离的三次方
        rr = (np.linalg.norm(r, ord=2, axis=1) ** (-3)).reshape((r.shape[0], 1))

        # 计算每个电流元在 p 点产生的磁场 (此时还没有乘系数 μ0/4π )
        dB = np.cross(self.elementary_current, r) * rr

        # 求和，即得到磁场，记得乘以系数 μ0/4π = 1e-7
        B = np.sum(dB, axis=0) * 1e-7

        return B

    def line_and_color(self, describe='r') -> List[Tuple[np.ndarray, str]]:
        """
        画图相关
        Parameters
        ----------
        describe 线描述信息

        Returns 线径和描述信息
        -------

        """

        # 需要转成全局坐标系
        return [(self.local_coordinate_system.line_to_global_coordinate(self.winding_path), describe)]
