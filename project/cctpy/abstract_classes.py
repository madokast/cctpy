"""
抽象 / 一般性对象
"""

import numpy as np
from typing import List, Tuple
import logging

from cctpy.baseutils import Vectors, Equal
from cctpy.constant import ORIGIN3, XI, ZI


class Magnet:
    """
    表示一个可以求磁场的对象，如 CCT 、 QS 磁铁
    所有实现此接口的类，可以计算出它在某一点的磁场

    本类（接口）只有一个接口方法 magnetic_field_at(self, point: np.ndarray)
    """

    def magnetic_field_at(self, point: np.ndarray) -> np.ndarray:
        """
        获得本对象 self 在点 point 处产生的磁场
        这个方法需要在子类中实现/重写
        ----------
        point 三维笛卡尔坐标系中的点，即一个三维矢量，如 [0,0,0]

        Returns 本对象 self 在点 point 处的磁场，用三维矢量表示
        -------
        """
        raise NotImplementedError

    def magnetic_field_along(self, line: np.ndarray) -> np.ndarray:
        """
        计算本对象在三维曲线 line 上的磁场分布
        ----------
        line 由离散点组成的三维曲线，即三维矢量的数组，如 [[0,0,0], [1,0,0], [2,0,0]]

        Returns 本对象在三维曲线 line 上的磁场分布，用三维矢量的数组表示
        -------
        """
        length = line.shape[0]  # 曲线上离散点的数目
        fields = np.empty((length, 3), dtype=np.float64)  # 提前开辟空间
        for i in range(length):
            fields[i, :] = self.magnetic_field_at(line[i, :])
        return fields


class LocalCoordinateSystem:
    """
    局部坐标系，各种磁铁需要指定它所在的局部坐标系才能产生磁场，同时也便于磁铁调整位置

    局部坐标系由位置 location 、主方向 main_direction 和次方向 second_direction 确定

    一般而言，默认元件入口中心处，即元件的位置

    一般而言，主方向 main_direction 表示理想粒子运动方向，一般是 z 方向
    次方向 second_direction 垂直于主方向，并且在相空间分析中看作 x 方向
    """

    def __init__(self, location: np.ndarray = ORIGIN3, main_direction: np.ndarray = ZI,
                 second_direction: np.ndarray = XI):
        """
        指定实体的位置和朝向
        Parameters
        ----------
        location 全局坐标系中实体位置，默认全局坐标系的远点
        main_direction 主朝向，默认全局坐标系 z 方向
        second_direction 次朝向，默认全局坐标系 x 方向
        """
        Equal.require_float_equal(
            np.inner(main_direction, second_direction), 0.0,
            f"创建 LocalCoordinateSystem 对象异常，main_direction{main_direction}和second_direction{second_direction}不正交"
        )

        # 局部坐标系，原心
        self.location = location.copy()

        # 局部坐标系的 x y z 三方向
        self.ZI = Vectors.normalize_self(main_direction.copy())
        self.XI = Vectors.normalize_self(second_direction.copy())
        self.YI = np.cross(self.ZI, self.XI)

    def point_to_local_coordinate(self, global_coordinate_point: np.ndarray) -> np.ndarray:
        """
        全局坐标系 -> 局部坐标系
        Parameters
        ----------
        global_coordinate_point 全局坐标系中的点

        Returns 这一点在局部坐标系中的坐标
        -------
        """
        location_to_global_coordinate = global_coordinate_point - self.location
        x = np.inner(self.XI, location_to_global_coordinate)
        y = np.inner(self.YI, location_to_global_coordinate)
        z = np.inner(self.ZI, location_to_global_coordinate)
        return np.array([x, y, z], dtype=np.float64)

    def line_to_local_coordinate(self, global_coordinate_line: np.ndarray) -> np.ndarray:
        length = global_coordinate_line.shape[0]
        location_line = np.empty((length, 3), dtype=np.float64)  # 提前开辟空间
        for i in range(length):
            location_line[i, :] = self.point_to_local_coordinate(global_coordinate_line[i, :])
        return location_line

    def set_location(self, location: np.ndarray):
        self.location = location.copy()

    def set_direction(self, main_direction: np.ndarray, second_direction: np.ndarray) -> None:
        self.ZI = Vectors.normalize_self(main_direction.copy())
        self.XI = Vectors.normalize_self(second_direction.copy())
        self.YI = np.cross(self.ZI, self.XI)

    @staticmethod
    def create_by_y_and_z_direction(location: np.ndarray, y_direction: np.ndarray, z_direction: np.ndarray):
        Equal.require_float_equal(
            np.inner(y_direction, z_direction), 0.0,
            f"创建 LocalCoordinateSystem 对象异常，y_direction{y_direction}和z_direction{z_direction}不正交"
        )

        x_direction = np.cross(y_direction, z_direction)
        return LocalCoordinateSystem(location, z_direction, x_direction)


class Plotable:
    """
    表示一个可以进行绘图的对象
    """

    def line_and_color(self, describe='r') -> List[Tuple[np.ndarray, str]]:
        """
        返回用于绘图的 线 和 绘图选项（如线型、线颜色、线粗细，默认红色）
        Returns [(线，绘图选项)]
        -------
        """
        raise NotImplementedError
