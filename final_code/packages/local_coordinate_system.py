"""
CCT 建模优化代码
局部坐标系

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
from packages.point import P3
from packages.constants import *
from packages.base_utils import BaseUtils


class LocalCoordinateSystem:
    """
    局部坐标系。
    各种磁铁都放置在局部坐标系中，而粒子在全局坐标系中运动，
    为了求磁铁在粒子位置产生的磁场，需要引入局部坐标的概念和坐标变换
    """

    def __init__(
            self,
            location: P3 = P3.origin(),
            x_direction: P3 = P3.x_direct(),
            z_direction: P3 = P3.z_direct(),
    ):
        """
        Parameters
        ----------
        location 全局坐标系中实体位置，默认全局坐标系的远点
        x_direction 局部坐标系 x 方向，默认全局坐标系 x 方向
        z_direction 局部坐标系 z 方向，默认全局坐标系 z 方向
        """
        BaseUtils.equal(
            x_direction.copy().normalize() * z_direction.copy().normalize(),
            0.0,
            err=1e-4,
            msg=f"创建 LocalCoordinateSystem 对象异常，x_direction{x_direction}和z_direction{z_direction}不正交",
        )

        # 局部坐标系，原心
        self.location: P3 = location.copy()

        # 局部坐标系的 x y z 三方向
        self.ZI: P3 = z_direction.copy().normalize()
        self.XI: P3 = x_direction.copy().normalize()
        self.YI: P3 = self.ZI @ self.XI

    def point_to_local_coordinate(self, global_coordinate_point: P3) -> P3:
        """
        全局坐标系 -> 局部坐标系
        Parameters
        ----------
        global_coordinate_point 全局坐标系中的点

        Returns 这一点在局部坐标系中的坐标
        -------
        """
        location_to_global_coordinate = global_coordinate_point - self.location
        x = self.XI * location_to_global_coordinate
        y = self.YI * location_to_global_coordinate
        z = self.ZI * location_to_global_coordinate
        return P3(x, y, z)

    def point_to_global_coordinate(self, local_coordinate_point: P3) -> P3:
        """
        局部坐标系 -> 全局坐标系
        Parameters
        ----------
        local_coordinate_point 局部坐标系

        Returns 全局坐标系
        -------

        """

        return self.location + (
            self.XI * local_coordinate_point.x
            + self.YI * local_coordinate_point.y
            + self.ZI * local_coordinate_point.z
        )

    def vector_to_local_coordinate(self, global_coordinate_vector: P3) -> P3:
        """
        全局坐标系 -> 局部坐标系
        Parameters
        ----------
        global_coordinate_point 全局坐标系中的矢量

        Returns 局部坐标系中的矢量坐标
        -------
        """
        x = self.XI * global_coordinate_vector
        y = self.YI * global_coordinate_vector
        z = self.ZI * global_coordinate_vector
        return P3(x, y, z)

    def vector_to_global_coordinate(self, local_coordinate_vector: P3) -> P3:
        """
        局部坐标系 -> 全局坐标系
        Parameters
        ----------
        local_coordinate_point 局部坐标中的矢量

        Returns 全局坐标系中的矢量
        -------

        """

        return (
            self.XI * local_coordinate_vector.x
            + self.YI * local_coordinate_vector.y
            + self.ZI * local_coordinate_vector.z
        )

    def __str__(self) -> str:
        """
        用于打印坐标轴信息
        """
        return f"LOCATION={self.location}, xi={self.XI}, yi={self.YI}, zi={self.ZI}"

    def __repr__(self) -> str:
        """
        同 __str__
        """
        return self.__str__()

    def __eq__(self, other: "LocalCoordinateSystem", err: float = 1e-6, msg: Optional[str] = None) -> bool:
        """
        判断两个坐标系是否相同
        err 指定绝对误差
        msg 如果指定，则判断结果为不相等时，抛出异常
        """
        return (
            BaseUtils.equal(self.location, other.location, err, msg) and
            BaseUtils.equal(self.XI, other.XI, err, msg) and
            BaseUtils.equal(self.YI, other.YI, err, msg) and
            BaseUtils.equal(self.ZI, other.ZI, err, msg)
        )

    @staticmethod
    def create_by_y_and_z_direction(
            location: P3, y_direction: P3, z_direction: P3
    ) -> "LocalCoordinateSystem":
        """
        由 原点 location y方向 y_direction 和 z方向 z_direction 创建坐标系
        Parameters
        ----------
        location 原点
        y_direction y方向
        z_direction z方向

        Returns 坐标系
        -------

        """
        BaseUtils.equal(
            y_direction * z_direction,
            0.0,
            msg=f"创建 LocalCoordinateSystem 对象异常，y_direction{y_direction}和z_direction{z_direction}不正交",
        )

        x_direction = y_direction @ z_direction
        return LocalCoordinateSystem(
            location=location, x_direction=x_direction, z_direction=z_direction
        )

    @staticmethod
    def global_coordinate_system() -> "LocalCoordinateSystem":
        """
        获取全局坐标系
        Returns 全局坐标系
        -------

        """
        return LocalCoordinateSystem()

    def copy(self: "LocalCoordinateSystem") -> "LocalCoordinateSystem":
        """
        无依赖拷贝坐标系

        since v0.1.3
        """
        return LocalCoordinateSystem(
            location=self.location.copy(),
            x_direction=self.XI.copy(),
            z_direction=self.ZI.copy()
        )
