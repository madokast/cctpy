""""
基础工具函数
"""
import logging
import math
from typing import Iterable, Callable, List

import numpy as np


class Converter:
    """
    转换器
    """

    @staticmethod
    def angle_to_radian(angle: float) -> float:
        """
        角度转弧度
        Parameters
        ----------
        angle 角度

        Returns 弧度
        -------

        """
        return angle * np.pi / 180.

    @staticmethod
    def radian_to_angle(radian: float) -> float:
        """
        弧度转角度
        Parameters
        ----------
        radian 弧度

        Returns 角度
        -------

        """
        return radian / np.pi * 180.


class Equal:
    """
    判断是否相等的工具类
    """

    @staticmethod
    def equal_float(a: float, b: float, err: float = 1e-10) -> bool:
        """
        判断浮点数是否相等
        a b 浮点数
        err 容许误差
        return a == b ?
        """
        return np.abs(a - b) < err

    @staticmethod
    def equal_vector(v1: np.ndarray, v2: np.ndarray, err: float = 1e-10) -> bool:
        """
        判断两个矢量是否相等
        err 允许误差
        retun 两个矢量只差的长度小于 err
        """
        diff = v1 - v2
        return Equal.equal_float(Vectors.length(abs(diff)), 0, err)

    @staticmethod
    def require_float_equal(a: float, b: float, msg: str, err: float = 1e-10) -> None:
        if not Equal.equal_float(a, b, err):
            logging.error(msg)

    @staticmethod
    def require_true(judge: bool, msg: str) -> None:
        if not judge:
            logging.error(msg)


class Vectors:
    """
    操作矢量的工具类
    """

    @staticmethod
    def length(vector: np.ndarray) -> float:
        """
        求矢量的长度
        vector 一维矢量，[a1,a2,a3...]
        return 矢量长度，sqrt(a1**2+a2**2+...)

        实际计算的是 ndarray 的二范数
        """
        return np.linalg.norm(vector, ord=2)

    @staticmethod
    def update_length(vector: np.ndarray, new_length: float) -> np.ndarray:
        """
        原地改变矢量的长度
        vector 矢量
        new_length 长度 应大于 0
        Returns vector
        """
        len0 = Vectors.length(vector)
        vector *= (new_length / len0)
        return vector

    @staticmethod
    def normalize_self(vector: np.ndarray) -> np.ndarray:
        """
        原地矢量长度归一化
        vector 应当是一个一维矢量，如 [a,b,c]
        将矢量的长度变为 1
        """
        return Vectors.update_length(vector, 1.)

    @staticmethod
    def rotate_self_z_axis(v: np.ndarray, phi: float) -> np.ndarray:
        """
        矢量绕 z 轴旋转 phi 弧度
        注意是原地操作，矢量 v 会发生变化，返回的不是新矢量
        Parameters
        ----------
        v 要被旋转的矢量
        phi 旋转弧度

        Returns v 自身
        -------

        """
        # 获取 x 和 y 坐标
        x = v[0]
        y = v[1]

        # 旋转矩阵
        r11 = np.cos(phi)
        r12 = -np.sin(phi)
        r21 = -r12
        r22 = r11

        # 旋转
        v[0] = r11 * x + r12 * y
        v[1] = r21 * x + r22 * y

        return v

    @staticmethod
    def angle_to_x_axis(v: np.ndarray) -> float:
        """
        矢量 this 和 x 轴的夹角
        具体计算方法同极坐标的 θ 角
        Parameters
        ----------
        v 矢量

        Returns 矢量 this 和 x 轴的夹角
        -------

        """
        return math.atan2(v[1], v[0])

    @staticmethod
    def create(x: float, y: float, z: float = float('nan')) -> np.ndarray:
        if math.isnan(z):
            return np.array([x, y], dtype=np.float64)
        else:
            return np.array([x, y, z], dtype=np.float64)

    @staticmethod
    def create_float32(x: float, y: float, z: float) -> np.ndarray:
        return np.array([x, y, z], dtype=np.float32)

    @staticmethod
    def empty() -> np.ndarray:
        return np.empty(3)

    @staticmethod
    def empty_float32() -> np.ndarray:
        return np.empty(3, dtype=np.float32)

    @staticmethod
    def random() -> np.ndarray:
        return np.random.randn(3)

    @staticmethod
    def random_float32() -> np.ndarray:
        return np.random.randn(3).astype(np.float32)

    @staticmethod
    def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.cross(a, b)


class Stream:
    """
    对象流，实质是链表 list
    为了链式调用
    """

    def __init__(self, li: List):
        self.__li = li

    @staticmethod
    def linspace(start, end, number):
        return Stream(np.linspace(start, end, number).tolist())

    def map(self, func: Callable):
        """
        核心方法，元素 map
        其实就是列表推导式，但是可以链式调用
        Parameters
        ----------
        func map函数

        Returns 新的流
        -------

        """
        return Stream([func(e) for e in self.__li])

    def to_list(self) -> List:
        return self.__li

    def to_vector(self) -> np.ndarray:
        return np.array(self.__li)

    def join(self, delimiter=' ') -> str:
        return delimiter.join(self.__li)


class Ellipse:
    """
    椭圆类
    Ax^2+Bxy+Cy^2=D
    """

    def __init__(self, A: float, B: float, C: float, D: float):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def point_at(self, theta: float) -> np.ndarray:
        """
        原点出发，方向th弧度的射线和椭圆Ax^2+Bxy+Cy^2=D的交点
        Parameters
        ----------
        theta 弧度

        Returns 方向th弧度的射线和椭圆Ax^2+Bxy+Cy^2=D的交点
        -------

        """
        d = np.zeros(2)

        while theta < 0:
            theta += 2 * np.pi

        while theta > 2 * np.pi:
            theta -= 2 * np.pi

        if Equal.equal_float(theta, 0) or Equal.equal_float(theta, 2 * np.pi):
            d[0] = np.sqrt(self.D / self.A)
            d[1] = 0

        if Equal.equal_float(theta, np.pi):
            d[0] = -np.sqrt(self.D / self.A)
            d[1] = 0

        t = 0.0

        if theta > 0 and theta < np.pi:
            t = 1 / np.tan(theta)
            d[1] = np.sqrt(self.D / (self.A * t * t + self.B * t + self.C))
            d[0] = t * d[1]

        if theta > np.pi and theta < 2 * np.pi:
            theta -= np.pi
            t = 1 / np.tan(theta)
            d[1] = -np.sqrt(self.D / (self.A * t * t + self.B * t + self.C))
            d[0] = t * d[1]

        return d

    @property
    def circumference(self) -> float:
        """
        计算椭圆周长
        Returns 计算椭圆周长
        -------

        """
        num: int = 3600 * 4
        c: float = 0.0
        for i in range(num):
            c += Vectors.length(
                self.point_at(2.0 * np.pi / float(num) * (i + 1)) -
                self.point_at(2.0 * np.pi / float(num) * (i))
            )

        return c

    def point_after(self, length: float) -> np.ndarray:
        """
        在椭圆 Ax^2+Bxy+Cy^2=D 上行走 length，返回此时的点
        规定起点：椭圆与X轴正方向的交点
        规定行走方向：逆时针
        Parameters
        ----------
        length 行走距离

        Returns 椭圆 Ax^2+Bxy+Cy^2=D 上行走 length，返回此时的点
        -------

        """
        step_theta = Converter.angle_to_radian(0.05)
        theta = 0.0
        while length > 0.0:
            length -= Vectors.length(
                self.point_at(theta + step_theta) -
                self.point_at(theta)
            )

            theta += step_theta

        return self.point_at(theta)

    def uniform_distribution_points_along_edge(self, num: int) -> np.ndarray:
        points = np.empty((num, 2))
        c = self.circumference
        for i in range(num):
            points[i, :] = self.point_after(c / num * i)

        return points


class Scalar:
    """
    标量，但是使用 np [] 表示，主要用于向 cuda 传值
    """

    @staticmethod
    def of(a: float) -> np.ndarray:
        return np.array([a])

    @staticmethod
    def of_float32(a: float) -> np.ndarray:
        return np.array([a], dtype=np.float32)

    @staticmethod
    def of_int32(a: int) -> np.ndarray:
        return np.array([a], dtype=np.int32)

    @staticmethod
    def empty() -> np.ndarray:
        return np.empty(1)

    @staticmethod
    def empty_float32() -> np.ndarray:
        return np.empty(1, dtype=np.float32)

    @staticmethod
    def random() -> np.ndarray:
        return np.random.randn(1)

    @staticmethod
    def random_float32() -> np.ndarray:
        return np.random.randn(1).astype(np.float32)


class Circle:
    """
    圆
    """

    @staticmethod
    def unit_circle(phi: float) -> np.ndarray:
        """
        单位圆（极坐标）
        返回：极坐标(r=1.0,phi=phi)的点的直角坐标(x,y)
        Parameters
        ----------
        phi 极坐标phi

        Returns 单位圆上的一点
        -------

        """
        x = np.cos(phi)
        y = np.sin(phi)

        return np.array([x, y])


class Debug:
    @staticmethod
    def print_traceback():
        """
        打印函数调用栈
        Returns
        -------

        """
        import sys

        f = sys._getframe()
        while f is not None:
            print(f)
            f = f.f_back
