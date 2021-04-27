"""
CCT 建模优化全套解决方案
用户手册见
开发手册见 cctpy_developer_manual.pdf

v0.1   初版 2020年12月3日
v0.1.1
"""
import multiprocessing  # since v0.1.1 多线程计算
import time  # since v0.1.1 统计计算时长
from typing import Callable, Generic, List, NoReturn, Optional, Tuple, TypeVar, Union
import matplotlib.pyplot as plt
import math
import sys
import os  # since v0.1.1 查看CPU核心数
import numpy
from scipy.integrate import solve_ivp  # since v0.1.1 ODE45
import warnings  # since v0.1.1 提醒方法过失

# 常量
M: float = 1.0  # 一米
MM: float = 0.001  # 一毫米
LIGHT_SPEED: float = 299792458.0 * M  # 光速
RAD: float = 1.0  # 一弧度
MRAD: float = 1.0 * MM * RAD  # 一毫弧度
J: float = 1.0  # 焦耳
eV = 1.6021766208e-19 * J  # 电子伏特转焦耳
MeV = 1000 * 1000 * eV  # 兆电子伏特
MeV_PER_C = 5.3442857792e-22  # kg m/s 动量单位
T = TypeVar("T")  # 泛型，仅用于类型标记


class P2:
    """
    二维点 / 二维向量
    """

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = float(x)
        self.y = float(y)

    def length(self) -> float:
        """
        求矢量长度
        """
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalize(self) -> "P2":
        """
        矢量长度归一，返回新矢量
        """
        return self * (1 / self.length())

    def change_length(self, new_length: float) -> "P2":
        """
        改变长度，返回新矢量
        """
        return self.normalize() * float(new_length)

    def copy(self) -> "P2":
        """
        拷贝
        """
        return P2(self.x, self.y)

    def __add__(self, other) -> "P2":
        """
        矢量加法，返回新矢量
        """
        return P2(self.x + other.x, self.y + other.y)

    def __neg__(self) -> "P2":
        """
        相反方向的矢量
        """
        return P2(-self.x, -self.y)

    def __sub__(self, other) -> "P2":
        """
        矢量减法，返回新矢量
        """
        return self.__add__(other.__neg__())

    def __iadd__(self, other) -> "P2":
        """
        矢量原地相加，self 自身改变
        """
        self.x += other.x
        self.y += other.y
        return self  # 必须显式返回

    def __isub__(self, other) -> "P2":
        """
        矢量原地减法，self 自身改变
        """
        self.x -= other.x
        self.y -= other.y
        return self

    def __matmul(self, m: List[List[float]]) -> "P2":
        """
        2*2矩阵和 self 相乘，仅仅用于矢量旋转
        """
        return P2(
            m[0][0] * self.x + m[0][1] *
            self.y, m[1][0] * self.x + m[1][1] * self.y
        )

    @staticmethod
    def __rotate_r(phi: float) -> List[List[float]]:
        """
        旋转矩阵
        """
        return [[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]]

    def rotate(self, phi: float) -> "P2":
        """
        矢量自身旋转
        """
        return self.__matmul(P2.__rotate_r(phi))

    def angle_to_x_axis(self) -> float:
        """
        矢量和 x 轴的夹角，弧度
        """
        a = float(math.atan2(self.y, self.x))
        return a if a >= 0 else math.pi * 2 + a

    def __mul__(self, other: Union[float, int, "P2"]) -> Union[float, "P2"]:
        """
        矢量乘标量，各元素相乘，返回新矢量
        矢量乘矢量，内积，返回标量
        """
        if isinstance(other, float) or isinstance(other, int):
            return P2(self.x * other, self.y * other)
        else:
            return self.x * other.x + self.y * other.y

    def __rmul__(self, other: Union[float, int]) -> "P2":
        """
        当左操作数不支持相应的操作时被调用
        """
        return self.__mul__(other)

    def __truediv__(self, number: Union[int, float]) -> "P2":
        """
        矢量除法 p2 / number，实际上是 p2 * (1/number)
        """
        if isinstance(number, int) or isinstance(number, float):
            return self * (1 / number)
        else:
            raise ValueError("P2仅支持数字除法")

    def angle_to(self, other: "P2") -> float:
        """
        矢量 self 到 另一个矢量 other 的夹角
        """
        to_x = self.angle_to_x_axis()
        s = self.rotate(-to_x)
        o = other.rotate(-to_x)
        return o.angle_to_x_axis()
        # 下面求的仅仅是 矢量 self 和 另一个矢量 other 的夹角
        # theta = (self * other) / (self.length() * other.length())
        # return math.acos(theta)

    def to_p3(
            self, transformation: Callable[["P2"], "P3"] = lambda p2: P3(p2.x, p2.y, 0.0)
    ) -> "P3":
        """
        二维矢量转为三维
        默认情况返回 [x,y,0]
        """
        return transformation(self)

    def __str__(self) -> str:
        """
        用于打印矢量值
        """
        return f"[{self.x}, {self.y}]"

    def __repr__(self) -> str:
        """
        == __str__ 用于打印矢量值
        """
        return self.__str__()

    def __eq__(self, other: "P2", err: float = 1e-6, msg: Optional[str] = None) -> bool:
        """
        矢量相等判断
        """
        return BaseUtils.equal(self.x, other.x, err, msg) and BaseUtils.equal(
            self.y, other.y, err, msg
        )

    @staticmethod
    def x_direct(x: float = 1.0) -> "P2":
        """
        返回 x 方向的矢量，或者 x 轴上的点
        """
        return P2(x=x)

    @staticmethod
    def y_direct(y: float = 1.0) -> "P2":
        """
        返回 y 方向的矢量，或者 y 轴上的点
        """
        return P2(y=y)

    @staticmethod
    def origin() -> "P2":
        """
        返回原点
        """
        return P2()

    @staticmethod
    def zeros() -> "P2":
        """
        返回零矢量
        """
        return P2()

    def to_list(self) -> List[float]:
        """
        p2 点 (x,y) 转为数组 [x,y]
        """
        return [self.x, self.y]

    @staticmethod
    def from_numpy_ndarry(ndarray: numpy.ndarray) -> Union["P2", List["P2"]]:
        """
        将 numpy 数组转为 P2，可以适应不同形状的数组
        当数组为 1*2 或 2*1 时，转为单个 P2 点
        当数组为 n*2 转为 P2 数组
        举例如下
        array([1, 2]) ==》P2 [1.0, 2.0]
        array([[1],
               [2]])  ==》P2 [1.0, 2.0]
        array([[1, 2],
               [3, 4],
               [5, 6]])  ==》List[P2] [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        array([[1, 2, 3],
               [4, 5, 6]]) ==》 ValueError: 无法将[[1 2 3], [4 5 6]]转为P2或List[P2]
        """
        if ndarray.shape == (2,) or ndarray.shape == (2, 1):
            return P2(ndarray[0], ndarray[1])
        elif len(ndarray.shape) == 2 and ndarray.shape[1] == 2:
            return [P2.from_numpy_ndarry(sub_array) for sub_array in ndarray]
        else:
            raise ValueError(f"无法将{ndarray}转为P2或List[P2]")

    @staticmethod
    def extract(p2_list: List['P2']) -> Tuple[List[float], List[float]]:
        """
        分别抽取 P2 数组中的 x 坐标和 y 坐标
        举例如下
        p2_list = [1,2], [2,3], [5,4]
        则返回 [1,2,5] 和 [2,3,4]
        这个方法主要用于 matplotlib 绘图
        since v0.1.1
        """
        return ([
            p.x for p in p2_list
        ], [
            p.y for p in p2_list
        ])


class P3:
    """
    三维点 / 三维矢量
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def length(self) -> float:
        """
        矢量长度
        """
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self) -> "P3":
        """
        正则化，返回新矢量
        """
        return self * (1 / self.length())

    def change_length(self, new_length: float) -> "P3":
        """
        改变长度，返回新矢量
        """
        return self.normalize() * new_length

    def copy(self) -> "P3":
        """
        拷贝
        """
        return P3(self.x, self.y, self.z)

    def __add__(self, other) -> "P3":
        """
        矢量相加
        """
        return P3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __neg__(self) -> "P3":
        """
        相反矢量
        """
        return P3(-self.x, -self.y, -self.z)

    def __sub__(self, other) -> "P3":
        """
        矢量相减
        """
        return self.__add__(other.__neg__())

    def __iadd__(self, other) -> "P3":
        """
        矢量原地相加
        """
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self

    def __isub__(self, other) -> "P3":
        """
        矢量原地减法
        """
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z
        return self

    def __mul__(self, other: Union[float, int, "P3"]) -> Union[float, "P3"]:
        """
        矢量乘标量，各元素相乘，返回新矢量
        矢量乘矢量，内积，返回标量
        """
        if isinstance(other, float) or isinstance(other, int):
            return P3(self.x * other, self.y * other, self.z * other)
        else:
            return self.x * other.x + self.y * other.y + self.z * other.z

    def __rmul__(self, other: Union[float, int]) -> "P3":
        """
        当左操作数不支持相应的操作时被调用
        """
        return self.__mul__(other)

    def __truediv__(self, number: Union[int, float]) -> "P3":
        if isinstance(number, int) or isinstance(number, float):
            return self * (1 / number)
        else:
            raise ValueError("P2仅支持数字除法")

    def __matmul__(self, other: "P3") -> "P3":
        """
        矢量叉乘 / 外积，返回新矢量
        """
        return P3(
            self.y * other.z - self.z * other.y,
            -self.x * other.z + self.z * other.x,
            self.x * other.y - self.y * other.x,
        )

    def __str__(self) -> str:
        """
        矢量信息
        """
        return f"[{self.x}, {self.y}, {self.z}]"

    def __repr__(self) -> str:
        """
        同 __str__
        """
        return self.__str__()

    def __eq__(self, other: "P3", err: float = 1e-6, msg: Optional[str] = None) -> bool:
        """
        矢量相等判断
        """
        return (
            BaseUtils.equal(self.x, other.x, err, msg)
            and BaseUtils.equal(self.y, other.y, err, msg)
            and BaseUtils.equal(self.z, other.z, err, msg)
        )

    @staticmethod
    def x_direct(x: float = 1.0) -> "P3":
        """
        创建平行于 x 方向的矢量，或者 x 轴上的点
        """
        return P3(x=x)

    @staticmethod
    def y_direct(y: float = 1.0) -> "P3":
        """
        创建平行于 y 方向的矢量，或者 y 轴上的点
        """
        return P3(y=y)

    @staticmethod
    def z_direct(z: float = 1.0) -> "P3":
        """
        创建平行于 z 方向的矢量，或者 z 轴上的点
        """
        return P3(z=z)

    @staticmethod
    def origin() -> "P3":
        """
        返回坐标原点
        """
        return P3()

    @staticmethod
    def zeros() -> "P3":
        """
        返回零矢量
        """
        return P3()

    def to_list(self) -> List[float]:
        """
        点 (x,y,z) 转为数组 [x,y,z]
        """
        return [self.x, self.y, self.z]

    @staticmethod
    def from_numpy_ndarry(ndarray: numpy.ndarray) -> Union["P3", List["P3"]]:
        """
        将 numpy 数组转为 P3 点或 P3 数组
        根据 numpy 数组形状有不同的返回值
        举例如下
        array([1, 2, 3])  ==》P3 [1.0, 2.0, 3.0]
        array([[1],
              [2],
              [3]])       ==》P3 [1.0, 2.0, 3.0]
        array([[1, 2, 3],
              [4, 5, 6]]) ==》 List[P3] [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        """
        if ndarray.shape == (3,) or ndarray.shape == (3, 1):
            return P3(ndarray[0], ndarray[1], ndarray[2])
        elif len(ndarray.shape) == 2 and ndarray.shape[1] == 3:
            return [P3.from_numpy_ndarry(sub_array) for sub_array in ndarray]
        else:
            raise ValueError(f"无法将{ndarray}转为P3或List[P3]")

    def to_numpy_ndarry3(self, numpy_dtype=numpy.float64) -> numpy.ndarray:
        """
        点 (x,y,z) 转为 numpy 数组 [x,y,z]
        numpy_dtype 指定数据类型
        refactor v0.1.1 新增数据类型
        """
        return numpy.array(self.to_list(), dtype=numpy_dtype)

    def to_p2(
            p, transformation: Callable[["P3"], P2] = lambda p3: P2(p3.x, p3.y)
    ) -> P2:
        """
        根据规则 transformation 将 P3 转为 P2
        默认为抛弃 z 分量
        """
        return transformation(p)

    def populate(self, other: 'P3') -> None:
        """
        将 other 的值赋到 self 中
        since v0.1.1
        """
        self.x = other.x
        self.y = other.y
        self.z = other.z


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
            x_direction * z_direction,
            0.0,
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
        return f"ORIGIN={self.location}, xi={self.XI}, yi={self.YI}, zi={self.ZI}"

    def __repr__(self) -> str:
        """
        同 __str__
        """
        return self.__str__()

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


class ValueWithDistance(Generic[T]):
    """
    辅助对象，带有距离的一个量，通常用于描述线上磁场分布
    """

    def __init__(self, value: T, distance: float) -> None:
        self.value: T = value
        self.distance: float = distance


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
        -------

        """
        raise NotImplementedError

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
        number: int = int(math.ceil(self.get_length() / step))
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
        number: int = int(math.ceil(self.get_length() / step))
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
        return f"Line2，起点{self.point_at_start()}，长度{self.get_length()}"


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
        return f"直线段，起点{self.start_point}，方向{self.direct}，长度{self.length}"

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
        return (
            f"弧线段，起点{self.point_at_start()}，"
            + f"方向{self.direct_at_start()}，顺时针{self.clockwise}，半径{self.radius}，角度{self.total_phi}"
        )

    def __repr__(self) -> str:
        return self.__str__()


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
        details = "\t\n".join(self.__trajectoryList.__str__())
        return f"Trajectory:{details}"

    def __repr__(self) -> str:
        return self.__str__()

    class __TrajectoryBuilder:
        """
        Trajectory 使用 set_start_point 进行构造时的中间产物
        """

        def __init__(self, start_point: P2):
            self.start_point = start_point

        def first_line(self, direct: P2, length: float) -> "Trajectory":
            """
            设置 Trajectory 第一条直线段
            注意：Trajectory 只能以直线开头，不能以圆弧开头
            """
            return Trajectory(StraightLine2(length, direct, self.start_point))

    @staticmethod
    def set_start_point(start_point: P2):
        """
        设置 Trajectory 起点
        """
        return Trajectory.__TrajectoryBuilder(start_point)

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
        distances = BaseUtils.linspace(0, length, int(length / step))
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
        distances = BaseUtils.linspace(0, length, int(length / step))

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
        distances = BaseUtils.linspace(0, length, int(length / step))

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
                2]

            ret.append(P2(s, gradient))
        return ret

    def multipole_field_along(
            self,
            line2: Line2,
            order: int,
            good_field_area_width: float = 10 * MM,
            step: float = 1 * MM,
            point_number: int = 4,
    ) -> List[List[P2]]:
        """
        计算本对象在二维曲线 line2 上的各极磁场分布
        """
        raise NotImplementedError

    def integration_field(
            self,
            line2: Line2,
            p2_t0_p3: Callable[[P2], P3] = lambda p2: P3(p2.x, p2.y, 0.0),
            step: float = 1 * MM,
    ) -> float:
        """
        计算本对象在二维曲线 line2 上的积分场
        """
        raise NotImplementedError

    def slice_to_cosy_script(
            self,
            Bp: float,
            aperture_radius: float,
            line2: Line2,
            good_field_area_width: float,
            min_step_length: float,
            tolerance: float,
    ) -> str:
        """
        将本对象在由二维曲线 line2 切成 COSY 切片
        """
        raise NotImplementedError

    @staticmethod
    def no_magnet():
        """
        返回一个不产生磁场的 Magnet
        实现代码完备性
        """

        class NoMagnet(Magnet):
            def magnetic_field_at(self, point: P3) -> P3:
                return P3.zeros()

        return NoMagnet()


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


class Protons:
    """
    质子相关常量和计算
    """

    # 静止质量
    STATIC_MASS_KG = 1.672621898e-27

    # 静止能量 = m0 * c ^ 2 单位焦耳
    STATIC_ENERGY_J = STATIC_MASS_KG * LIGHT_SPEED * LIGHT_SPEED

    # 静止能量 eV 为单位
    STATIC_ENERGY_eV = STATIC_ENERGY_J / eV

    # 静止能量 MeV 为单位，应该是 STATIC_ENERGY_J / MeV。但是写成字面量
    STATIC_ENERGY_MeV = 938.2720813

    # 电荷量 库伦
    CHARGE_QUANTITY = 1.6021766208e-19

    @classmethod
    def get_total_energy_MeV(cls, kinetic_energy_MeV: float) -> float:
        """
        计算总能量 MeV = 静止能量 + 动能
        Parameters
        ----------
        kinetic_energy_MeV 动能 MeV 一般为 250 Mev

        Returns 总能量 MeV
        -------

        """
        return cls.STATIC_ENERGY_MeV + kinetic_energy_MeV

    @classmethod
    def get_total_energy_J(cls, kinetic_energy_MeV: float) -> float:
        """
        计算总能量 焦耳
        Parameters
        ----------
        kinetic_energy_MeV 动能 MeV 一般为 250 Mev

        Returns 总能量 焦耳
        -------

        """
        return cls.get_total_energy_MeV(kinetic_energy_MeV) * MeV

    @classmethod
    def get_relativistic_mass(cls, kinetic_energy_MeV: float) -> float:
        """
        计算动质量 kg = 动能 / (c^2)
        Parameters
        ----------
        kinetic_energy_MeV 动能 MeV 一般为 250 Mev

        Returns 动质量 kg
        -------

        """
        return cls.get_total_energy_J(kinetic_energy_MeV) / LIGHT_SPEED / LIGHT_SPEED

    @classmethod
    def get_speed_m_per_s(cls, kinetic_energy_MeV: float) -> float:
        """
        计算速度 m/s = c * sqrt( 1 - (m0/m)^2 )
        Parameters
        ----------
        kinetic_energy_MeV 动能 MeV 一般为 250 Mev

        Returns 速度 m/s
        -------

        """
        return LIGHT_SPEED * math.sqrt(
            1
            - (cls.STATIC_MASS_KG / cls.get_relativistic_mass(kinetic_energy_MeV)) ** 2
        )

    @classmethod
    def get_momentum_kg_m_pre_s(cls, kinetic_energy_MeV: float) -> float:
        """
        动量 kg m/s
        Parameters
        ----------
        kinetic_energy_MeV 动能 MeV 一般为 250 Mev

        Returns 动量 kg m/s
        -------

        """
        return cls.get_relativistic_mass(kinetic_energy_MeV) * cls.get_speed_m_per_s(
            kinetic_energy_MeV
        )

    @classmethod
    def getMomentum_MeV_pre_c(cls, kinetic_energy_MeV: float) -> float:
        """
        动量 MeV/c
        Parameters 动能 MeV 一般为 250 Mev
        ----------
        kinetic_energy_MeV

        Returns 动量 MeV/c
        -------

        """
        return cls.get_momentum_kg_m_pre_s(kinetic_energy_MeV) / MeV_PER_C

    @classmethod
    def get_magnetic_stiffness(cls, kinetic_energy_MeV: float) -> float:
        """
        磁钢度 T/m
        Parameters
        ----------
        kinetic_energy_MeV 动能 MeV 一般为 250 Mev

        Returns 磁钢度 T/m
        -------

        """
        return cls.get_momentum_kg_m_pre_s(kinetic_energy_MeV) / cls.CHARGE_QUANTITY

    # ------------------  动量分散相关  ----------------------
    @classmethod
    def get_kinetic_energy_MeV(cls, momentum_KG_M_PER_S: float) -> float:
        # 速度
        speed = momentum_KG_M_PER_S / math.sqrt(
            cls.STATIC_MASS_KG ** 2 + (momentum_KG_M_PER_S / LIGHT_SPEED) ** 2
        )
        # 动质量
        relativistic_mass = cls.STATIC_MASS_KG / math.sqrt(
            1 - (speed / LIGHT_SPEED) ** 2
        )
        # 总能量 J
        total_energy_J = relativistic_mass * LIGHT_SPEED * LIGHT_SPEED
        # 动能 J
        k = total_energy_J - cls.STATIC_ENERGY_J

        return k / MeV

    # @classmethod
    # def get动量分散后的动能(cls, 原动能_MeV: float, 动量分散: float):
    #     """
    #     英文版见下
    #     Parameters
    #     ----------
    #     原动能_MeV
    #     动量分散
    #
    #     Returns 动量分散后的动能 MeV
    #     -------
    #
    #     """
    #     原动量 = cls.get_momentum_kg_m_pre_s(原动能_MeV)
    #
    #     新动量 = 原动量 * (1 + 动量分散)
    #
    #     新动能 = cls.get_kinetic_energy_MeV(新动量)
    #
    #     return 新动能

    @classmethod
    def get_kinetic_energy_MeV_after_momentum_dispersion(
            cls, old_kinetic_energy_MeV: float, momentum_dispersion: float
    ) -> float:
        """
        中文版见上
        Parameters
        ----------
        old_kinetic_energy_MeV 原动能_MeV
        momentum_dispersion 动量分散

        Returns 动量分散后的动能 MeV
        -------

        """
        momentum0 = cls.get_momentum_kg_m_pre_s(old_kinetic_energy_MeV)

        momentum = momentum0 * (1 + momentum_dispersion)

        kinetic_energy = cls.get_kinetic_energy_MeV(momentum)

        return kinetic_energy

    # @classmethod
    # def convert动量分散_TO_能量分散(cls, 动量分散: float, 动能_MeV: float) -> float:
    #     """
    #     下方法的中文版
    #     Parameters
    #     ----------
    #     动量分散
    #     动能_MeV
    #
    #     Returns convert动量分散_TO_能量分散
    #     -------
    #
    #     """
    #     k = (动能_MeV + cls.STATIC_ENERGY_MeV) / \
    #         (动能_MeV + 2 * cls.STATIC_ENERGY_MeV)
    #
    #     return 动量分散 / k

    @classmethod
    def convert_momentum_dispersion_to_energy_dispersion(
            cls, momentum_dispersion: float, kinetic_energy_MeV: float
    ) -> float:
        """
        上方法的英文版
        Parameters
        ----------
        momentum_dispersion 动量分散
        kinetic_energy_MeV 动能_MeV

        Returns convert动量分散_TO_能量分散
        -------

        """
        k = (kinetic_energy_MeV + cls.STATIC_ENERGY_MeV) / (
            kinetic_energy_MeV + 2 * cls.STATIC_ENERGY_MeV
        )

        return momentum_dispersion / k

    # @classmethod
    # def convert能量分散_TO_动量分散(cls, 能量分散: float, 动能_MeV: float) -> float:
    #     k = (动能_MeV + cls.STATIC_ENERGY_MeV) / \
    #         (动能_MeV + 2 * cls.STATIC_ENERGY_MeV)
    #     return 能量分散 * k

    @classmethod
    def convert_energy_dispersion_to_momentum_dispersion(
            cls, energyDispersion: float, kineticEnergy_MeV: float
    ) -> float:
        """
        上方法的英文版
        Parameters
        ----------
        energyDispersion 能量分散
        kineticEnergy_MeV 动能，典型值 250

        Returns 动量分散
        -------

        """
        k = (kineticEnergy_MeV + cls.STATIC_ENERGY_MeV) / (
            kineticEnergy_MeV + 2 * cls.STATIC_ENERGY_MeV
        )
        return energyDispersion * k


class RunningParticle:
    """
    在全局坐标系中运动的一个粒子
    position 位置，三维矢量，单位 [m, m, m]
    velocity 速度，三位矢量，单位 [m/s, m/s, m/s]
    relativistic_mass 相对论质量，又称为动质量，单位 kg， M=Mo/√(1-v^2/c^2)
    e 电荷量，单位 C 库伦
    speed 速率，单位 m/s
    distance 运动距离，单位 m
    """

    def __init__(
            self,
            position: P3,
            velocity: P3,
            relativistic_mass: float,
            e: float,
            speed: float,
            distance: float = 0.0,
    ):
        """
        在全局坐标系中运动的一个粒子
        Parameters
        ----------
        position 位置，三维矢量，单位 [m, m, m]
        velocity 速度，三位矢量，单位 [m/s, m/s, m/s]
        relativistic_mass 相对论质量，又称为动质量，单位 kg， M=Mo/√(1-v^2/c^2)
        e 电荷量，单位 C 库伦
        speed 速率，单位 m/s
        distance 运动距离，单位 m
        """
        self.position = position
        self.velocity = velocity
        self.relativistic_mass = relativistic_mass
        self.e = e
        self.speed = speed
        self.distance = distance

    def run_self_in_magnetic_field(
            self, magnetic_field: P3, footstep: float = 1 * MM
    ) -> None:
        """
        粒子在磁场 magnetic_field 中运动 footstep 长度
        Parameters
        ----------
        magnetic_field 磁场，看作恒定场
        footstep 步长，默认 1 MM

        Returns None
        -------
        """
        warnings.warn(
            "run_self_in_magnetic_field 函数已经废弃，因为没有使用 Runge-Kutta 数值积分方法，误差过大", DeprecationWarning)
        # 计算受力 qvb
        f = (self.velocity @ magnetic_field) * self.e
        # 计算加速度 a = f/m
        a = f / self.relativistic_mass
        # 计算运动时间
        t = footstep / self.speed
        # 位置变化
        self.position += self.velocity * t
        # 速度变化
        self.velocity += t * a
        # 运动距离
        self.distance += footstep

    def copy(self) -> "RunningParticle":
        """
        深拷贝粒子
        Returns 深拷贝粒子
        -------

        """
        return RunningParticle(
            self.position.copy(),
            self.velocity.copy(),
            self.relativistic_mass,
            self.e,
            self.speed,
            self.distance,
        )

    def compute_scalar_momentum(self) -> float:
        """
        获得标量动量
        Returns 标量动量
        -------

        """
        return self.speed * self.relativistic_mass

    def change_scalar_momentum(self, scalar_momentum: float) -> None:
        """
        改变粒子的标量动量。
        注意：真正改变的是粒子的速度和动质量
        这个方法用于生成一组动量分散的粒子

        scalar_momentum 标量动量
        Returns None
        -------

        """
        # 先求 静止质量
        m0 = self.relativistic_mass * math.sqrt(
            1 - (self.speed ** 2) / (LIGHT_SPEED ** 2)
        )
        # 求新的速率
        new_speed = scalar_momentum / math.sqrt(
            m0 ** 2 + (scalar_momentum / LIGHT_SPEED) ** 2
        )
        # 求新的动质量
        new_relativistic_mass = m0 / \
            math.sqrt(1 - (new_speed / LIGHT_SPEED) ** 2)
        # 求新的速度
        new_velocity: P3 = self.velocity.change_length(new_speed)

        # 写入
        self.relativistic_mass = new_relativistic_mass
        self.speed = new_speed
        self.velocity = new_velocity

        # 验证
        BaseUtils.equal(
            scalar_momentum,
            self.compute_scalar_momentum(),
            msg=f"RunningParticle::change_scalar_momentum异常，scalar_momentum{scalar_momentum}!=self.compute_scalar_momentum{self.compute_scalar_momentum}",
            err=1e-6,
        )

        BaseUtils.equal(
            self.speed,
            self.velocity.length(),
            msg=f"RunningParticle::change_scalar_momentum异常,self.speed{self.speed}!=Vectors.length(self.velocity){self.velocity.length()}",
        )

    def get_natural_coordinate_system(
            self, y_direction: P3 = P3.z_direct()
    ) -> LocalCoordinateSystem:
        return LocalCoordinateSystem.create_by_y_and_z_direction(
            self.position, y_direction, self.velocity
        )

    def __str__(self) -> str:
        return f"p={self.position},v={self.velocity},v0={self.speed}"

    def __repr__(self) -> str:
        return self.__str__()

    def to_numpy_array_data(self, numpy_dtype=numpy.float64) -> numpy.ndarray:
        """
        RunningParticle 转为 numpy_array_data
        主要用于 GPU 加速
        numpy_array_data 是一个一维数组，分别是
        (px0, py1, pz2, vx3, vy4, vz5, rm6, e7, speed8, distance9) len = 10

        since v0.1.1
        """
        data_list: List[float] = (
            self.position.to_list() +
            self.velocity.to_list() +
            [self.relativistic_mass, self.e, self.speed, self.distance]
        )
        return numpy.array(data_list, dtype=numpy_dtype)

    @staticmethod
    def from_numpy_array_data(numpy_array) -> 'RunningParticle':
        """
        上函数的逆函数
        see to_numpy_array_data
        since v0.1.1
        """
        pos = P3(numpy_array[0], numpy_array[1], numpy_array[2])
        vel = P3(numpy_array[3], numpy_array[4], numpy_array[5])

        return RunningParticle(
            position=pos,
            velocity=vel,
            relativistic_mass=numpy_array[6],
            e=numpy_array[7],
            speed=numpy_array[8],
            distance=numpy_array[9]
        )

    def populate(self, other: 'RunningParticle') -> None:
        """
        将 other 的值赋到 self 中
        since v0.1.1
        """
        self.position.populate(other.position)
        self.velocity.populate(other.velocity)
        self.relativistic_mass = other.relativistic_mass
        self.e = other.e
        self.speed = other.speed
        self.distance = other.distance

    def __sub__(self, other: 'RunningParticle') -> "RunningParticle":
        """
        粒子"减法" 只用来显示两个粒子的差异
        一般用于 debug
        since v0.1.1
        """
        return RunningParticle(
            position=self.position - other.position,
            velocity=self.velocity - other.velocity,
            relativistic_mass=self.relativistic_mass - other.relativistic_mass,
            e=self.e - other.e,
            speed=self.speed - other.speed,
            distance=self.distance - other.distance,
        )


class ParticleRunner:
    """
    粒子运动工具类
    """

    @staticmethod
    def __callback_for_runge_kutta4(particle: RunningParticle, magnet: Magnet) -> Callable[
            [float, numpy.ndarray], numpy.ndarray]:
        """
        [p,v] --> [v,a] 将二阶微分方程转为一阶
        see BaseUtils.runge_kutta4()
        since v0.1.1
        """
        k: float = particle.e / particle.relativistic_mass

        def callback(t: float, Y: numpy.ndarray) -> numpy.ndarray:
            # 想不到啊，一直厌恶闭包的我，居然自然而然地写了出来
            # nonlocal k, magnet
            position: P3 = Y[0]
            velocity: P3 = Y[1]

            accelerate: P3 = k * \
                (velocity @ magnet.magnetic_field_at(position))

            return numpy.array([velocity, accelerate])

        return callback

    @staticmethod
    def __callback_for_solve_ode(particle: RunningParticle, magnet: Magnet) -> Callable[
            [float, numpy.ndarray], numpy.ndarray]:
        """
        see BaseUtils.solve_ode()
        since v0.1.1
        """
        k: float = particle.e / particle.relativistic_mass

        def callback(t: float, Y: numpy.ndarray) -> numpy.ndarray:
            # 想不到啊，一直厌恶闭包的我，居然自然而然地写了出来
            # nonlocal k, magnet
            position: P3 = P3(Y[0], Y[1], Y[2])
            velocity: P3 = P3(Y[3], Y[4], Y[5])

            accelerate: P3 = k * \
                (velocity @ magnet.magnetic_field_at(position))

            return numpy.array([velocity.x, velocity.y, velocity.z, accelerate.x, accelerate.y, accelerate.z])

        return callback

    @staticmethod
    def run_only(
            p: Union[RunningParticle, List[RunningParticle]], m: Magnet, length: float, footstep: float = 20 * MM,
            concurrency_level: int = 1
    ) -> Union[RunningParticle, List[RunningParticle]]:
        """
        让粒子 p 在磁场 m 中运动 length 距离，步长 footstep
        Parameters
        ----------
        p 粒子
        m 磁场
        length 运动长度
        footstep 步长

        Returns None
        -------
        refactor v0.1.1 使用 runge kutta 和 加入多进程支持
        """
        if isinstance(p, RunningParticle):
            dt = footstep / p.speed
            t_end = length / p.speed
            Y0 = numpy.array([p.position, p.velocity])
            func = ParticleRunner.__callback_for_runge_kutta4(
                particle=p, magnet=m)
            Y1 = BaseUtils.runge_kutta4(
                t0=0.0, t_end=t_end, Y0=Y0, y_derived_function=func, dt=dt, record=False)
            p.position = Y1[0]
            p.velocity = Y1[1]
            p.distance += length
            return p
        elif concurrency_level == 1:
            particle_number = len(p)
            print(f"track {particle_number} particles")
            print("当前使用单线程进行粒子跟踪，如果函数支持多线程并行，推荐使用多线程")
            particle_index = 0
            start_time = time.time()
            for this_p in p:
                ParticleRunner.run_only(this_p, m, length, footstep)
                particle_index += 1
                if particle_index == 1:
                    time_run_one_particle = time.time() - start_time
                    print(
                        f"运行一个粒子需要{time_run_one_particle:.5f}秒，估计总耗时{time_run_one_particle * particle_number:.5f}秒")
                print(
                    '\b'*8 + f'{(particle_index / particle_number * 100):>6.2f}% ', end='', flush=True)
            print(' finished')
            print(f"实际用时{(time.time()-start_time):.5f}秒")
            return p
        else:
            results: List[RunningParticle] = BaseUtils.submit_process_task(
                task=ParticleRunner.run_only,
                param_list=[
                    [this_p, m, length, footstep] for this_p in p
                ],
                concurrency_level=concurrency_level
            )
            particle_number = len(p)
            for i in range(particle_number):
                p[i].position = results[i].position
                p[i].velocity = results[i].velocity
                p[i].distance = results[i].distance

            return p

    @staticmethod
    def run_only_ode(
            p: Union[RunningParticle, List[RunningParticle]], m: Magnet, length: float, footstep: float = 1 * MM,
            absolute_tolerance: float = 1e-8, relative_tolerance: float = 1e-8
    ) -> None:
        """
        让粒子 p 在磁场 m 中运动 length 距离，步长 footstep
        使用 scipy 提供的 ode 法
        Parameters
        ----------
        p 粒子
        m 磁场
        length 运动长度
        footstep 步长

        Returns None
        -------
        refactor v0.1.1 ode45
        """
        if isinstance(p, RunningParticle):
            dt = footstep / p.speed
            t_end = length / p.speed
            Y0 = numpy.array([p.position.x, p.position.y, p.position.z,
                              p.velocity.x, p.velocity.y, p.velocity.z])
            func = ParticleRunner.__callback_for_solve_ode(
                particle=p, magnet=m)
            Y1 = BaseUtils.solve_ode(
                t0=0.0, t_end=t_end, Y0=Y0, y_derived_function=func, dt=dt, record=False,
                absolute_tolerance=absolute_tolerance, relative_tolerance=relative_tolerance)
            p.position = P3(Y1[0][-1], Y1[1][-1], Y1[2][-1])
            p.velocity = P3(Y1[3][-1], Y1[4][-1], Y1[5][-1])
            p.distance += length
            return None
        else:
            particle_number = len(p)
            print(f"track {particle_number} particles")
            for this_p in p:
                print('▇', end='', flush=True)
                ParticleRunner.run_only(this_p, m, length, footstep)
            print(' finished')

    @staticmethod
    def run_get_trajectory(
            p: RunningParticle, m: Magnet, length: float, footstep: float = 1 * MM
    ) -> List[P3]:
        """
        让粒子 p 在磁场 m 中运动 length 距离，步长 footstep
        获得粒子的轨迹
        Parameters
        ----------
        p 粒子
        m 磁场
        length 运动长度
        footstep 步长

        Returns 轨迹 np.ndarray，是三维点的数组
        -------
        refactor v0.1.1 runge kutta
        """
        dt = footstep / p.speed
        t_end = length / p.speed
        Y0 = numpy.array([p.position, p.velocity])
        func = ParticleRunner.__callback_for_runge_kutta4(
            particle=p, magnet=m)
        _, Ys = BaseUtils.runge_kutta4(
            t0=0.0, t_end=t_end, Y0=Y0, y_derived_function=func, dt=dt, record=True)
        p.distance += length

        trajectory: List[P3] = []
        for y in Ys:
            trajectory.append(y[0])

        return trajectory

    @staticmethod
    def run_get_all_info(
            p: RunningParticle, m: Magnet, length: float, footstep: float = 1 * MM
    ) -> List[RunningParticle]:
        """
        让粒子 p 在磁场 m 中运动 length 距离，步长 footstep
        获得粒子全部信息
        Parameters
        ----------
        p 粒子
        m 磁场
        length 运动长度
        footstep 步长

        Returns 每一步处的粒子全部信息 List[RunningParticle]
        -------
        refactor v0.1.1 runge kutta
        """
        distance0 = p.distance

        dt = footstep / p.speed
        t_end = length / p.speed
        Y0 = numpy.array([p.position, p.velocity])
        func = ParticleRunner.__callback_for_runge_kutta4(
            particle=p, magnet=m)
        ts, Ys = BaseUtils.runge_kutta4(
            t0=0.0, t_end=t_end, Y0=Y0, y_derived_function=func, dt=dt, record=True)
        p.distance += length

        all_info: List[RunningParticle] = []

        for i in range(len(ts)):
            t: float = ts[i]
            pos: P3 = Ys[i][0]
            vel: P3 = Ys[i][1]

            this_p = p.copy()
            this_p.position = pos
            this_p.velocity = vel
            this_p.distance = distance0 + t * this_p.speed

            all_info.append(this_p)

        return all_info

    @staticmethod
    def run_only_deprecated(
            p: RunningParticle, m: Magnet, length: float, footstep: float = 1 * MM
    ) -> None:
        """
        让粒子 p 在磁场 m 中运动 length 距离，步长 footstep
        Parameters
        ----------
        p 粒子
        m 磁场
        length 运动长度
        footstep 步长

        Returns None
        -------
        refactor v0.1.1 保存过时方法
        """
        warnings.warn(
            "run_only_deprecated 已过时，因为没有使用 Runge-Kutta 数值积分方法，误差过大", category=DeprecationWarning)
        distance = 0.0
        while distance < length:
            p.run_self_in_magnetic_field(
                m.magnetic_field_at(p.position), footstep=footstep
            )
            distance += footstep

    @staticmethod
    def run_get_trajectory_deprecated(
            p: RunningParticle, m: Magnet, length: float, footstep: float = 1 * MM
    ) -> List[P3]:
        """
        让粒子 p 在磁场 m 中运动 length 距离，步长 footstep
        获得粒子的轨迹
        Parameters
        ----------
        p 粒子
        m 磁场
        length 运动长度
        footstep 步长

        Returns 轨迹 np.ndarray，是三维点的数组
        -------
        refactor v0.1.1 保存过时方法
        """
        warnings.warn(
            "run_get_trajectory_deprecated 已过时，因为没有使用 Runge-Kutta 数值积分方法，误差过大", category=DeprecationWarning)
        trajectory: List[P3] = [p.position.copy()]

        i = 1
        distance = 0.0
        while distance < length:
            p.run_self_in_magnetic_field(
                m.magnetic_field_at(p.position), footstep=footstep
            )
            distance += footstep
            trajectory.append(p.position.copy())
            i += 1

        return trajectory

    @staticmethod
    def run_get_all_info_deprecated(
            p: RunningParticle, m: Magnet, length: float, footstep: float = 1 * MM
    ) -> List[RunningParticle]:
        """
        让粒子 p 在磁场 m 中运动 length 距离，步长 footstep
        获得粒子全部信息
        Parameters
        ----------
        p 粒子
        m 磁场
        length 运动长度
        footstep 步长

        Returns 每一步处的粒子全部信息 List[RunningParticle]
        -------
        refactor v0.1.1 保存过时方法
        """
        warnings.warn(
            "run_get_all_info_deprecated 已过时，因为没有使用 Runge-Kutta 数值积分方法，误差过大", category=DeprecationWarning)

        all_info: List[RunningParticle] = [p.copy()]
        distance = 0.0
        while distance < length:
            p.run_self_in_magnetic_field(
                m.magnetic_field_at(p.position), footstep=footstep
            )
            distance += footstep
            all_info.append(p.copy())

        return all_info


class PhaseSpaceParticle:
    XXP_PLANE = 1
    YYP_PLANE = 2

    """
    相空间中的粒子，6个坐标 x xp y yp z delta
    """

    def __init__(
            self, x: float, xp: float, y: float, yp: float, z: float, delta: float
    ):
        self.x = x
        self.xp = xp
        self.y = y
        self.yp = yp
        self.z = z
        self.delta = delta

    def project_to_xxp_plane(self) -> P2:
        """
        投影到 x-xp 平面
        Returns [self.x, self.xp]
        -------

        """
        return P2(self.x, self.xp)

    def project_to_yyp_plane(self) -> P2:
        """
        投影到 y-yp 平面
        Returns [self.y, self.yp]
        -------

        """
        return P2(self.y, self.yp)

    def project_to_plane(self, plane_id: int) -> P2:
        if plane_id == PhaseSpaceParticle.XXP_PLANE:
            return self.project_to_xxp_plane()
        elif plane_id == PhaseSpaceParticle.YYP_PLANE:
            return self.project_to_yyp_plane()
        else:
            raise ValueError(f"没有处理plane_id({plane_id})的方法")

    @staticmethod
    def phase_space_particles_along_positive_ellipse_in_xxp_plane(
            xMax: float, xpMax: float, delta: float, number: int
    ) -> List["PhaseSpaceParticle"]:
        """
        获取分布于 x xp 平面上 正相椭圆上的 PhaseSpaceParticles
        注意是 正相椭圆
        Parameters
        ----------
        xMax 相椭圆参数 x 最大值
        xpMax 相椭圆参数 xp 最大值
        delta 动量分散
        number 粒子数目

        Returns 分布于 x xp 平面上 正相椭圆上的 PhaseSpaceParticles
        -------

        """
        A: float = 1 / (xMax ** 2)
        B: float = 0
        C: float = 1 / (xpMax ** 2)
        D: float = 1

        return [
            PhaseSpaceParticle(p.x, p.y, 0, 0, 0, delta)
            for p in BaseUtils.Ellipse(
                A, B, C, D
            ).uniform_distribution_points_along_edge(number)
        ]

    @staticmethod
    def phase_space_particles_along_positive_ellipse_in_yyp_plane(
            yMax: float, ypMax: float, delta: float, number: int
    ) -> List["PhaseSpaceParticle"]:
        """
        获取分布于 y yp 平面上 正相椭圆上的 PhaseSpaceParticles
        注意是 正相椭圆
        Parameters
        ----------
        yMax 相椭圆参数 y 最大值
        ypMax 相椭圆参数 yp 最大值
        delta 动量分散
        number 粒子数目

        Returns 分布于 y yp 平面上 正相椭圆上的 PhaseSpaceParticles
        -------

        """
        A: float = 1 / (yMax ** 2)
        B: float = 0
        C: float = 1 / (ypMax ** 2)
        D: float = 1

        return [
            PhaseSpaceParticle(0, 0, p.x, p.y, 0, delta)
            for p in BaseUtils.Ellipse(
                A, B, C, D
            ).uniform_distribution_points_along_edge(number)
        ]

    @staticmethod
    def phase_space_particles_along_positive_ellipse_in_plane(
            plane_id: int, xMax: float, xpMax: float, delta: float, number: int
    ) -> List["PhaseSpaceParticle"]:
        """
        获取分布于 x xp 平面上或 y yp 平面上的，正相椭圆上的 PhaseSpaceParticles
        Parameters
        ----------
        xxPlane x 平面或 y 平面，true：x 平面，false:y 平面
        xMax 相椭圆参数 x/y 最大值
        xpMax 相椭圆参数 xp/yp 最大值
        delta 动量分散
        number 粒子数目

        Returns 分布于 x xp 平面上或 y yp 平面上的，正相椭圆上的 PhaseSpaceParticles
        -------

        """
        if plane_id == PhaseSpaceParticle.XXP_PLANE:
            return PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_xxp_plane(
                xMax, xpMax, delta, number
            )
        elif plane_id == PhaseSpaceParticle.YYP_PLANE:
            return PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_yyp_plane(
                xMax, xpMax, delta, number
            )
        else:
            raise ValueError(f"没有处理plane_id({plane_id})的方法")

    @staticmethod
    def phase_space_particles_project_to_xxp_plane(
            phase_space_particles: List,
    ) -> List[P2]:
        """
        相空间粒子群投影到 x 平面
        Parameters
        ----------
        phase_space_particles 相空间粒子群

        Returns 相空间粒子群投影到 x 平面 [[x1,xp1], [x2,xp2] .. ]
        -------

        """
        return [p.project_to_xxp_plane() for p in phase_space_particles]

    @staticmethod
    def phase_space_particles_project_to_yyp_plane(
            phase_space_particles: List,
    ) -> List[P2]:
        """
        相空间粒子群投影到 y 平面
        Parameters
        ----------
        phase_space_particles 相空间粒子群

        Returns 相空间粒子群投影到 y 平面 [[y1,yp1], [y2,yp2] .. ]
        -------

        """
        return [p.project_to_yyp_plane() for p in phase_space_particles]

    @staticmethod
    def phase_space_particles_project_to_plane(
            phase_space_particles: List, plane_id: int
    ) -> List[P2]:
        """
        相空间粒子群投影到 x/y 平面
        Parameters
        ----------
        phase_space_particles 相空间粒子群
        plane_id 投影到 x 或 y 平面

        Returns 相空间粒子群投影到 x/y 平面
        -------

        """
        if plane_id == PhaseSpaceParticle.XXP_PLANE:
            return PhaseSpaceParticle.phase_space_particles_project_to_xxp_plane(
                phase_space_particles
            )
        elif plane_id == PhaseSpaceParticle.YYP_PLANE:
            return PhaseSpaceParticle.phase_space_particles_project_to_yyp_plane(
                phase_space_particles
            )
        else:
            raise ValueError(f"没有处理plane_id({plane_id})的方法")

    @staticmethod
    def create_from_running_particle(
            ideal_particle: RunningParticle,
            coordinate_system: LocalCoordinateSystem,
            running_particle: RunningParticle,
    ) -> "PhaseSpaceParticle":
        """
        将实际粒子 running_particle 映射为 PhaseSpaceParticle
        这需要一个理想粒子/参考粒子 ideal_particle
        和一个参考粒子的自然坐标系 coordinate_system
        """
        # x y z
        relative_position = running_particle.position - ideal_particle.position
        x = coordinate_system.XI * relative_position
        y = coordinate_system.YI * relative_position
        z = coordinate_system.ZI * relative_position

        # xp yp
        relative_velocity = running_particle.velocity - ideal_particle.velocity
        xp = (coordinate_system.XI * relative_velocity) / ideal_particle.speed
        yp = (coordinate_system.YI * relative_velocity) / ideal_particle.speed

        # delta
        rm = running_particle.compute_scalar_momentum()
        im = ideal_particle.compute_scalar_momentum()
        delta = (rm - im) / im

        return PhaseSpaceParticle(x, xp, y, yp, z, delta)

    @staticmethod
    def create_from_running_particles(
            ideal_particle: RunningParticle,
            coordinate_system: LocalCoordinateSystem,
            running_particles: List[RunningParticle],
    ) -> List["PhaseSpaceParticle"]:
        """
        将多个实际粒子 running_particles 映射为 PhaseSpaceParticles
        参数意义见上函数 create_from_running_particle
        """
        return [
            PhaseSpaceParticle.create_from_running_particle(
                ideal_particle, coordinate_system, rp
            )
            for rp in running_particles
        ]

    @staticmethod
    def convert_delta_from_momentum_dispersion_to_energy_dispersion(
            phaseSpaceParticle, centerKineticEnergy_MeV
    ) -> "PhaseSpaceParticle":
        """
        动量分散改动能分散
        Parameters
        ----------
        phaseSpaceParticle 原粒子
        centerKineticEnergy_MeV 中心动能，如 250

        Returns 动量分散改动能分散后的粒子
        -------

        """
        copied: PhaseSpaceParticle = phaseSpaceParticle.copy()
        deltaMomentumDispersion = copied.delta
        deltaEnergyDispersion = (
            Protons.convert_momentum_dispersion_to_energy_dispersion(
                deltaMomentumDispersion, centerKineticEnergy_MeV
            )
        )

        copied.delta = deltaEnergyDispersion

        return copied

    @staticmethod
    def convert_delta_from_momentum_dispersion_to_energy_dispersion_for_list(
            phaseSpaceParticles: List, centerKineticEnergy_MeV
    ) -> List["PhaseSpaceParticle"]:
        """
        动量分散改动能分散，见上方法 convert_delta_from_momentum_dispersion_to_energy_dispersion
        Parameters
        ----------
        phaseSpaceParticles
        centerKineticEnergy_MeV

        Returns
        -------

        """
        return [
            PhaseSpaceParticle.convert_delta_from_momentum_dispersion_to_energy_dispersion(
                pp, centerKineticEnergy_MeV
            )
            for pp in phaseSpaceParticles
        ]

    @staticmethod
    def convert_delta_from_energy_dispersion_to_momentum_dispersion(
            phaseSpaceParticle, centerKineticEnergy_MeV: float
    ) -> "PhaseSpaceParticle":
        """
        将相空间粒子 phaseSpaceParticle 中 delta 从能量分散转为动量分散
        centerKineticEnergy_MeV 中心动能
        """
        copied = phaseSpaceParticle.copy()

        EnergyDispersion = copied.getDelta()

        MomentumDispersion = Protons.convert_energy_dispersion_to_momentum_dispersion(
            EnergyDispersion, centerKineticEnergy_MeV
        )

        copied.delta = MomentumDispersion

        return copied

    @staticmethod
    def convert_delta_from_energy_dispersion_to_momentum_dispersion_for_list(
            phaseSpaceParticles: List, centerKineticEnergy_MeV: float
    ) -> List["PhaseSpaceParticle"]:
        """
        将多个相空间粒子 phaseSpaceParticles 中 delta 从能量分散转为动量分散
        centerKineticEnergy_MeV 中心动能
        """
        return [
            PhaseSpaceParticle.convert_delta_from_energy_dispersion_to_momentum_dispersion(
                pp, centerKineticEnergy_MeV
            )
            for pp in phaseSpaceParticles
        ]

    def __str__(self) -> str:
        return (
            f"x={self.x},xp={self.xp},y={self.y},yp={self.yp},z={self.z},d={self.delta}"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self) -> "PhaseSpaceParticle":
        """
        PhaseSpaceParticle 深拷贝
        """
        return PhaseSpaceParticle(self.x, self.xp, self.y, self.yp, self.z, self.delta)


class ParticleFactory:
    """
    质子工厂
    """

    @staticmethod
    def create_proton(
            position: P3, direct: P3, kinetic_MeV: float = 250
    ) -> RunningParticle:
        """
        生成一个质子（即 RunningParticle 对象），
        位置为 position，
        运动方向为 direct，
        动能为 kinetic_MeV
        """
        # 速率
        speed = LIGHT_SPEED * math.sqrt(
            1.0
            - (Protons.STATIC_ENERGY_MeV /
               (Protons.STATIC_ENERGY_MeV + kinetic_MeV))
            ** 2
        )

        # mass kg
        relativistic_mass = Protons.STATIC_MASS_KG / math.sqrt(
            1.0 - (speed ** 2) / (LIGHT_SPEED ** 2)
        )

        return RunningParticle(
            position,
            direct.copy().change_length(speed),
            relativistic_mass,
            Protons.CHARGE_QUANTITY,
            speed,
        )

    @staticmethod
    def create_proton_by_position_and_velocity(
            position: P3, velocity: P3
    ) -> RunningParticle:
        """
        生成一个质子，
        位置为 position，
        速度为 velocity
        """
        speed = velocity.length()

        relativistic_mass = 0.0

        try:
            relativistic_mass = Protons.STATIC_MASS_KG / math.sqrt(
                1.0 - (speed ** 2) / (LIGHT_SPEED ** 2)
            )
        except RuntimeWarning as e:
            print(
                f"ParticleFactory::create_proton_by_position_and_velocity 莫名其妙的异常 speed={speed} LIGHT_SPEED={LIGHT_SPEED} e={e}"
            )

        return RunningParticle(
            position, velocity, relativistic_mass, Protons.CHARGE_QUANTITY, speed
        )

    @staticmethod
    def create_proton_along(
            trajectory: Line2, s: float = 0.0, kinetic_MeV: float = 250
    ) -> RunningParticle:
        """
        生成一个沿着设计轨道 trajectory 的质子，
        位于轨道 s 位置，
        动能为 kinetic_MeV。
        这个函数一般用于生成参考粒子
        """
        return ParticleFactory.create_proton(
            trajectory.point_at(s).to_p3(),
            trajectory.direct_at(s).to_p3(),
            kinetic_MeV=kinetic_MeV,
        )

    @staticmethod
    def create_from_phase_space_particle(
            ideal_particle: RunningParticle,
            coordinate_system: LocalCoordinateSystem,
            phase_space_particle: PhaseSpaceParticle,
    ) -> RunningParticle:
        """
        将相空间粒子 phase_space_particle 映射为实际粒子（质子）
        Parameters
        ----------
        ideal_particle 理想粒子
        coordinate_system 相空间坐标系
        phase_space_particle 相空间粒子

        Returns 通过理想粒子，相空间坐标系 和 相空间粒子，来创造粒子
        -------

        """
        x = phase_space_particle.x
        xp = phase_space_particle.xp
        y = phase_space_particle.y
        yp = phase_space_particle.yp
        z = phase_space_particle.z
        delta = phase_space_particle.delta

        p = ideal_particle.copy()
        # 知道 LocalCoordinateSystem 的用处了吧
        p.position += coordinate_system.XI * x
        p.position += coordinate_system.YI * y
        p.position += coordinate_system.ZI * z

        if delta != 0.0:
            scalar_momentum = p.compute_scalar_momentum() * (1.0 + delta)
            p.change_scalar_momentum(scalar_momentum)  # 这个方法就是为了修改动量而写的

        p.velocity += coordinate_system.XI * (xp * p.speed)
        p.velocity += coordinate_system.YI * (yp * p.speed)

        return p

    @staticmethod
    def create_from_phase_space_particles(
            ideal_particle: RunningParticle,
            coordinate_system: LocalCoordinateSystem,
            phase_space_particles: List[PhaseSpaceParticle],
    ) -> List[RunningParticle]:
        """
        将多个相空间粒子 phase_space_particle 映射为实际粒子（质子）
        详见上函数 create_from_phase_space_particle
        """
        return [
            ParticleFactory.create_from_phase_space_particle(
                ideal_particle, coordinate_system, p
            )
            for p in phase_space_particles
        ]


class CCT(Magnet, ApertureObject):
    """
    表示一层弯曲 CCT 线圈
    """

    def __init__(
            self,
            # CCT 局部坐标系
            local_coordinate_system: LocalCoordinateSystem,
            # 大半径：偏转半径
            big_r: float,
            # 小半径（孔径/2）
            small_r: float,
            # 偏转角度，即 phi0*winding_number，典型值 67.5
            bending_angle: float,
            # 各极倾斜角，典型值 [30,90,90,90]
            tilt_angles: List[float],
            # 匝数
            winding_number: int,
            # 电流
            current: float,
            # CCT 路径在二维 ξ-φ 坐标系中的起点
            starting_point_in_ksi_phi_coordinate: P2,
            # CCT 路径在二维 ξ-φ 坐标系中的终点
            end_point_in_ksi_phi_coordinate: P2,
            # 每匝线圈离散电流元数目，数字越大计算精度越高
            disperse_number_per_winding: int = 120,
    ):
        self.local_coordinate_system = local_coordinate_system
        self.big_r = float(big_r)
        self.small_r = float(small_r)
        self.bending_angle = float(bending_angle)
        self.tilt_angles = [float(e) for e in tilt_angles]
        self.winding_number = int(winding_number)
        self.current = float(current)
        self.starting_point_in_ksi_phi_coordinate = starting_point_in_ksi_phi_coordinate
        self.end_point_in_ksi_phi_coordinate = end_point_in_ksi_phi_coordinate
        self.disperse_number_per_winding = int(disperse_number_per_winding)

        # 弯转角度，弧度制
        self.bending_radian = BaseUtils.angle_to_radian(self.bending_angle)

        # 倾斜角，弧度制
        self.tilt_radians = BaseUtils.angle_to_radian(self.tilt_angles)

        # 每绕制一匝，φ 方向前进长度
        self.phi0 = self.bending_radian / self.winding_number

        # 极点 a
        self.a = math.sqrt(self.big_r ** 2 - self.small_r ** 2)

        # 双极坐标系另一个常量 η
        self.eta = 0.5 * \
            math.log((self.big_r + self.a) / (self.big_r - self.a))

        # 建立 ξ-φ 坐标到三维 xyz 坐标的转换器
        self.bipolar_toroidal_coordinate_system = CCT.BipolarToroidalCoordinateSystem(
            self.a, self.eta, self.big_r, self.small_r
        )

        # CCT 路径的在 ξ-φ 坐标的表示 函数 φ(ξ)
        def phi_ksi_function(ksi): return self.__phi_ksi_function(ksi)

        # CCT 路径的在 ξ-φ 坐标的表示 函数 P(ξ)=(ξ,φ(ξ))
        def p2_function(ksi): return P2(ksi, phi_ksi_function(ksi))

        # CCT 路径的在 xyz 坐标的表示 函数 P(ξ)=P(x(ξ),y(ξ),z(ξ))
        def p3_function(ksi): return self.bipolar_toroidal_coordinate_system.convert(
            p2_function(ksi)
        )

        # 总匝数
        self.total_disperse_number = self.winding_number * self.disperse_number_per_winding

        dispersed_path2: List[List[float]] = [
            p2_function(ksi).to_list()
            for ksi in BaseUtils.linspace(
                self.starting_point_in_ksi_phi_coordinate.x,
                self.end_point_in_ksi_phi_coordinate.x,
                self.total_disperse_number + 1,
            )  # +1 为了满足分段正确性，即匝数 m，需要用 m+1 个点
        ]

        self.dispersed_path3_points: List[P3] = [
            p3_function(ksi)
            for ksi in BaseUtils.linspace(
                self.starting_point_in_ksi_phi_coordinate.x,
                self.end_point_in_ksi_phi_coordinate.x,
                self.total_disperse_number + 1,
            )  # +1 为了满足分段正确性，见上
        ]

        dispersed_path3: List[List[float]] = [
            p.to_list() for p in self.dispersed_path3_points
        ]

        # 为了速度，转为 numpy
        self.dispersed_path2: numpy.ndarray = numpy.array(dispersed_path2)
        self.dispersed_path3: numpy.ndarray = numpy.array(dispersed_path3)

        # 电流元 (miu0/4pi) * current * (p[i+1] - p[i])
        # refactor v0.1.1
        self.elementary_current = 1e-7 * current * (
            self.dispersed_path3[1:] - self.dispersed_path3[:-1]
        )

        # 电流元的位置 (p[i+1]+p[i])/2
        self.elementary_current_position = 0.5 * (
            self.dispersed_path3[1:] + self.dispersed_path3[:-1]
        )

    def __phi_ksi_function(self, ksi: float) -> float:
        """
        返回一个函数，完成 ξ 到 φ 的映射
        """
        x1 = self.starting_point_in_ksi_phi_coordinate.x
        y1 = self.starting_point_in_ksi_phi_coordinate.y
        x2 = self.end_point_in_ksi_phi_coordinate.x
        y2 = self.end_point_in_ksi_phi_coordinate.y

        k = (y2 - y1) / (x2 - x1)
        b = -k * x1 + y1

        phi = k * ksi + b
        for i in range(len(self.tilt_radians)):
            if BaseUtils.equal(self.tilt_angles[i], 90.0):
                continue
            phi += (
                (1 / math.tan(self.tilt_radians[i]))
                / ((i + 1) * math.sinh(self.eta))
                * math.sin((i + 1) * ksi)
            )
        return phi

    class BipolarToroidalCoordinateSystem:
        """
        双极点坐标系
        """

        def __init__(self, a: float, eta: float, big_r: float, small_r: float):
            self.a = a
            self.eta = eta
            self.big_r = big_r
            self.small_r = small_r

            BaseUtils.equal(
                big_r,
                math.sqrt(a * a / (1 - 1 / math.pow(math.cosh(eta), 2))),
                msg=f"BipolarToroidalCoordinateSystem:init 错误1 a({a})eta({eta})R({big_r})r({small_r})",
            )

            BaseUtils.equal(
                small_r,
                big_r / math.cosh(eta),
                msg=f"BipolarToroidalCoordinateSystem:init 错误2 a({a})eta({eta})R({big_r})r({small_r})",
            )

        def convert(self, p: P2) -> P3:
            """
            将二维坐标 (ξ,φ) 转为三维坐标 (x,y,z)
            """
            ksi = p.x
            phi = p.y
            temp = self.a / (math.cosh(self.eta) - math.cos(ksi))
            return P3(
                temp * math.sinh(self.eta) * math.cos(phi),
                temp * math.sinh(self.eta) * math.sin(phi),
                temp * math.sin(ksi),
            )

        def main_normal_direction_at(self, p: P2) -> P3:
            """
            返回二维坐标 (ξ,φ) 映射到的三维坐标 (x,y,z) 点，
            它在圆环面上的法向量
            即返回值 P3 在这点 (x,y,z) 垂直于圆环面
            """
            phi = p.y

            center = P3(self.big_r * math.cos(phi),
                        self.big_r * math.sin(phi), 0)

            face_point = self.convert(p)

            return (face_point - center).normalize()

        def __str__(self):
            return f"BipolarToroidalCoordinateSystem a({self.a})eta({self.eta})R({self.big_r})r({self.small_r})"

        def __repr__(self) -> str:
            return self.__str__()

    def magnetic_field_at(self, point: P3) -> P3:
        """
        计算 CCT 在全局坐标系点 P3 参数的磁场
        为了计算效率，使用 numpy
        """
        # point 转为局部坐标，并变成 numpy 向量
        p = numpy.array(
            self.local_coordinate_system.point_to_local_coordinate(
                point).to_list()
        )

        # 点 p 到电流元中点
        r = p - self.elementary_current_position

        # 点 p 到电流元中点的距离的三次方
        rr = (numpy.linalg.norm(r, ord=2, axis=1)
              ** (-3)).reshape((r.shape[0], 1))

        # 计算每个电流元在 p 点产生的磁场 (此时还没有乘系数 μ0/4π )
        dB = numpy.cross(self.elementary_current, r) * rr

        # 求和，即得到磁场，
        # (不用乘乘以系数 μ0/4π = 1e-7)
        # refactor v0.1.1
        B = numpy.sum(dB, axis=0)

        # 转回 P3
        B_P3: P3 = P3.from_numpy_ndarry(B)

        # 从局部坐标转回全局坐标
        B_P3: P3 = self.local_coordinate_system.vector_to_global_coordinate(
            B_P3)

        return B_P3

    # from ApertureObject
    def is_out_of_aperture(self, point: P3) -> bool:
        """
        判断点 point 是在 CCT 的孔径内还是孔径外
        只有当粒子轴向投影在元件内部时，才会进行判断，
        否则即时粒子距离轴线很远，也认为粒子没有超出孔径，
        这是因为粒子不在元件内时，很可能处于另一个大孔径元件中，这样会造成误判。
        """
        # 转为局部坐标
        local_point = self.local_coordinate_system.point_to_local_coordinate(
            point)
        local_point_p2 = local_point.to_p2()

        # 查看偏转方向
        clockwise = self.end_point_in_ksi_phi_coordinate.y < 0

        # 映射到 cct 所在圆环轴上
        phi = local_point_p2.angle_to_x_axis()

        # 查看是否在 cct 轴上
        if clockwise:
            # phi 应大于 2pi-bending_radian 小于 2pi
            if phi > (2 * math.pi - self.bending_radian):
                return (
                    abs(local_point.z) > self.small_r
                    or local_point_p2.length() > (self.big_r + self.small_r)
                    or local_point_p2.length() < (self.big_r - self.small_r)
                )
            else:
                return False
        else:
            if phi < self.bending_radian:
                return (
                    abs(local_point.z) > self.small_r
                    or local_point_p2.length() > (self.big_r + self.small_r)
                    or local_point_p2.length() < (self.big_r - self.small_r)
                )
            else:
                return False

    def __str__(self):
        return (
            f"CCT: local_coordinate_system({self.local_coordinate_system})big_r({self.big_r})small_r({self.small_r})"
            + f"bending_angle({self.bending_angle})tilt_angles({self.tilt_angles})winding_number({self.winding_number})"
            + f"current({self.current})starting_point_in_ksi_phi_coordinate({self.starting_point_in_ksi_phi_coordinate})"
            + f"end_point_in_ksi_phi_coordinate({self.end_point_in_ksi_phi_coordinate})"
            + f"disperse_number_per_winding({self.disperse_number_per_winding})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def create_cct_along(
            # 设计轨道
            trajectory: Line2,
            # 设计轨道上该 CCT 起点
            s: float,
            # 大半径：偏转半径
            big_r: float,
            # 小半径（孔径/2）
            small_r: float,
            # 偏转角度，即 phi0*winding_number，典型值 67.5
            bending_angle: float,
            # 各极倾斜角，典型值 [30,90,90,90]
            tilt_angles: List[float],
            # 匝数
            winding_number: int,
            # 电流
            current: float,
            # CCT 路径在二维 ξ-φ 坐标系中的起点
            starting_point_in_ksi_phi_coordinate: P2,
            # CCT 路径在二维 ξ-φ 坐标系中的终点
            end_point_in_ksi_phi_coordinate: P2,
            # 每匝线圈离散电流元数目，数字越大计算精度越高
            disperse_number_per_winding: int = 120,
    ) -> "CCT":
        """
        按照设计轨迹 trajectory 上 s 位置处创建 CCT
        """
        start_point: P2 = trajectory.point_at(s)
        arc_length: float = big_r * BaseUtils.angle_to_radian(bending_angle)
        end_point: P2 = trajectory.point_at(s)

        midpoint0: P2 = trajectory.point_at(s + arc_length / 3 * 1)
        midpoint1: P2 = trajectory.point_at(s + arc_length / 3 * 2)

        c1, r1 = BaseUtils.circle_center_and_radius(
            start_point, midpoint0, midpoint1)
        c2, r2 = BaseUtils.circle_center_and_radius(
            midpoint0, midpoint1, end_point)
        BaseUtils.equal(
            c1, c2, msg=f"构建 CCT 存在异常，通过设计轨道判断 CCT 圆心不一致，c1{c1}，c2{c2}")
        BaseUtils.equal(
            r1, r2, msg=f"构建 CCT 存在异常，通过设计轨道判断 CCT 半径不一致，r1{r1}，r2{r2}")
        center: P2 = (c1 + c2) * 0.5

        start_direct: P2 = trajectory.direct_at(s)
        pos: int = StraightLine2(
            1, start_direct, start_point).position_of(center)

        lcs = None
        if pos == 0:
            raise ValueError(f"错误：圆心{center}在设计轨道{trajectory}上")
        elif pos == 1:
            lcs = LocalCoordinateSystem.create_by_y_and_z_direction(
                location=center.to_p3(),
                y_direction=-start_direct.to_p3(),  # diff
                z_direction=P3.z_direct(),
            )
        else:
            lcs = LocalCoordinateSystem.create_by_y_and_z_direction(
                location=center.to_p3(),
                y_direction=start_direct.to_p3(),  # diff
                z_direction=P3.z_direct(),
            )
        return CCT(
            local_coordinate_system=lcs,
            big_r=big_r,
            small_r=small_r,
            bending_angle=bending_angle,
            tilt_angles=tilt_angles,
            winding_number=winding_number,
            current=current,
            starting_point_in_ksi_phi_coordinate=starting_point_in_ksi_phi_coordinate,
            end_point_in_ksi_phi_coordinate=end_point_in_ksi_phi_coordinate,
            disperse_number_per_winding=disperse_number_per_winding,
        )

    def global_path3(self) -> List[P3]:
        """
        获取 CCT 路径点，以全局坐标系的形式
        主要目的是为了 CUDA 计算
        since v0.1.1
        """
        return [
            self.local_coordinate_system.point_to_global_coordinate(p)
            for p in self.dispersed_path3_points
        ]

    def global_current_elements_and_elementary_current_positions(self, numpy_dtype=numpy.float64) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        获取全局坐标系下的
        电流元 (miu0/4pi) * current * (p[i+1] - p[i])
        和
        电流元的位置 (p[i+1]+p[i])/2
        主要目的是为了 CUDA 计算
        since v0.1.1
        """
        global_path3: List[P3] = self.global_path3()

        global_path3_numpy_array = numpy.array(
            [p.to_list() for p in global_path3], dtype=numpy_dtype)

        global_current_elements = 1e-7 * self.current * \
            (global_path3_numpy_array[1:] - global_path3_numpy_array[:-1])

        global_elementary_current_positions = 0.5 * \
            (global_path3_numpy_array[1:] + global_path3_numpy_array[:-1])
        return (
            global_current_elements.flatten(),
            global_elementary_current_positions.flatten()
        )


class QS(Magnet, ApertureObject):
    """
    硬边 QS 磁铁，由以下参数完全确定：

    length 磁铁长度 / m
    gradient 四极场梯度 / Tm-1
    second_gradient 六极场梯度 / Tm-2
    aperture_radius 孔径（半径） / m
    local_coordinate_system 局部坐标系

    ① QS 磁铁入口中心位置，是局部坐标系的原心
    ② 理想粒子运动方向，是局部坐标系 Z 方向
    ③ 相空间中 X 方向
    因此，垂直屏幕向外（向面部）是 Y 方向

    """

    def __init__(
            self,
            local_coordinate_system: LocalCoordinateSystem,
            length: float,
            gradient: float,
            second_gradient: float,
            aperture_radius: float,
    ):
        self.local_coordinate_system = local_coordinate_system
        self.length = float(length)
        self.gradient = float(gradient)
        self.second_gradient = float(second_gradient)
        self.aperture_radius = float(aperture_radius)

    def __str__(self) -> str:
        """
        since v0.1.1
        """
        return f"local_coordinate_system={self.local_coordinate_system}, length={self.length}, gradient={self.gradient}, second_gradient={self.second_gradient}, aperture_radius={self.aperture_radius}"

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

        if local_point.z > 0 and local_point.z < self.length:
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


class Beamline(Line2, Magnet, ApertureObject):
    def __init__(self, trajectory: Trajectory) -> None:
        """
        不要直接调用构造器
        请使用 set_start_point
        """
        self.magnets: List[Magnet] = []
        self.trajectory: Trajectory = trajectory

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
            footstep: float = 1 * MM,
    ) -> List[P3]:
        """
        束流跟踪，运行一个理想粒子，返回轨迹
        s 起点位置
        length 粒子运行长度，默认运行到束线尾部
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
            concurrency_level: int = 1
    ) -> Tuple[List[P2], List[P2]]:
        """
        束流跟踪，运行一个相椭圆，返回一个长度 2 的元素，表示相空间 x-xp 平面和 y-yp 平面上粒子投影（单位 mm / mrad）
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
            p=rp_x + rp_y, m=self, length=length, footstep=footstep, concurrency_level=concurrency_level
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

        return (
            [pp.project_to_xxp_plane() / MM for pp in pp_x_end],
            [pp.project_to_yyp_plane() / MM for pp in pp_y_end],
        )

    # from ApertureObject
    def is_out_of_aperture(self, point: P3) -> bool:
        """
        判断点 point 是否超出 Beamline 的任意一个元件的孔径
        只有当粒子轴向投影在元件内部时，才会进行判断，
        否则即时粒子距离轴线很远，也认为粒子没有超出孔径，
        这是因为粒子不在元件内时，很可能处于另一个大孔径元件中，这样会造成误判。

        注意：这个函数的效率极低！应使用下面的方法
        """
        for m in self.magnets:
            if isinstance(m, ApertureObject) and m.is_out_of_aperture(point):
                return True

        return False

    def trace_is_out_of_aperture(
            self, trace_with_distance: List[ValueWithDistance[P3]]
    ) -> bool:
        """
        判断一条粒子轨迹是否超出孔径
        """
        raise NotImplemented

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
            return Beamline(
                Trajectory.set_start_point(self.start_point).first_line(
                    direct=direct, length=length
                )
            )

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
        self.trajectory.add_strait_line(length=length)

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

        self.magnets.append(
            QS.create_qs_along(
                trajectory=self.trajectory,
                s=old_length,
                length=length,
                gradient=gradient,
                second_gradient=second_gradient,
                aperture_radius=aperture_radius,
            )
        )

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
        self.trajectory.add_arc_line(
            radius=big_r, clockwise=bending_angle < 0, angle_deg=abs(bending_angle)
        )
        self.magnets.append(
            CCT.create_cct_along(
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
        )

        self.magnets.append(
            CCT.create_cct_along(
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
        )
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
        """
        if len(small_rs) != 4:
            raise ValueError(
                f"small_rs({small_rs})，长度应为4，分别是二极CCT外层、内层，四极CCT外层、内层")
        if not BaseUtils.is_sorted(small_rs[::-1]):
            raise ValueError(
                f"small_rs({small_rs})，应从大到小排列，分别是二极CCT外层、内层，四极CCT外层、内层")

        total_bending_angle = sum(bending_angles)
        old_length = self.trajectory.get_length()
        self.trajectory.add_arc_line(
            radius=big_r,
            clockwise=total_bending_angle < 0,
            angle_deg=abs(total_bending_angle),
        )

        # 构建二极 CCT 外层
        self.magnets.append(
            CCT.create_cct_along(
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
        )

        # 构建二极 CCT 内层
        self.magnets.append(
            CCT.create_cct_along(
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
        )

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
        self.magnets.append(
            CCT.create_cct_along(
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
        )

        self.magnets.append(
            CCT.create_cct_along(
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
        )

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
            self.magnets.append(
                CCT.create_cct_along(
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
            )

            self.magnets.append(
                CCT.create_cct_along(
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
            )

        return self

    def __str__(self) -> str:
        return f"beamline(magnet_size={len(self.magnets)}, traj_len={self.trajectory.get_length()})"


class BaseUtils:
    """
    这里存放一些简单的工具，如
    1. 判断两个对象是否相等
    2. numpy 中用于生成均匀分布的 linspace 方法
    3. 角度转弧度 angle_to_radian 和 弧度转角度 radian_to_angle
    4. 打印函数调用栈 print_traceback （这个主要用于 debug）
    5. 椭圆。用于生成椭圆圆周上均匀分布的若干点
    """

    @staticmethod
    def equal(
            a: Union[float, int, P2, P3],
            b: Union[float, int, P2, P3],
            err: float = 1e-6,
            msg: Optional[str] = None,
    ) -> bool:
        """
        判断 a b 是否相等，相等返回 true
        当 a b 不相等时，若 msg 为空，返回 flase，否则抛出异常，异常信息即 msg

        示例：
        """
        if (isinstance(a, float) or isinstance(a, int)) and (
                isinstance(b, float) or isinstance(b, int)
        ):
            if (
                    a == b
                    or abs(a - b) <= err
                    or ((a + b != 0.0) and ((2 * abs((a - b) / (a + b))) <= err))
            ):
                return True
            else:
                if msg is None:
                    return False
                else:
                    raise AssertionError(msg)
        elif (isinstance(a, P2) and isinstance(b, P2)) or (
                isinstance(a, P3) and isinstance(b, P3)
        ):
            if a.__eq__(b, err=err, msg=msg):
                return True
            else:
                if msg is None:
                    return False
                else:
                    raise AssertionError(msg)
        else:
            if a == b:
                return True
            else:
                if msg is None:
                    return False
                else:
                    raise AssertionError(msg)

    @staticmethod
    def linspace(
            start: Union[float, int, P2, P3], end: Union[float, int, P2, P3], number: int
    ) -> List[Union[float, P2, P3]]:
        """
        同 numpy 的 linspace
        """
        # 除法改成乘法以适应 P2 P3 对象
        d = (end - start) * (1 / (number - 1))
        # i 转为浮点以适应 P2 P3 对象
        return [start + d * float(i) for i in range(number)]

    @staticmethod
    def angle_to_radian(
            deg: Union[float, int, List[Union[float, int]]]
    ) -> Union[float, List[float]]:
        """
        角度值转弧度制
        对于单个角度，或者角度数组都可以使用
        """
        if isinstance(deg, float) or isinstance(deg, int):
            return deg / 180.0 * math.pi
        elif isinstance(deg, List):
            return [BaseUtils.angle_to_radian(d) for d in deg]
        else:
            raise NotImplementedError

    @staticmethod
    def radian_to_angle(
            rad: Union[float, int, List[Union[float, int]]]
    ) -> Union[float, List[float]]:
        """
        弧度制转角度制
        对于单个弧度，或者弧度数组都可以使用
        """
        if isinstance(rad, float) or isinstance(rad, int):
            return rad * 180.0 / math.pi
        elif isinstance(rad, List):
            return [BaseUtils.radian_to_angle(d) for d in rad]
        else:
            raise NotImplementedError

    @staticmethod
    def circle_center_and_radius(p1: P2, p2: P2, p3: P2) -> Tuple[P2, float]:
        """
        已知三个二维点 p1 p2 p3
        求由这三个点组成的圆的圆心和半径
        方法来自：https://blog.csdn.net/liutaojia/article/details/83625151
        """
        x1 = p1.x
        x2 = p2.x
        x3 = p3.x
        y1 = p1.y
        y2 = p2.y
        y3 = p3.y
        z1 = x2 ** 2 + y2 ** 2 - x1 ** 2 - y1 ** 2
        z2 = x3 ** 2 + y3 ** 2 - x1 ** 2 - y1 ** 2
        z3 = x3 ** 2 + y3 ** 2 - x2 ** 2 - y2 ** 2
        A = numpy.array(
            [[(x2 - x1), (y2 - y1)], [(x3 - x1), (y3 - y1)], [(x3 - x2), (y3 - y2)]]
        )
        B = 0.5 * numpy.array([[z1], [z2], [z3]])
        c = numpy.linalg.inv(A.T @ A) @ A.T @ B
        c = P2.from_numpy_ndarry(c)
        # c = (A'*A)\A'*B;
        R1 = math.sqrt((c.x - x1) ** 2 + (c.y - y1) ** 2)
        R2 = math.sqrt((c.x - x2) ** 2 + (c.y - y2) ** 2)
        R3 = math.sqrt((c.x - x3) ** 2 + (c.y - y3) ** 2)
        R = (R1 + R2 + R3) / 3
        return c, R

    @staticmethod
    def polynomial_fitting(xs: List[float], ys: List[float], order: int) -> List[float]:
        """
        多项式拟合
        xs 自变量，ys 变量，拟合阶数为 order，返回一个数组
        数组第 0 项为拟合常数项
        数组第 i 项为拟合 i 次项
        """
        fit = numpy.polyfit(xs, ys, order)
        return fit[::-1].tolist()

    @staticmethod
    def list_multiply(
            li: Union[List[int], List[float], List[P2], List[P3]], number: Union[int, float]
    ) -> Union[List[int], List[float], List[P2], List[P3]]:
        """
        让数组中每个元素都乘以一个数
        """
        return [e * number for e in li]

    @staticmethod
    def is_sorted(li: List) -> bool:
        """
        判断数组是否有序
        这个方法来自 https://www.zhihu.com/question/368573897
        虽然无法快速退出，但很简洁
        """
        return all([li[i] <= li[i + 1] for i in range(len(li) - 1)])

    @staticmethod
    def print_traceback() -> None:
        """
        打印函数调用栈
        用于 debug
        -------

        """
        f = sys._getframe()
        while f is not None:
            print(f)
            f = f.f_back

    @staticmethod
    def runge_kutta4(t0: float, t_end: float, Y0: T, y_derived_function: Callable[[float, T], T], dt: float,
                     record: bool = False) -> Union[T, Tuple[List[float], List[T]]]:
        """
        4 阶 runge kutta 法求解微分方程组
        since v0.1.1
        """
        number: int = math.ceil((t_end - t0) / dt)
        dt = (t_end - t0) / float(number)

        if record:
            ts = [t0]
            Ys = [Y0]
            for ignore in range(number):
                k1 = y_derived_function(t0, Y0)
                k2 = y_derived_function(t0 + dt / 2, Y0 + dt / 2 * k1)
                k3 = y_derived_function(t0 + dt / 2, Y0 + dt / 2 * k2)
                k4 = y_derived_function(t0 + dt, Y0 + dt * k3)

                t0 = t0 + dt
                Y0 = Y0 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                ts.append(t0)
                Ys.append(Y0)
            return (ts, Ys)
        else:
            for ignore in range(number):
                k1 = y_derived_function(t0, Y0)
                k2 = y_derived_function(t0 + dt / 2, Y0 + dt / 2 * k1)
                k3 = y_derived_function(t0 + dt / 2, Y0 + dt / 2 * k2)
                k4 = y_derived_function(t0 + dt, Y0 + dt * k3)

                t0 += dt
                Y0 += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            return Y0

    @staticmethod
    def solve_ode(t0: float, t_end: float, Y0: T, y_derived_function: Callable[[float, T], T], dt: float,
                  record: bool = False, absolute_tolerance: float = 1e-8, relative_tolerance: float = 1e-8) -> Union[
            T, Tuple[List[float], List[T]]]:
        """
        scipy 中 ode45
        即变步长 4 阶 runge kutta 法
        since v0.1.1
        """
        if record:
            raise NotImplementedError
            # number: int = math.ceil((t_end-t0)/dt)
            # t_eval = numpy.linspace(t0, t_end, number)
            # s = solve_ivp(y_derived_function, [
            #               t0, t_end], Y0, t_eval=t_eval, rtol=1e-8, atol=1e-8, first_step=dt, max_step=dt)
        else:
            s = solve_ivp(y_derived_function, [
                t0, t_end], Y0, rtol=1e-8, atol=1e-8, first_step=dt, max_step=dt)
            return s.y

    # 多进程安全提示 since v0.1.1
    __I_AM_SURE_MY_CODE_CLOSED_IN_IF_NAME_EQUAL_MAIN: bool = False

    @classmethod
    def i_am_sure_my_code_closed_in_if_name_equal_main(cls):
        """
        多线程安全提示
        since v0.1.1
        """
        cls.__I_AM_SURE_MY_CODE_CLOSED_IN_IF_NAME_EQUAL_MAIN = True

    @classmethod
    def submit_process_task(cls, task: Callable[..., T], param_list: List[List], concurrency_level: Optional[int] = None) -> \
            List[T]:
        """
        提交任务多进程并行
        task 要运行的任务，是一个函数
        T 任务返回值
        param_list 任务参数数组，数组每个元素表示一个 task 的输出组合
        concurrency_level 并发等级，默认为 CPU 核心数


        因为 python 具有全局解释器锁，所以 CPU 密集任务无法使用线程加速，只能使用进程
        see https://www.cnblogs.com/dragon-123/p/10247252.html

        since v0.1.1
        """
        if not cls.__I_AM_SURE_MY_CODE_CLOSED_IN_IF_NAME_EQUAL_MAIN:
            raise PermissionError(
                "在使用CPU并行计算前，应确保你的脚本写在if __name__ == '__main__':"
                + "代码块内部，并显式调用BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main()函数"
            )

        if concurrency_level is None:
            concurrency_level = os.cpu_count()
        print(f"处理并行任务，任务数目{len(param_list)}，并行等级{concurrency_level}")
        start = time.time()
        pool = multiprocessing.Pool(processes=concurrency_level)  # 开启一次性进程池
        r = pool.starmap(task, param_list)  # 执行任务
        pool.close()  # 停止接受任务
        pool.join()  # 等待完成
        print(f"任务完成，用时{time.time() - start}秒")
        return r

    class Ellipse:
        """
        椭圆类
        Ax^2+Bxy+Cy^2=D
        """

        def __init__(self, A: float, B: float, C: float, D: float):
            self.A = float(A)
            self.B = float(B)
            self.C = float(C)
            self.D = float(D)

        def point_at(self, theta: float) -> P2:
            """
            原点出发，方向th弧度的射线和椭圆Ax^2+Bxy+Cy^2=D的交点
            Parameters
            ----------
            theta 弧度

            Returns 方向th弧度的射线和椭圆Ax^2+Bxy+Cy^2=D的交点
            -------

            """
            d = P2()

            while theta < 0:
                theta += 2 * math.pi

            while theta > 2 * math.pi:
                theta -= 2 * math.pi

            if BaseUtils.equal(theta, 0) or BaseUtils.equal(theta, 2 * math.pi):
                d.x = math.sqrt(self.D / self.A)
                d.y = 0

            if BaseUtils.equal(theta, math.pi):
                d.x = -math.sqrt(self.D / self.A)
                d.y = 0

            t = 0.0

            if 0 < theta < math.pi:
                t = 1 / math.tan(theta)
                d.y = math.sqrt(
                    self.D / (self.A * t * t + self.B * t + self.C))
                d.x = t * d.y

            if math.pi < theta < 2 * math.pi:
                theta -= math.pi
                t = 1 / math.tan(theta)
                d.y = -math.sqrt(self.D / (self.A * t *
                                           t + self.B * t + self.C))
                d.x = t * d.y

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
                c += (
                    self.point_at(2.0 * math.pi / float(num) * (i + 1))
                    - self.point_at(2.0 * math.pi / float(num) * (i))
                ).length()

            return c

        def point_after(self, length: float) -> P2:
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
            step_theta = BaseUtils.angle_to_radian(0.05)
            theta = 0.0
            while length > 0.0:
                length -= (
                    self.point_at(theta + step_theta) - self.point_at(theta)
                ).length()

                theta += step_theta

            return self.point_at(theta)

        def uniform_distribution_points_along_edge(self, num: int) -> List[P2]:
            """
            返回椭圆圆周上均匀分布的 num 个点
            """
            points = []
            c = self.circumference
            for i in range(num):
                points.append(self.point_after(c / num * i))

            return points

    class Statistic:
        """
        统计器
        since v0.1.1
        """

        def __init__(self):
            self.__data: List[float] = []

        def add(self, val: float):
            self.__data.append(val)

        def max(self):
            return numpy.max(self.__data)

        def min(self):
            return numpy.min(self.__data)

        def var(self):
            return numpy.var(self.__data)

        def average(self):
            return sum(self.__data) / len(self.__data)

        def clear(self):
            self.__data: List[float] = []


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
    def plot_beamline(beamline: Beamline, describes=["r-"]) -> None:
        """
        绘制 beamline
        包括 beamline 上的磁铁和设计轨道
        """
        size = len(beamline.magnets)
        for i in range(size):
            b = beamline.magnets[i]
            d = describes[i] if i < len(describes) else describes[-1]
            if isinstance(b, QS):
                Plot3.plot_qs(b, d)
            elif isinstance(b, CCT):
                Plot3.plot_cct(b, d)
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
    def set_center(center: P3, cube_size: float) -> None:
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
            elif isinstance(param1, numpy.ndarray):
                Plot2.plot_ndarry2ds(param1, describe=describe)
            elif isinstance(param1, CCT):
                Plot2.plot_cct_outline(param1, describe=describe)
            elif isinstance(param1, QS):
                Plot2.plot_qs(param1, describe=describe)
            elif isinstance(param1, Beamline):
                Plot2.plot_beamline(param1, describes=["k-", "r-"])
            elif isinstance(param1, Line2):
                Plot2.plot_line2(param1, describe=describe)
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
                print(f"无法绘制{data}")
        else:
            print(f"无法绘制{data}")

    @staticmethod
    def plot_xy(x: float, y: float, describe="r") -> None:
        """
        绘制点 (x,y)
        绘制图象时只有 plot_xy 和 plot_xy_array 访问底层，所以需要判断是否初始化
        """
        if not Plot2.INIT:
            Plot2.__init()

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
    def plot_p2s(ps: List[P2], describe="r-") -> None:
        """
        绘制点 P2 数组，多个点
        """
        Plot2.plot_xy_array([p.x for p in ps], [p.y for p in ps], describe)

    @staticmethod
    def plot_p3s(
            ps: List[P3], p3_to_p2: Callable = lambda p3: P2(p3.x, p3.y), describe="r-"
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
    def plot_beamline(beamline: Beamline, describes=["r-"]) -> None:
        """
        绘制 beamline
        包括 beamline 上的磁铁和设计轨道
        """
        size = len(beamline.magnets)
        for i in range(size):
            b = beamline.magnets[i]
            d = describes[i + 1] if i < (len(describes) - 1) else describes[-1]
            if isinstance(b, QS):
                Plot2.plot_qs(b, d)
            elif isinstance(b, CCT):
                Plot2.plot_cct_outline(b, d)
            else:
                print(f"无法绘制{b}")
        Plot2.plot_line2(beamline.trajectory, describe=describes[0])

    @staticmethod
    def plot_line2(line: Line2, step: float = 1 * MM, describe="r-") -> None:
        """
        绘制 line2
        """
        p2s = line.disperse2d(step)
        Plot2.plot_p2s(p2s, describe)

    @staticmethod
    def equal():
        """
        设置坐标轴比例相同
        """
        if not Plot2.INIT:
            Plot2.__init()
        plt.axis("equal")

    @staticmethod
    def info(
            x_label: str = "",
            y_label: str = "",
            title: str = "",
            font_size: int = 12,
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
    def legend(*labels: Tuple, font_size: int = 12, font_family: str = "Times New Roman") -> NoReturn:
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
            raise RuntimeError("Plot2::请在show前调用plot")

        plt.show()


class HUST_SC_GANTRY:
    def __init__(
        self,
        # ------------------ 前偏转段 ---------------#
        # 漂移段
        DL1=0.8001322,
        GAP1=0.1765959,
        GAP2=0.2960518,
        # qs 磁铁
        qs1_length=0.2997797,
        qs1_aperture_radius=30 * MM,
        qs1_gradient=28.33,
        qs1_second_gradient=-140.44 * 2.0,
        qs2_length=0.2585548,
        qs2_aperture_radius=30 * MM,
        qs2_gradient=-12.12,
        qs2_second_gradient=316.22 * 2.0,
        # cct 偏转半径
        cct12_big_r=0.95,
        # cct 孔径
        agcct12_inner_small_r=35 * MM,
        agcct12_outer_small_r=35 * MM,
        dicct12_inner_small_r=35 * MM,
        dicct12_outer_small_r=35 * MM,
        # cct 匝数
        agcct1_winding_number=30,
        agcct2_winding_number=39,
        dicct12_winding_number=71,
        # cct 角度
        dicct12_bending_angle=22.5,
        agcct1_bending_angle=9.782608695652174,
        agcct2_bending_angle=12.717391304347826,
        # cct 倾斜角（倾角 90 度表示不倾斜）
        dicct12_tilt_angles=[30, 80],
        agcct12_tilt_angles=[90, 30],
        # cct 电流
        dicct12_current=-6192,
        agcct12_current=-3319,
        # ------------------ 后偏转段 ---------------#
        # 漂移段
        DL2=2.1162209,
        GAP3=0.1978111,
        # qs 磁铁
        qs3_length=0.2382791,
        qs3_aperture_radius=60 * MM,
        qs3_gradient=-7.3733,
        qs3_second_gradient=-45.31 * 2,
        # cct 偏转半径
        cct345_big_r=0.95,
        # cct 孔径
        agcct345_inner_small_r=83 * MM,
        agcct345_outer_small_r=83 * MM + 15 * MM,
        dicct345_inner_small_r=83 * MM + 15 * MM * 2,
        dicct345_outer_small_r=83 * MM + 15 * MM * 3,
        # cct 匝数
        agcct3_winding_number=21,
        agcct4_winding_number=50,
        agcct5_winding_number=50,
        dicct345_winding_number=128,
        # cct 角度（负数表示顺时针偏转）
        dicct345_bending_angle=-67.5,
        agcct3_bending_angle=-(8 + 3.716404),
        agcct4_bending_angle=-(8 + 19.93897),
        agcct5_bending_angle=-(8 + 19.844626),
        # cct 倾斜角（倾角 90 度表示不倾斜）
        dicct345_tilt_angles=[30, 80],
        agcct345_tilt_angles=[90, 30],
        # cct 电流
        dicct345_current=9664,
        agcct345_current=-6000,

        part_per_winding=120,
    ) -> None:
        # ------------------ 前偏转段 ---------------#
        # 漂移段
        self.DL1 = DL1
        self.GAP1 = GAP1
        self.GAP2 = GAP2
        # qs 磁铁
        self.qs1_length = qs1_length
        self.qs1_aperture_radius = qs1_aperture_radius
        self.qs1_gradient = qs1_gradient
        self.qs1_second_gradient = qs1_second_gradient
        self.qs2_length = qs2_length
        self.qs2_aperture_radius = qs2_aperture_radius
        self.qs2_gradient = qs2_gradient
        self.qs2_second_gradient = qs2_second_gradient
        # cct 偏转半径
        self.cct12_big_r = cct12_big_r
        # cct 孔径
        self.agcct12_inner_small_r = agcct12_inner_small_r
        self.agcct12_outer_small_r = agcct12_outer_small_r
        self.dicct12_inner_small_r = dicct12_inner_small_r
        self.dicct12_outer_small_r = dicct12_outer_small_r
        # cct 匝数
        self.agcct1_winding_number = agcct1_winding_number
        self.agcct2_winding_number = agcct2_winding_number
        self.dicct12_winding_number = dicct12_winding_number
        # cct 角度
        self.dicct12_bending_angle = dicct12_bending_angle
        self.agcct1_bending_angle = agcct1_bending_angle
        self.agcct2_bending_angle = agcct2_bending_angle
        # cct 倾斜角（倾角 90 度表示不倾斜）
        self.dicct12_tilt_angles = dicct12_tilt_angles
        self.agcct12_tilt_angles = agcct12_tilt_angles
        # cct 电流
        self.dicct12_current = dicct12_current
        self.agcct12_current = agcct12_current
        # ------------------ 后偏转段 ---------------#
        # 漂移段
        self.DL2 = DL2
        self.GAP3 = GAP3
        # qs 磁铁
        self.qs3_length = qs3_length
        self.qs3_aperture_radius = qs3_aperture_radius
        self.qs3_gradient = qs3_gradient
        self.qs3_second_gradient = qs3_second_gradient
        # cct 偏转半径
        self.cct345_big_r = cct345_big_r
        # cct 孔径
        self.agcct345_inner_small_r = agcct345_inner_small_r
        self.agcct345_outer_small_r = agcct345_outer_small_r
        self.dicct345_inner_small_r = dicct345_inner_small_r
        self.dicct345_outer_small_r = dicct345_outer_small_r
        # cct 匝数
        self.agcct3_winding_number = agcct3_winding_number
        self.agcct4_winding_number = agcct4_winding_number
        self.agcct5_winding_number = agcct5_winding_number
        self.dicct345_winding_number = dicct345_winding_number
        # cct 角度（负数表示顺时针偏转）
        self.dicct345_bending_angle = dicct345_bending_angle
        self.agcct3_bending_angle = agcct3_bending_angle
        self.agcct4_bending_angle = agcct4_bending_angle
        self.agcct5_bending_angle = agcct5_bending_angle
        # cct 倾斜角（倾角 90 度表示不倾斜）
        self.dicct345_tilt_angles = dicct345_tilt_angles
        self.agcct345_tilt_angles = agcct345_tilt_angles
        # cct 电流
        self.dicct345_current = dicct345_current
        self.agcct345_current = agcct345_current

        self.part_per_winding = part_per_winding

        # -------- make ---------
        self.__beamline = None
        self.__first_bending_part_length = None

    def create_beamline(self):
        if self.__beamline is None:
            self.__beamline = (
                Beamline.set_start_point(P2.origin())  # 设置束线的起点
                # 设置束线中第一个漂移段（束线必须以漂移段开始）
                .first_drift(direct=P2.x_direct(), length=self.DL1)
                .append_agcct(  # 尾接 acgcct
                    big_r=self.cct12_big_r,  # 偏转半径
                    # 二极 CCT 和四极 CCT 孔径
                    small_rs=[self.dicct12_outer_small_r, self.dicct12_inner_small_r,
                              self.agcct12_outer_small_r, self.agcct12_inner_small_r],
                    bending_angles=[self.agcct1_bending_angle,
                                    self.agcct2_bending_angle],  # agcct 每段偏转角度
                    tilt_angles=[self.dicct12_tilt_angles,
                                 self.agcct12_tilt_angles],  # 二极 CCT 和四极 CCT 倾斜角
                    winding_numbers=[[self.dicct12_winding_number], [
                        self.agcct1_winding_number, self.agcct2_winding_number]],  # 二极 CCT 和四极 CCT 匝数
                    # 二极 CCT 和四极 CCT 电流
                    currents=[self.dicct12_current, self.agcct12_current],
                    disperse_number_per_winding=self.part_per_winding  # 每匝分段数目
                )
                .append_drift(self.GAP1)  # 尾接漂移段
                .append_qs(  # 尾接 QS 磁铁
                    length=self.qs1_length,
                    gradient=self.qs1_gradient,
                    second_gradient=self.qs1_second_gradient,
                    aperture_radius=self.qs1_aperture_radius
                )
                .append_drift(self.GAP2)
                .append_qs(
                    length=self.qs2_length,
                    gradient=self.qs2_gradient,
                    second_gradient=self.qs2_second_gradient,
                    aperture_radius=self.qs2_aperture_radius
                )
                .append_drift(self.GAP2)
                .append_qs(
                    length=self.qs1_length,
                    gradient=self.qs1_gradient,
                    second_gradient=self.qs1_second_gradient,
                    aperture_radius=self.qs1_aperture_radius
                )
                .append_drift(self.GAP1)
                .append_agcct(
                    big_r=self.cct12_big_r,
                    small_rs=[self.dicct12_outer_small_r, self.dicct12_inner_small_r,
                              self.agcct12_outer_small_r, self.agcct12_inner_small_r],
                    bending_angles=[self.agcct2_bending_angle,
                                    self.agcct1_bending_angle],
                    tilt_angles=[self.dicct12_tilt_angles,
                                 self.agcct12_tilt_angles],
                    winding_numbers=[[self.dicct12_winding_number], [
                        self.agcct2_winding_number, self.agcct1_winding_number]],
                    currents=[self.dicct12_current, self.agcct12_current],
                    disperse_number_per_winding=self.part_per_winding
                )
                .append_drift(self.DL1)
            )

            # 把偏转段的磁铁都删了
            self.__beamline.magnets.clear()

            self.__first_bending_part_length = self.__beamline.get_length()

            self.__beamline = (
                self.__beamline.append_drift(self.DL2)
                .append_agcct(
                    big_r=self.cct345_big_r,
                    small_rs=[self.dicct345_outer_small_r, self.dicct345_inner_small_r,
                              self.agcct345_outer_small_r, self.agcct345_inner_small_r],
                    bending_angles=[self.agcct3_bending_angle,
                                    self.agcct4_bending_angle, self.agcct5_bending_angle],
                    tilt_angles=[self.dicct345_tilt_angles,
                                 self.agcct345_tilt_angles],
                    winding_numbers=[[self.dicct345_winding_number], [
                        self.agcct3_winding_number, self.agcct4_winding_number, self.agcct5_winding_number]],
                    currents=[self.dicct345_current, self.agcct345_current],
                    disperse_number_per_winding=self.part_per_winding
                )
                .append_drift(self.GAP3)
                .append_qs(
                    length=self.qs3_length,
                    gradient=self.qs3_gradient,
                    second_gradient=self.qs3_second_gradient,
                    aperture_radius=self.qs3_aperture_radius
                )
                .append_drift(self.GAP3)
                .append_agcct(
                    big_r=self.cct345_big_r,
                    small_rs=[self.dicct345_outer_small_r, self.dicct345_inner_small_r,
                              self.agcct345_outer_small_r, self.agcct345_inner_small_r],
                    bending_angles=[self.agcct5_bending_angle,
                                    self.agcct4_bending_angle, self.agcct3_bending_angle],
                    tilt_angles=[self.dicct345_tilt_angles,
                                 self.agcct345_tilt_angles],
                    winding_numbers=[[self.dicct345_winding_number], [
                        self.agcct5_winding_number, self.agcct4_winding_number, self.agcct3_winding_number]],
                    currents=[self.dicct345_current, self.agcct345_current],
                    disperse_number_per_winding=self.part_per_winding
                )
                .append_drift(self.DL2)
            )

        return self.__beamline

    def first_bending_part_length(self):
        if self.__beamline is None:
            self.beamline()

        return self.__first_bending_part_length

    def create_second_bending_part(self, start_point: P2, start_driect: P2) -> Beamline:
        return (
            Beamline.set_start_point(start_point)  # 设置束线的起点
            # 设置束线中第一个漂移段（束线必须以漂移段开始）
            .first_drift(direct=start_driect, length=self.DL2)
            .append_agcct(
                big_r=self.cct345_big_r,
                small_rs=[self.dicct345_outer_small_r, self.dicct345_inner_small_r,
                          self.agcct345_outer_small_r, self.agcct345_inner_small_r],
                bending_angles=[self.agcct3_bending_angle,
                                self.agcct4_bending_angle, self.agcct5_bending_angle],
                tilt_angles=[self.dicct345_tilt_angles,
                             self.agcct345_tilt_angles],
                winding_numbers=[[self.dicct345_winding_number], [
                    self.agcct3_winding_number, self.agcct4_winding_number, self.agcct5_winding_number]],
                currents=[self.dicct345_current, self.agcct345_current],
                disperse_number_per_winding=self.part_per_winding
            )
            .append_drift(self.GAP3)
            .append_qs(
                length=self.qs3_length,
                gradient=self.qs3_gradient,
                second_gradient=self.qs3_second_gradient,
                aperture_radius=self.qs3_aperture_radius
            )
            .append_drift(self.GAP3)
            .append_agcct(
                big_r=self.cct345_big_r,
                small_rs=[self.dicct345_outer_small_r, self.dicct345_inner_small_r,
                          self.agcct345_outer_small_r, self.agcct345_inner_small_r],
                bending_angles=[self.agcct5_bending_angle,
                                self.agcct4_bending_angle, self.agcct3_bending_angle],
                tilt_angles=[self.dicct345_tilt_angles,
                             self.agcct345_tilt_angles],
                winding_numbers=[[self.dicct345_winding_number], [
                    self.agcct5_winding_number, self.agcct4_winding_number, self.agcct3_winding_number]],
                currents=[self.dicct345_current, self.agcct345_current],
                disperse_number_per_winding=self.part_per_winding
            )
            .append_drift(self.DL2)
        )


def beamline_phase_ellipse_multi_delta(bl: Beamline, particle_number: int,
                                       dps: List[float], describles: str = ['r.', 'y.', 'b.', 'k.', 'g.', 'c.', 'm.']):
    if len(dps) > len(describles):
        raise ValueError(
            f'describles(size={len(describles)}) 长度应大于 dps(size={len(dps)})')
    xs = []
    ys = []
    for dp in dps:
        x, y = bl.track_phase_ellipse(
            x_sigma_mm=3.5, xp_sigma_mrad=7.5,
            y_sigma_mm=3.5, yp_sigma_mrad=7.5,
            delta=dp, particle_number=particle_number,
            kinetic_MeV=215, concurrency_level=16,
            footstep=10*MM
        )
        xs.append(x)
        ys.append(y)

    plt.subplot(121)

    for i in range(len(dps)):
        plt.plot(*P2.extract(xs[i]), describles[i])
    plt.xlabel(xlabel='x/mm')
    plt.ylabel(ylabel='xp/mr')
    plt.title(label='x-plane')
    plt.legend(['dp'+str(int(dp*100)) for dp in dps])
    plt.axis("equal")

    plt.subplot(122)
    for i in range(len(dps)):
        plt.plot(*P2.extract(ys[i]), describles[i])
    plt.xlabel(xlabel='y/mm')
    plt.ylabel(ylabel='yp/mr')
    plt.title(label='y-plane')
    plt.legend(['dp'+str(int(dp*100)) for dp in dps])
    plt.axis("equal")

    plt.show()


if __name__ == "__main__":
    BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main()
    gantry = HUST_SC_GANTRY(
        qs3_gradient=-2.150817888202199435e-01,
        qs3_second_gradient=1.227152952916246242e+01,
        dicct345_tilt_angles=[30, 6.969617162192470516e+01, 1.171773015895775671e+02, 9.995912917953801013e+01 ],
        agcct345_tilt_angles=[1.121898374542038397e+02 , 30, 1.127610021302552781e+02, 9.601700316814657299e+01],
        dicct345_current=-7.127565592411993748e+03,
        agcct345_current=6.430413641971808829e+03,
    )
    bl_all = gantry.create_beamline()

    f = gantry.first_bending_part_length()

    sp = bl_all.trajectory.point_at(f)
    sd = bl_all.trajectory.direct_at(f)

    bl = gantry.create_second_bending_part(sp, sd)

    beamline_phase_ellipse_multi_delta(
        bl,8,[-0.05,0,0.05]
    )
