"""
CCT 建模优化代码
点、坐标系

作者：赵润晓
日期：2021年4月24日
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
from packages.constants import *

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

    def __add__(self, another) -> "P2":
        """
        矢量加法，返回新矢量
        """
        return P2(self.x + another.x, self.y + another.y)

    def __neg__(self) -> "P2":
        """
        相反方向的矢量
        """
        return P2(-self.x, -self.y)

    def __sub__(self, another) -> "P2":
        """
        矢量减法，返回新矢量
        """
        return self.__add__(another.__neg__())

    def __iadd__(self, another) -> "P2":
        """
        矢量原地相加，self 自身改变
        """
        self.x += another.x
        self.y += another.y
        return self  # 必须显式返回

    def __isub__(self, another) -> "P2":
        """
        矢量原地减法，self 自身改变
        """
        self.x -= another.x
        self.y -= another.y
        return self

    def _matmul(self, matrix: List[List[float]]) -> "P2":
        """
        2*2矩阵和 self 相乘，仅仅用于矢量旋转。返回新矢量
        """
        return P2(
            matrix[0][0] * self.x + 
            matrix[0][1] * self.y, 
            matrix[1][0] * self.x + 
            matrix[1][1] * self.y
        )

    @staticmethod
    def _rotation_matrix(phi: float) -> List[List[float]]:
        """
        获取旋转矩阵
        """
        return [[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]]

    def rotate(self, phi: float) -> "P2":
        """
        矢量旋转，返回新矢量
        正角表示逆时针旋转
        """
        return self._matmul(P2._rotation_matrix(phi))

    def angle_to_x_axis(self) -> float:
        """
        矢量和 x 轴的夹角，弧度
        """
        a = float(math.atan2(self.y, self.x))
        return a if a >= 0 else math.pi * 2 + a

    def __mul__(self, another: Union[float, int, "P2"]) -> Union[float, "P2"]:
        """
        矢量乘标量，各元素相乘，返回新矢量
        矢量乘矢量，内积，返回标量
        """
        if isinstance(another, float) or isinstance(another, int):
            return P2(self.x * another, self.y * another)
        else:
            return self.x * another.x + self.y * another.y

    def __rmul__(self, another: Union[float, int]) -> "P2":
        """
        当左操作数不支持相应的操作时被调用
        """
        return self.__mul__(another)

    def __truediv__(self, number: Union[int, float]) -> "P2":
        """
        矢量除法 p2 / number，实际上是 p2 * (1/number)
        """
        if isinstance(number, int) or isinstance(number, float):
            return self * (1 / number)
        else:
            raise ValueError(f"P2{self}仅支持数字除法")

    def angle_to(self, another: "P2") -> float:
        """
        矢量 self 到 另一个矢量 another 的夹角
        """
        to_x = self.angle_to_x_axis()
        s = self.rotate(-to_x)
        o = another.rotate(-to_x)
        return o.angle_to_x_axis()
        # 下面求的仅仅是 矢量 self 和 另一个矢量 another 的夹角
        # theta = (self * another) / (self.length() * another.length())
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
        return f"({self.x}, {self.y})"

    def __repr__(self) -> str:
        """
        == __str__ 用于打印矢量值
        """
        return self.__str__()

    def __eq__(self, another: "P2", err: float = 1e-6, msg: Optional[str] = None) -> bool:
        """
        矢量相等判断
        """
        if not isinstance(another,P2):
            raise ValueError(f"{another} 不是 P2 不能进行相等判断")

        if abs(self.x-another.x)<=err and abs(self.y-another.y)<=err:
            return True
        else:
            if msg is None:
                return False
            else:
                raise AssertionError(msg)

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

    @classmethod
    def from_list(cls, number_list: List) -> Union["P2", List["P2"]]:
        """
        将 list 转为 P2 或者 P2 数组
        如果 list 中元素为数字，则取前两个元素转为 P2
        如果 list 中元素也是 list，则迭代进行
        """
        list_len = len(number_list)
        if list_len == 0:
            raise ValueError("P2.from_list number_list 长度必须大于0")
        element = number_list[0]

        if isinstance(element, int) or isinstance(element, float):
            if list_len>=2:
                if list_len>=3:
                    warnings.warn(f"{list}长度过长，仅将前 2 项转为 P2")
                return P2(element, number_list[1])
            else:
                raise ValueError(f"{number_list}过短，无法转为 P2 或者 P2 数组")
        elif isinstance(element, List):
            return [cls.from_list(number_list[i]) for i in range(list_len)]
        else:
            raise ValueError(f"P2.from_list 无法将{number_list}转为 P2 或者 P2 数组")

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
        if not isinstance(p2_list,list):
            p2_list = [p2_list]
        
        if len(p2_list)<=0:
            return ([],[])
        
        if not isinstance(p2_list[0],P2):
            raise ValueError(f"p2_list 不是 P2 数组，p2_list={p2_list}")

        return ([
            p.x for p in p2_list
        ], [
            p.y for p in p2_list
        ])

    @staticmethod
    def extract_x(p2_list: List['P2']) -> List[float]:
        """
        see extract
        since v0.1.3
        """
        return P2.extract(p2_list)[0]

    @staticmethod
    def extract_y(p2_list: List['P2']) -> List[float]:
        """
        see extract
        since v0.1.3
        """
        return P2.extract(p2_list)[1]


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

    def __add__(self, another) -> "P3":
        """
        矢量相加
        """
        return P3(self.x + another.x, self.y + another.y, self.z + another.z)

    def __neg__(self) -> "P3":
        """
        相反矢量
        """
        return P3(-self.x, -self.y, -self.z)

    def __sub__(self, another) -> "P3":
        """
        矢量相减
        """
        return self.__add__(another.__neg__())

    def __iadd__(self, another) -> "P3":
        """
        矢量原地相加
        """
        self.x += another.x
        self.y += another.y
        self.z += another.z
        return self

    def __isub__(self, another) -> "P3":
        """
        矢量原地减法
        """
        self.x -= another.x
        self.y -= another.y
        self.z -= another.z
        return self

    def __mul__(self, another: Union[float, int, "P3"]) -> Union[float, "P3"]:
        """
        矢量乘标量，各元素相乘，返回新矢量
        矢量乘矢量，内积，返回标量
        """
        if isinstance(another, float) or isinstance(another, int):
            return P3(self.x * another, self.y * another, self.z * another)
        elif isinstance(another,P3):
            return self.x * another.x + self.y * another.y + self.z * another.z
        else:
            raise ValueError(f"{self}和{another}不支持乘法运算")

    def __rmul__(self, another: Union[float, int]) -> "P3":
        """
        当左操作数不支持相应的操作时被调用
        """
        return self.__mul__(another)

    def __truediv__(self, number: Union[int, float]) -> "P3":
        if isinstance(number, int) or isinstance(number, float):
            if number == 0 or number==0.0:
               raise ValueError(f"{self}/{number}，除0异常") 
            return self * (1 / number)
        else:
            raise ValueError("P2仅支持数字除法")

    def __matmul__(self, another: "P3") -> "P3":
        """
        矢量叉乘 / 外积，返回新矢量
        """
        return P3(
            self.y * another.z - self.z * another.y,
            -self.x * another.z + self.z * another.x,
            self.x * another.y - self.y * another.x,
        )

    def __str__(self) -> str:
        """
        矢量信息
        """
        return f"({self.x}, {self.y}, {self.z})"

    def __repr__(self) -> str:
        """
        同 __str__
        """
        return self.__str__()

    def __eq__(self, another: "P3", err: float = 1e-6, msg: Optional[str] = None) -> bool:
        """
        矢量相等判断
        """
        if not isinstance(another,P3):
            raise ValueError(f"{another} 不是 P3 不能进行相等判断")

        if abs(self.x-another.x)<=err and abs(self.y-another.y)<=err and abs(self.z-another.z)<=err:
            return True
        else:
            if msg is None:
                return False
            else:
                raise AssertionError(msg)

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

    def populate(self, another: 'P3') -> None:
        """
        将 another 的值赋到 self 中
        since v0.1.1
        """
        self.x = another.x
        self.y = another.y
        self.z = another.z

    @staticmethod
    def random() -> 'P3':
        """
        随机产生一个 P3
        random.random() 返回随机生成的一个实数，它在[0,1)范围内。
        since v0.1.1
        """
        return P3(random.random(), random.random(), random.random())

    @staticmethod
    def as_p3(anything) -> 'P3':
        """
        伪类型转换
        用于 IDE 智能提示
        """
        return anything

    @staticmethod
    def extract(p3_list: List['P3']) -> Tuple[List[float], List[float],List[float]]:
        """
        提取 P3 数组中的 x y z，各自组成数组
        含义见 P2.extract()
        """
        return ([
            p.x for p in p3_list
        ], [
            p.y for p in p3_list
        ],[
            p.z for p in p3_list
        ])

    @staticmethod
    def extract_x(p3_list: List['P3']) -> List[float]:
        """
        提取 P3 数组中的 x y z，各自组成数组
        """
        return [p.x for p in p3_list]

    @staticmethod
    def extract_y(p3_list: List['P3']) -> List[float]:
        """
        提取 P3 数组中的 x y z，各自组成数组
        """
        return [p.y for p in p3_list]

    @staticmethod
    def extract_z(p3_list: List['P3']) -> List[float]:
        """
        提取 P3 数组中的 x y z，各自组成数组
        """
        return [p.z for p in p3_list]


class ValueWithDistance(Generic[T]):
    """
    辅助对象，带有距离的一个量，通常用于描述线上磁场分布
    """

    def __init__(self, value: T, distance: float) -> None:
        self.value: T = value
        self.distance: float = distance

    def __str__(self) -> str:
        """
        转为字符串
        """
        return f"({self.distance}:{self.value})"

    def __repr__(self) -> str:
        """
        同 __str__()
        """
        return self.__str__()

    @staticmethod
    def convert_to_p2(
        data:Union["ValueWithDistance",List["ValueWithDistance"]],
        convertor:Callable[[T],float]
    )->Union[P2,List[P2]]:
        """
        将 ValueWithDistance 对象转为 P2 对象
        其中 p2.x = distance
        """
        if isinstance(data,ValueWithDistance):
            return P2(data.distance,convertor(data.value))
        else:
            return [
                P2(each.distance,convertor(each.value))
                for each in data
            ]

