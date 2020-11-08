""""
基础工具函数
"""

import numpy as np


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
