""""
基础工具函数
"""

from numpy import ndarray, linalg, abs


def equal_float(a: float, b: float, err: float = 1e-10) -> bool:
    """
    浮点数相等判断
    err 允许误差
    return |a-b|<err
    """
    return abs(a - b) < err


def equal_vector(v1: ndarray, v2: ndarray, err: float = 1e-10) -> bool:
    """
    判断两个矢量是否相等
    err 允许误差
    """
    s = v1 - v2
    return equal_float(length(abs(s)), 0, err)


def length(vector: ndarray) -> float:
    """
    求矢量的长度
    vector 一维矢量，[a1,a2,a3...]
    return 矢量长度，sqrt(a1**2+a2**2+...)

    实际计算的是 ndarray 的二范数
    """
    return linalg.norm(vector, ord=2)


def update_length(vector: ndarray, new_length: float) -> ndarray:
    """
    原地改变矢量的长度
    vector 矢量
    new_length 长度 应大于0
    Returns vector
    """
    len = length(vector)
    vector *= (new_length / len)
    return vector

def normalize_locally(vector: ndarray) -> ndarray:
    """
    原地矢量归一化
    vector 应当是一个一维矢量，如 [a,b,c]
    即将矢量的长度变为 1
    """
    return update_length(vector, 1.)
