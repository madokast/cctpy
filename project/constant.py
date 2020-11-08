"""
常量
"""

from numpy import zeros, ndarray, array, float64

M: float = 1.  # 单位 米，默认的长度单位
MM: float = 0.001 * M  # 单位 毫米
ORIGIN3: ndarray = zeros((3,))  # 三维坐标系原点
ZERO3: ndarray = ORIGIN3  # 三维零矢量
XI: ndarray = array([1, 0, 0], dtype=float64)
YI: ndarray = array([0, 1, 0], dtype=float64)
ZI: ndarray = array([0, 0, 1], dtype=float64)
