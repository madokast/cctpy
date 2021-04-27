"""
CCT 建模优化代码
常量

作者：赵润晓
日期：2021年4月24日
"""
from typing import TypeVar

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