"""
CCT 建模优化代码
ValueWithDistance 使用示例

作者：赵润晓
日期：2021年4月27日
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *


# ValueWithDistance 代表距离和一个对象的组合

# 例如距离 0 上存在一句话 "hello"
hello_at_0 = ValueWithDistance(value="hello",distance=0)
print("hello_at_0",hello_at_0)
# (0:hello)

# 例如距离 1 上存在一个二维点 (2,3)
point23_at_1 = ValueWithDistance(value=P2(2,3),distance=1)
print("point23_at_1",point23_at_1)
# (1:(2.0, 3.0))

# 例如距离 2.5 上存在一个三维零矢量 (0,0,0)
zero3_at_2d5 = ValueWithDistance(value=P3.zeros(),distance=2.5)
print("zero3_at_2d5",zero3_at_2d5)
# (2.5:(0.0, 0.0, 0.0))

# 这个类对象常出现在数组中，用来显示一条路径上的磁场分布