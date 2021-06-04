"""
CCT 建模优化代码
绝对值最大值测试

作者：赵润晓
日期：2021年6月4日
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

arr = [1,2,3,-4]
print(numpy.abs(arr))
print(type(numpy.abs(arr)))

print(numpy.max(numpy.abs(arr)))

s = BaseUtils.Statistic()
s.add_all(arr)
print(s.absolute_max())
