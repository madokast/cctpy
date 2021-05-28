"""
CCT 建模优化代码
绘图工具

作者：赵润晓
日期：2021年5月9日
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys

sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

Plot2.plot_xxx()
Plot2.show()

Plot3.plot_xxx()
Plot3.show()