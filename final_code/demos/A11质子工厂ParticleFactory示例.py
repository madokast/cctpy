"""
CCT 建模优化代码
A11 质子工厂 ParticleFactory 示例


作者：赵润晓
日期：2021年4月29日
"""

from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

# ParticleFactory 类提供了方便的构造质子/质子群的函数

print(ParticleFactory)