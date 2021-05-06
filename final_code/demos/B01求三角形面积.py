"""
CCT 建模优化代码
求三角形面积，展示P2类的使用方法

作者：赵润晓
日期：2021年5月3日
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *


# 方法一：海伦公式
# 定义三角形三个顶点
A = P2(21, 8)
B = P2(33, 28)
C = P2(39, 6)

# 求三条边的长度
AB_length = (A-B).length()
AC_length = (A-C).length()
BC_length = (B-C).length()

# 求 p
p = (AB_length+AC_length+BC_length)/2

# 海伦公式
S = math.sqrt(p*(p-AB_length)*(p-AC_length)*(p-BC_length))

# 输出
print(S) # 191.99999999999994


# 方法一：正弦面机公式 S = (1/2)ab*sin(c)
# 定义三角形三个顶点
A = P2(21, 8)
B = P2(33, 28)
C = P2(39, 6)

# 求 AB 和 BC 矢量，用于求其长度和夹角
AB_vector = B-A
AC_vector = C-A

# 求矢量 AB 和 BC 夹角 ∠BAC。a.angle_to(b)求矢量 a 到 b 的角度，
angle_BAC = AC_vector.angle_to(AB_vector)

# 求 矢量 AB 和 BC 的长度
AB_length = AB_vector.length()
AC_length = AC_vector.length()

# 求面积
S = 0.5*AB_length*AC_length*math.sin(angle_BAC)

# 输出
print(S)
# 

a=P2()
b=P2()


included_angle = a.angle_to(b)
if included_angle > math.pi:
    included_angle = included_angle - math.pi