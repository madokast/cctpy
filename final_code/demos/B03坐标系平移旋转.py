"""
CCT 建模优化代码
坐标系平移旋转

作者：赵润晓
日期：2021年5月4日
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

# X'Y'Z' 局部坐标系，因为相对于全局坐标系只有原点 location 不同，坐标轴方向采用默认值，即于全局坐标系一致
lcs = LocalCoordinateSystem(location=P3(2,2,1))

# 定义全局坐标系中的点
p_gcs = P3(2,3,3)

# 转为局部坐标系
p_lcs = lcs.point_to_local_coordinate(p_gcs)

# 可以再次转回全局坐标
p_gcs_2 = lcs.point_to_global_coordinate(p_lcs)

print(p_gcs)
print(p_lcs)
print(p_gcs == p_gcs_2)

# ------------------------------------------------ #

# X'Y'Z'
lcs_1 = LocalCoordinateSystem(x_direction=P3(
    math.cos(BaseUtils.angle_to_radian(30)),
    math.sin(BaseUtils.angle_to_radian(30)),
    0
))

# X''Y''Z''
lcs_2 = LocalCoordinateSystem(location=P3(8,8,0),
    x_direction=P3(
    math.cos(BaseUtils.angle_to_radian(30)),
    math.sin(BaseUtils.angle_to_radian(30)),
    0
))

p = P3(10,16,0)

p1 = lcs_1.point_to_local_coordinate(p)
p2 = lcs_2.point_to_local_coordinate(p)

print(p1) # (16.660254037844386, 8.856406460551021, 0.0)
print(p2) # (5.732050807568877, 5.92820323027551, 0.0)