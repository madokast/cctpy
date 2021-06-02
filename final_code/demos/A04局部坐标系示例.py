"""
CCT 建模优化代码
局部坐标系

作者：赵润晓
日期：2021年4月27日
"""

from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

# 为了便于磁场建模、粒子跟踪、束流分析，cctpy 中引入了全局坐标系和局部坐标系的概念
# 各种磁铁都放置在局部坐标系中，而粒子在全局坐标系中运动，为了求磁铁在粒子位置产生的磁场，需要引入局部坐标的概念和坐标变换。

# 局部坐标系有4个参数：原点、x轴方向、y轴方向、z轴方向。注意x轴方向、y轴方向、z轴方向不是互相独立的，可以通过右手法则确定，因此构建一个局部坐标系，需要指定3个参数。

# 注：这里的坐标系都是三维直角坐标系，且无缩放 

# 构造一个局部坐标系，需要指定坐标原点，以及 x 轴和 z 轴的方向（y 轴方向随之确定）
# LocalCoordinateSystem() 传参参数如下
# location 全局坐标系中实体位置，默认全局坐标系的远点
# x_direction 局部坐标系 x 方向，默认全局坐标系 x 方向
# z_direction 局部坐标系 z 方向，默认全局坐标系 z 方向
# y 方向由 x 方向和 z 方向计算获得
default_lcs = LocalCoordinateSystem()
print(default_lcs)
# LOCATION=(0.0, 0.0, 0.0), xi=(1.0, 0.0, 0.0), yi=(0.0, 1.0, 0.0), zi=(0.0, 0.0, 1.0)


# 坐标平移。构建一个局部坐标系，原点为 (2,2,1)，x y z 三个轴的方向和全局坐标系一致
lcs221 = LocalCoordinateSystem(location=P3(2,2,1))
# 定义全局坐标i的点 (2,3,3)
point_gcs_233 = P3(2,3,3)
# 函数 point_to_local_coordinate(global_coordinate_point) 将全局坐标系表示的点 global_coordinate_point 转为局部坐标
point_lcs_233 = lcs221.point_to_local_coordinate(point_gcs_233)
# 查看坐标
print(point_lcs_233)
# (0.0, 1.0, 2.0)


# 函数 point_to_global_coordinate(local_coordinate_point) 将局部坐标系表示的点 local_coordinate_point 转为全局坐标
print(lcs221.point_to_global_coordinate(point_lcs_233))
# (2.0, 3.0, 3.0)


# 函数 vector_to_local_coordinate() 和 vector_to_global_coordinate()
# 因为矢量具有平移不变性，所以和点的行为不同
# 全局坐标系和局部坐标系 lcs221 的转换中，矢量的坐标不变
vector_gcs_233 = P3(2,3,3)
vector_lcs_233 = lcs221.vector_to_local_coordinate(vector_gcs_233)
print(vector_gcs_233,vector_lcs_233)
# (2.0, 3.0, 3.0) (2.0, 3.0, 3.0)

vector_gcs_233 = lcs221.vector_to_global_coordinate(vector_lcs_233)
print(vector_gcs_233)
# (2.0, 3.0, 3.0)

# 函数 __str__() 和 __repr__() 将坐标系转为字符串
# 分别打印局部坐标系的原点、xyz三个轴方向在全局坐标系的坐标
# 下面三个打印结果相同
print(lcs221)
print(lcs221.__str__())
print(lcs221.__repr__())
# LOCATION=(2.0, 2.0, 1.0), xi=(1.0, 0.0, 0.0), yi=(0.0, 1.0, 0.0), zi=(0.0, 0.0, 1.0)

# 函数 __eq__() 判断局部两个坐标系是否相同。可以使用 == 符号自动调用
# 本质只对坐标原点和三个方向的相等判断
# 参数 err 指定绝对误差
# msg 如果指定，则判断结果为不相等时，抛出异常
lcs221_little_change = LocalCoordinateSystem(location=P3(2,2,1+1e-6))
print(lcs221==lcs221_little_change)
# True

# 类函数 create_by_y_and_z_direction() 由原点 location y方向 y_direction 和 z方向 z_direction 创建坐标系
lcs_created_by_y_and_z_direction = LocalCoordinateSystem.create_by_y_and_z_direction(
    location=P3(1,2,3),
    y_direction=P3.x_direct(),
    z_direction=P3.y_direct()
)
print(lcs_created_by_y_and_z_direction)
# LOCATION=(1.0, 2.0, 3.0), xi=(0.0, 0.0, 1.0), yi=(1.0, 0.0, 0.0), zi=(0.0, 1.0, 0.0)


# 类函数 global_coordinate_system() 获取全局坐标系，即 LOCATION=(0.0, 0.0, 0.0), xi=(1.0, 0.0, 0.0), yi=(0.0, 1.0, 0.0), zi=(0.0, 0.0, 1.0)
print(LocalCoordinateSystem.global_coordinate_system())
# LOCATION=(0.0, 0.0, 0.0), xi=(1.0, 0.0, 0.0), yi=(0.0, 1.0, 0.0), zi=(0.0, 0.0, 1.0)

# 函数 copy() 坐标系拷贝，拷贝后的坐标系和原坐标系无依赖关系
lcs221_copied = lcs221.copy()
lcs221_copied.location = P3(111,22,3)
print(lcs221)
print(lcs221_copied)
# LOCATION=(2.0, 2.0, 1.0), xi=(1.0, 0.0, 0.0), yi=(0.0, 1.0, 0.0), zi=(0.0, 0.0, 1.0)
# LOCATION=(111.0, 22.0, 3.0), xi=(1.0, 0.0, 0.0), yi=(0.0, 1.0, 0.0), zi=(0.0, 0.0, 1.0)


# 细节：
# 1. 创建坐标系时，传入的两个方向需要正交（垂直），若不正交则创建失败，会报错
try:
    lcs = LocalCoordinateSystem(x_direction=P3.x_direct(),z_direction=P3.x_direct())
except Exception as e:
    print("抓住异常：",e) 
    # 抓住异常： 创建 LocalCoordinateSystem 对象异常，x_direction(1.0, 0.0, 0.0)和z_direction(1.0, 0.0, 0.0)不正交

# 2. 创建坐标系时，传入的两个方向会自动归一化
lcs = LocalCoordinateSystem(x_direction=P3(x=2),z_direction=P3(z=3))
print(lcs)
# LOCATION=(0.0, 0.0, 0.0), xi=(1.0, 0.0, 0.0), yi=(0.0, 1.0, 0.0), zi=(0.0, 0.0, 1.0)






