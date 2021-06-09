"""
CCT 建模优化代码
A12 磁铁相关类 Magnet 示例


作者：赵润晓
日期：2021年5月1日
"""

from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

# Magnet 是所有磁体的抽向基类/父类，表示一个磁铁，所谓磁铁，即在空间中能产生磁场分布
# 核心函数是 magnetic_field_at(point)，获得 point 位置处的磁场，磁场是三维矢量，因此返回值是 P3
# 因为 Magnet 是抽向类，无法实例化，使用它的子类 UniformMagnet 加以说明
# 下面构造了一个产生 z 方向 1.5T 磁场的磁铁
uniform_magnet = UniformMagnet(magnetic_field=P3.z_direct(1.5))
print(uniform_magnet.magnetic_field_at(P3.origin())) # (0.0, 0.0, 1.5)
print(uniform_magnet.magnetic_field_at(P3(1,1,1))) # (0.0, 0.0, 1.5)

# 函数 magnetic_field_along() 求二维曲线 line2 上的磁场分布
# line2    二维曲线
# p2_t0_p3 是一个函数，用于把 line2 上的二维点转为三维，默认转为 z=0 的三维点
# step     步长
# 返回值是 ValueWithDistance[P3] 的数组，即 位置:磁场大小
# 下面设定一条直线
straight_line = StraightLine2(length=1,direct=P2.x_direct(),start_point=P2.origin())
field_with_distances = uniform_magnet.magnetic_field_along(straight_line,step=0.5)
for fd in field_with_distances:
    print(fd)
# 输出如下
# (0.0:(0.0, 0.0, 1.5))
# (0.5:(0.0, 0.0, 1.5))
# (1.0:(0.0, 0.0, 1.5))

# 函数 magnetic_field_bz_along 求二维曲线 line2 上的 z 方向磁场分布
# 因为设计轨道常在 xy 平面，z 方向磁场即让粒子偏转的磁场，所以 z 方向磁场分布需要经常研究
# 参数和 magnetic_field_along() 一样
# 返回值是 P2，其中 x 表示曲线位置，y 表示 z 方向磁场
# （曲线是有方向的，所以位置 x 表示从曲线起点出发 x 距离所在点）
bz_with_distances = uniform_magnet.magnetic_field_bz_along(straight_line,step=0.25)
for bzd in bz_with_distances:
    print(bzd)
# 输出如下：
# (0.0, 1.5)
# (0.25, 1.5)
# (0.5, 1.5)
# (0.75, 1.5)
# (1.0, 1.5)


# 函数 graident_field_along() 求二维曲线 line2 上的 z 方向的梯度
# 因为设计轨道常在 xy 平面，z 方向磁场即让粒子偏转的磁场，所以 z 方向磁场分布需要经常研究
# 参数和 magnetic_field_bz_along() 一样，多出 point_number 和 good_field_area_width
# good_field_area_width 求梯度需要涉及横向范围，这个参数确定水平垂线的长度，注意应小于等于好场区范围。
# point_number 水平垂线上取点数目，默认 4，越多则拟合越精确。因为求梯度是采用多项式拟合，取点越多拟合结果越精确，同时计算量越复杂
# 另外函数传入的二维曲线 line2 看作 z=0 的三维曲线

# 用 QS 磁铁举例。创建一个沿着 straight_line 的 QS
# qs 的起点位于 straight_line 的 0.1 m 位置，长度 0.8 m，梯度 10 T/m，二阶梯度 100 T/m2，半孔径 60 mm
qs = QS.create_qs_along(
    straight_line,
    s=0.1,
    length=0.8,
    gradient=10,
    second_gradient=100,
    aperture_radius=60*MM
)
graident_fields = qs.graident_field_along(straight_line,step=0.2)
print(graident_fields[0]) # (0.0, 0.0) 表示直线 straight_line 0.0 位置的梯度为 0.0
print(graident_fields[1]) # (0.2, 9.999999999999995) 表示直线 straight_line 0.2 位置的梯度为 10


# 函数 second_graident_field_along() 求二维曲线 line2 上的 z 方向的二阶梯度
# 参数和返回值类型和 graident_field_along() 一致
# 注意这里的二阶梯度，是多项式拟合后，二次项 * 2
second_graident_fields = qs.second_graident_field_along(straight_line,step=0.2)
print(second_graident_fields[0]) # (0.0, 0.0) 表示直线 straight_line 0.0 位置的二阶梯度为 0.0
print(second_graident_fields[1]) # (0.2, 99.99999999999922) 表示直线 straight_line 0.2 位置的梯度为 100

# 函数 multipole_field_along() 求二维曲线 line2 上的 z 方向的各极谐波分布
# 参数和 graident_field_along() 一致
# 多出的参数 order，表示多极谐波阶数。
# 返回值为 List[ValueWithDistance[List[float]]]，
# 是 ValueWithDistance 对象的一个数组，
# 表示沿着 line2 分布的各极谐波，
# 各极谐波采用数组表示，数组下标0表示二级场，
# 1表示四极场，2表示六极场，以此类推
# 另外 point_number 水平垂向取点数目，
# 只有取点数目多于 order，才能正确拟合高次谐波
# 注意这里的i阶梯度，是多项式拟合后，还会乘上 i!
multipole_fields = qs.multipole_field_along(
    line2=straight_line,
    order=5,
    good_field_area_width=15*MM,
    step=0.2*M,
    point_number=10
)
print(multipole_fields[0]) 
#(0.0:[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 表示直线 straight_line 0.0 位置的各极梯度
print(multipole_fields[1]) 
# (0.2:[6.197084996143446e-17, 10.000000000000005, 99.99999999999858, 2.4944462259294195e-10, 4.17706982954073e-07, -0.0004385844231670395]) 表示直线 straight_line 0.2 位置的各极梯度

# 函数 multipole_field_along_line3()
# 和 multipole_field_along() 一样，仅仅是二维曲线 line2 换成了 line3
# line3 三维曲线
# order 谐波阶数，0阶求二级场，1阶求二四极场，2阶求二四六极场，以此类推
# good_field_area_width 好场区范围，在水平垂向取点时，从这个范围选取，默认值 10 mm
# step 三维曲线 line2 步长，步长越多，返回的数组长度越长，默认值 1 mm
# point_number 水平垂向取点数目，只有取点数目多于 order，才能正确拟合高次谐波，默认值 10
# 新增参数 plane_direct，确定磁场计算时的中平面，即中平面垂直于 plane_direct，默认值 P3.z_direct()
# 返回值为 List[ValueWithDistance[List[float]]]，是 ValueWithDistance 对象的一个数组，表示沿着 line2 分布的各极谐波，各极谐波采用数组表示

# 创建一个三维曲线，这里创建两个三维点组成的直线段
line3 = TwoPointLine3(P3.origin(),P3(1,20*MM,30*MM))
# 计算 line3 上的多极场分布
multipole_fields = qs.multipole_field_along_line3(
    line3=line3,
    order=2,
    step=0.2*M,
)
print(multipole_fields[0])
print(multipole_fields[1])
# (0.0:[0.0, 0.0, 0.0])
# (0.20012995777744022:[0.039000000000000035, 10.397920623792068, 99.96001599360682])

line3 = TwoPointLine3(P3.origin(),P3(1,20*MM,30*MM))
multipole_fields = qs.multipole_field_along_line3(
    line3=line3,
    order=2,
    step=0.2*M,
    plane_direct=P3(-1,-1,2)
)
print(multipole_fields[0])
print(multipole_fields[1])
# (0.0:[0.0, 0.0, 0.0])
# (0.20012995777744022:[0.03900000000000005, 9.100500737879985, 62.138728323703226])



# 函数 integration_field() 计算二维曲线 line2 上的积分场
# 参数如下：
# line2     二维曲线
# p2_t0_p3 是一个函数，用于把 line2 上的二维点转为三维，默认转为 z=0 的三维点
# step     取点和积分的步长
integration_field_1mm = uniform_magnet.integration_field(straight_line,step=1*MM)
integration_field_10mm = uniform_magnet.integration_field(straight_line,step=10*MM)
print(integration_field_1mm) # 1.5000000000000013
print(integration_field_10mm) # 1.4999999999999976




# 类函数 no_magnet() 返回一个不产生磁场的磁铁
no_magnet = Magnet.no_magnet()
print(no_magnet.magnetic_field_at(P3.random())) 
# (0.0, 0.0, 0.0)

# 类函数 uniform_magnet() 返回一个产生恒定磁场的磁铁
# 参数传入磁场大小，是一个 P3 对象
uniform_magnet = Magnet.uniform_magnet(P3(1,2,3))
print(uniform_magnet.magnetic_field_at(P3.random())) 
# (1.0, 2.0, 3.0)

# 类函数 combine() 完成多个磁场的组合
# 参数数目不定，可以传入任意个磁场
combine_magnets = Magnet.combine(uniform_magnet,qs,uniform_magnet)
print(combine_magnets.magnetic_field_at(P3.random()))
# (2.0, 4.0, 6.0)

# UniformMagnet 表示产生均匀磁场的磁铁
# 有两种构造方式
# 1. 直接使用构造器
uniform_magnet = UniformMagnet(P3(1,2,3))
# 2. 使用  Magnet.uniform_magnet() 创建
uniform_magnet = Magnet.uniform_magnet(P3(1,2,3))


# LocalUniformMagnet 表示在一个圆柱区域内产生均匀磁场的磁铁
# 可以看作一个直线二极磁铁

# 构造器 LocalUniformMagnet()，参数如下
# local_coordinate_system 局部坐标系
#     局部坐标系的原点即二极磁铁的起点
#     局部坐标系的 z 方向即二极磁铁延申方向
#     局部坐标系的 y 方向即磁场方向
# length 磁铁长度
# aperture_radius 磁铁孔径，磁铁外部磁场为零
# magnetic_field 磁场大小，标量。磁场方向由 local_coordinate_system 确定
# 注意：不推荐使用构造器创建此对象
# 应该使用 create_local_uniform_magnet_along() 函数
lum = LocalUniformMagnet(
    local_coordinate_system=LocalCoordinateSystem.global_coordinate_system(),
    length=1.0,
    aperture_radius=100*MM,
    magnetic_field=1.23
)
print(lum.magnetic_field_at(P3(0,0,0.5)))
# (0.0, 1.23, 0.0)

# 函数 create_local_uniform_magnet_along() 创建一个沿着设计轨道的 LocalUniformMagnet，参数如下
#    trajectory 设计轨道
#    s 磁铁起点位于设计轨道上的位置
#    magnetic_field 磁场大小，标量，磁场垂直于设计轨道
#    length 磁铁长度
#    aperture_radius 磁铁孔径
# 首先建立设计轨道
traj = (Trajectory
        .set_start_point(P2.origin())
        .first_line(direct=P2.x_direct(),length=1.0)
        )
# 根据设计轨道建立 lum
lum = LocalUniformMagnet.create_local_uniform_magnet_along(
    trajectory=traj,
    s=0.5,
    magnetic_field=1.23,
    length=0.2,
    aperture_radius=100*MM
)
# 绘图
# Plot2.plot(traj,lum)
# Plot2.show()



# ApertureObject 表示一个具有孔径的元件，这是一个抽向类
# 可以判断点 point 是在这个对象的孔径内还是孔径外
# 只有当粒子轴向投影在元件内部时，才会进行判断，
# 否则即时粒子距离轴线很远，也认为粒子没有超出孔径，
# 这是因为粒子不在元件内时，很可能处于另一个大孔径元件中，这样会造成误判。

# ApertureObject 类只有一个函数 is_out_of_aperture(point)
# 判断点 point 是否在孔径内部
# 这个方法在 ApertureObject 中没有实现
# 使用 ApertureObject 的子类 QS 举例
# 创建一个 qs 磁铁，位于全局坐标系
# 则 qs 磁铁入口中心位置，在坐标系的圆心
# qs 磁铁轴线方向，和坐标系的 z 方向平行
qs = QS(
    local_coordinate_system=LocalCoordinateSystem.global_coordinate_system(),
    length=1.,
    gradient=0.0,
    second_gradient=0.0,
    aperture_radius=100*MM
)
print(qs.is_out_of_aperture(P3.origin())) # False
print(qs.is_out_of_aperture(P3(x=80*MM))) # False
print(qs.is_out_of_aperture(P3(x=120*MM))) # True
print(qs.is_out_of_aperture(P3(x=80*MM,z=0.5))) # False
print(qs.is_out_of_aperture(P3(x=80*MM,z=1.5))) # False


# CombinedMagnet可以实现多个磁铁组合
# 构造函数，传入参数为 *magnets，意味着参数数目不定，可以传入任意多个磁铁
um1 = UniformMagnet(magnetic_field=P3(1,0,0))
um2 = UniformMagnet(magnetic_field=P3(0,2,0))
um3 = UniformMagnet(magnetic_field=P3(0,0,3))
cm = CombinedMagnet(um1,um2,um3)
print(cm.magnetic_field_at(P3.random()))
# (1.0, 2.0, 3.0)

# 函数 add 添加一个磁铁
um4 = UniformMagnet(magnetic_field=P3(1,1,1))
cm.add(um4)
print(cm.magnetic_field_at(P3.random()))
# (2.0, 3.0, 4.0)

# 函数 get_magnets 暴露内部磁铁
magnets = cm.get_magnets()
for m in magnets:
    print(m)
# 均匀磁铁，磁场(1.0, 0.0, 0.0)
# 均匀磁铁，磁场(0.0, 2.0, 0.0)
# 均匀磁铁，磁场(0.0, 0.0, 3.0)
# 均匀磁铁，磁场(1.0, 1.0, 1.0)

# 函数 remove 移除一个磁铁
cm.remove(um1)
print(cm.magnetic_field_at(P3.random()))
# (1.0, 3.0, 4.0)
