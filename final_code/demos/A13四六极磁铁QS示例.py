"""
CCT 建模优化代码
A13 四六极磁铁 QS 示例


作者：赵润晓
日期：2021年5月1日
"""

from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))

from cctpy import *

# QS 类表示硬边四六极磁铁
# 它是 Magnet 和 ApertureObject 的子类，所以父类的函数都可以使用


# 为了确定 QS 磁铁，需要确定以下参数
# 1. 四极场大小、六极场大小
# 2. 孔径
# 3. 位置、方向
# 其中位置和方向的确定最难，因此 QS 类提供两种方式构建 QS 磁铁
# 1. 构造器，通过指定 QS 所在局部坐标系确定其位置、方向
# 2. 依靠设计轨道 trajectory，将 QS 安置在轨道的某个位置上

# QS 构造器
# 由以下参数完全确定：
# length 磁铁长度 / m
# gradient 四极场梯度 / Tm-1
# second_gradient 六极场梯度 / Tm-2
# aperture_radius 孔径（半径） / m
# local_coordinate_system 局部坐标系
# 其中最难理解的是 local_coordinate_system
# 下面画图加以说明：
#
#      ③
#      ↑
#      |----------|
# -----①-------------->②
#      |----------|
#
# ① QS 磁铁入口中心位置，是局部坐标系的原心
# ② 理想粒子运动方向，是局部坐标系 Z 方向
# ③ 磁铁的 X 方向，由此可知垂直屏幕向外（向面部）是 Y 方向
# 注意：这个构造方式不推荐使用

# 下面构造一个 QS 磁铁的入口中心为全局坐标系的(1,0,0),轴向为全局坐标系 y 方向，磁铁的 x 方向全局坐标的 -x 方向的QS磁铁
# 磁铁长度 0.27 m，四极场梯度 5T/m，六极场梯度 0T/m，孔径 100 mm
qs = QS(
    local_coordinate_system=LocalCoordinateSystem(
        location=P3(1,0,0),
        x_direction=-P3.x_direct(),
        z_direction=P3.y_direct()
    ),
    length=0.27,
    gradient=5,
    second_gradient=0.,
    aperture_radius=100*MM
)

# 查看入口中心位置的磁场
print(qs.magnetic_field_at(P3(1,0,0))) # (0.0, 0.0, 0.0)
print(qs.magnetic_field_at(P3(0.95,0,0))) # (0.0, 0.0, 0.2500000000000002)
print(qs.magnetic_field_at(P3(0.9,0,0))) # (0.0, 0.0, 0.4999999999999999)
print(qs.magnetic_field_at(P3(0.8,0,0))) # (0.0, 0.0, 0.0 径向超出
print(qs.magnetic_field_at(P3(0.9,0.3,0))) # (0.0, 0.0, 0.0) 轴向超出



# 第二种创造 QS 磁铁的方法是类函数 create_qs_along()
# 创建一个位于设计轨道 trajectory 上的 qs 磁铁
# 参数如下：
# trajectory 二维设计轨道，因为轨道是二维的，处于 xy 平面，
#   这也限制了 qs 磁铁的轴在 xy 平面上
#   这样的限制影响不大，因为通常的束线设计，磁铁元件都会位于一个平面内
# s 确定 qs 磁铁位于设计轨道 trajectory 的位置，
#   即沿着轨迹出发 s 距离处是 qs 磁铁的入口，同时此时轨迹的切向为 qs 磁铁的轴向
# length qs 磁铁的长度
# gradient 四极场梯度 / Tm-1
# second_gradient 六极场梯度 / Tm-2
# aperture_radius 孔径（半径） / m
# 这个方法更常用

# 为了构建 qs 磁铁，首先构建一条设计轨道
# 轨道的起点为原点，起始方向为 x 正方向
# 轨迹上连续拼接了三条直线段，长度分别是 0.5m、0.27m、0.5m
traj = (
    Trajectory.set_start_point(P2.origin())
        .first_line(direct=P2.x_direct(),length=0.5)
        .add_strait_line(length=0.27)
        .add_strait_line(length=0.5)
)

# 在轨道 traj 的 0.5m 处，建立一个 0.27m 的 QS 磁铁
qs = QS.create_qs_along(
    trajectory=traj,
    s=0.5,
    length=0.27,
    gradient=20,
    second_gradient=0.0,
    aperture_radius=10*MM
)

# 不妨进行一次束流跟踪
# 建立一个理想粒子 ip，位于轨道 traj 起点，方向自然和 traj 平行
ip = ParticleFactory.create_proton_along(
    trajectory=traj,
    s=0,
    kinetic_MeV=215
)
# 建立分布于 x-xp 平面的相椭圆的粒子（相空间粒子） pps
pps = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_xxp_plane(
    xMax=3.5*MM,xpMax=7.5*MM,delta=0.0,number=16
)
# 将 pps 转为实际三维空间粒子
ps = ParticleFactory.create_from_phase_space_particles(
    ideal_particle=ip,
    coordinate_system=ip.get_natural_coordinate_system(y_direction=P3.z_direct()),
    phase_space_particles=pps
)
# 进行粒子跟踪获得轨迹
for p in ps:
    # 轨迹是 P3 的数组
    t = ParticleRunner.run_get_trajectory(
        p=p,m=qs,length=traj.get_length(),footstep=10*MM
    )
    # 取出 x 和 y，各自组成数组
    tx,ty,_ = P3.extract(t)
    # 绘图
    Plot2.plot(tx,ty,describe=None)

# 展示，去除注释查看绘图结果
Plot2.plot(qs)
Plot2.show()

# 函数 __str__()、__repr__() 将 qs 转为字符串信息，使用 print(qs) 自动执行
# 下面三个语句均打印：
# QS:local_coordinate_system=LOCATION=(0.5, 0.0, 0.0), xi=(6.123233995736766e-17, 1.0, 0.0), yi=(0.0, 0.0, 1.0), zi=(1.0, 0.0, 0.0), length=0.27, gradient=20.0, second_gradient=0.0, aperture_radius=0.1
print(qs)
print(qs.__str__())
print(qs.__repr__())