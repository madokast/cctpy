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
# 其中函数返回值类型都是 RunningParticle 或者 RunningParticle 数组
# 其中的函数都是类函数，不需要构建 ParticleFactory 的实例

# 类函数 create_proton(position,direct,kinetic_MeV)
# 通过确定质子的 位置、速度方向、动能 来创建一个质子
# 以下创建一个位于原点、朝 x 正方向运动的、250 MeV 的质子
p = ParticleFactory.create_proton(
    position=P3.origin(),
    direct=P3.x_direct(),
    kinetic_MeV=250
)
print(p.detailed_info())
# Particle[p=(0.0, 0.0, 0.0), v=(183955178.0274753, 0.0, 0.0)], rm=2.1182873748205775e-27, e=1.6021766208e-19, speed=183955178.0274753, distance=0.0


# 类函数 create_proton_by_position_and_velocity(position,velocity)
# 通过粒子位置、速度来创建一个质子
# # 以下创建一个位于 (1,2,3),速度 (1e8,1e7,0) 的质子 
p = ParticleFactory.create_proton_by_position_and_velocity(
    position=P3(1,2,3),
    velocity=P3(1e8,1e7,0)
)
print(p.detailed_info())
# Particle[p=(1.0, 2.0, 3.0), v=(100000000.0, 10000000.0, 0.0)], rm=1.775348694518523e-27, e=1.6021766208e-19, speed=100498756.21120891, distance=0.0]

# 类函数 create_proton_along(trajectory,s,kinetic_MeV)
# 创建一个位于轨迹 trajectory 的 s 位置处，运动方向和轨迹相切的，动能为 kinetic_MeV 的质子
# 这个方法非常实用，下面举例说明
# 首先创建一个轨迹，一条直线段，起点位于原点、x 正方向、长度 1 米
traj = (Trajectory
    .set_start_point(P2.origin())
    .first_line(direct=P2.x_direct(),length=1))
# 利用轨迹创建质子，质子位于轨迹 0.5m 位置处，能量 215 MeV
p = ParticleFactory.create_proton_along(traj,0.5,215)
print(p.detailed_info())
# Particle[p=(0.5, 0.0, 0.0), v=(174317774.94179922, 0.0, 0.0)], rm=2.0558942080656965e-27, e=1.6021766208e-19, speed=174317774.94179922, distance=0.0]

# 再举一个例子，将轨迹后接一个逆时针圆弧，半径 1 米，45 度
traj.add_arc_line(radius=1,clockwise=False,angle_deg=45)
# 然后创建位于轨迹末尾的质子，动能 200 MeV
p = ParticleFactory.create_proton_along(traj,traj.get_length(),200)
print(p)
# p=(1.7071067811865475, 0.2928932188134523, 0.0),v=(120017673.79145485, 120017673.7914548, 0.0),v0=169730622.00034538

# 类函数 create_from_phase_space_particle(ideal_particle,coordinate_system,phase_space_particle)
# 将相空间粒子 (x,xp,y,yp,z,δ) 转为三维实际坐标粒子 (位置,速度..)
# 很明显需要提供以下参数
# ideal_particle         理想粒子，提供参照，RunningParticle 类
# coordinate_system      相空间坐标系，提供参照，LocalCoordinateSystem 类
# phase_space_particle   相空间粒子，待转换，PhaseSpaceParticle 类
# 下面举例说明
# 第一步：创建理想粒子 ip，位于原点，速度为 x 正方向，动能 215 MeV
ip = ParticleFactory.create_proton(
    position=P3.origin(),
    direct=P3.x_direct(),
    kinetic_MeV=215
)
# 第二步：创建理想粒子 ip 所属的坐标系
# 这个坐标系的 z 方向是粒子速度方向，y 方向需要指定，一般选为全局坐标系的 z 方向
# x 方向由上面两个方向正交得到
cs = ip.get_natural_coordinate_system(y_direction=P3.z_direct())
# 第三步：创建一个相空间粒子 pp
pp = PhaseSpaceParticle(
    x=3.5*MM,xp=7.5*MRAD,
    y=0,yp=0,
    z=0,delta=0.1
)
# 最后将 pp 转为实际三维空间粒子 p
p = ParticleFactory.create_from_phase_space_particle(
    ideal_particle=ip,
    coordinate_system=cs,
    phase_space_particle=pp
)
print(ip) # p=(0.0, 0.0, 0.0),v=(174317774.94179922, 0.0, 0.0),v0=174317774.94179922
print(pp) # x=0.0035,xp=0.0075,y=0,yp=0,z=0,d=0.01
print(p) # p=(0.0, 0.0035, 0.0),v=(185279475.92100683, 1389596.069407551, 0.0),v0=185284686.83298966

# 再从 实际粒子 转回 相空间粒子坐标，核对一下
pp2 = PhaseSpaceParticle.create_from_running_particle(
    ideal_particle=ip,
    coordinate_system=cs,
    running_particle=p
)
print(pp2) # x=0.0035,xp=0.007500000000000001,y=0.0,yp=0.0,z=0.0,d=0.09999999999999996



# 类函数 create_from_phase_space_particles(ideal_particle,coordinate_system,phase_space_particles)
# 函数功能和上函数 create_from_phase_space_particle 一致
# 将多个相空间粒子 (x,xp,y,yp,z,δ) 转为三维实际坐标粒子 (位置,速度..)
# 主要用于质子束流的模拟

# 类函数 distributed_particles() 随机产生某种分布的质子集合，即 PhaseSpaceParticle 数组
# 仅支持正相椭圆/正相椭球分布，即不支持有倾斜角的相椭圆/相椭球
# 函数参数很多，介绍如下：
# x xp y yp delta，指定相椭圆/球参数，例如 3.5mm 7.5mr 3.5mm 7.5mr 0.08
# number 指定生成的粒子数目
# distribution_area 指定粒子的分布区域，有边缘分布 DISTRIBUTION_AREA_EDGE 和全分布 DISTRIBUTION_AREA_FULL 两种
#    边缘分布指的是，粒子位于相椭圆的圆周 或者 位于相椭球的表面
#    全分布指的是，粒子位于相椭圆的内部 或者 位于相椭球的内部
# *_distributed 是一个布尔量，指定变量是否参于分布。默认不参与
#   共有 x_distributed xp_distributed y_distributed yp_distributed delta_distributed 五个
#   如果取值 True 则参与分布，对应的变量，会产生相应的分布
#   如果取值 Fasle，则不参与分布，即产生的粒子该坐标都是定值
#   例如 x=xp=1，x_distributed=true，xp_distributed=false时，表示 x 参与分布，xp 不参与
#   即生成的粒子类似 (x=0.13,xp=1), (x=-0.79,xp=1), (x=0.45,xp=1)...
#   再例如 x=xp=y=yp=delta=1，且 x_distributed 和 xp_distributed 为 False， y_distributed，yp_distributed，delta_distributed 三个设为 true,
#   则生成的粒子类似 (x=1,xp=1,y=0.3,yp=-0.5,delta=0.1) ...
# distribution_type 表示分布类型，字符串类型，当前仅支持均匀分布 "uniform" 和 高斯分布 "gauss"
# 下面举例说明
# 束流参数为 x=y=3.5mm，xp,yp=7.5mr，dp=8%。生成粒子数目20
# 1. 生成x/xp相椭圆圆周上，动量分散为0的粒子
ps = ParticleFactory.distributed_particles(
    3.5*MM, 7.5*MRAD, 3.5*MM, 7.5*MRAD, 0.0, 20,
    ParticleFactory.DISTRIBUTION_AREA_EDGE,
    x_distributed=True, xp_distributed=True
)
# 2. 生成y/yp相椭圆内部，动量分散均为0.05的粒子
ps = ParticleFactory.distributed_particles(
    3.5*MM, 7.5*MRAD, 3.5*MM, 7.5*MRAD, 0.0, 20,
    ParticleFactory.DISTRIBUTION_AREA_FULL,
    y_distributed=True, yp_distributed=True
)
# 3. 生成 x/xp/delta 三维相椭球球面的粒子
ps = ParticleFactory.distributed_particles(
    3.5*MM, 7.5*MRAD, 3.5*MM, 7.5*MRAD, 0.08, 20,
    ParticleFactory.DISTRIBUTION_AREA_EDGE,
    x_distributed=True, xp_distributed=True, delta_distributed=True
)
