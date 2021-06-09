"""
CCT 建模优化代码
A08 描述全局坐标系中的一个粒子 RunningParticle 示例

注意：在实际建模、仿真时，不要使用 RunningParticle 内部的函数
应使用 ParticleFactory 来创建粒子
使用 ParticleRunner 让粒子/束流在磁场中运动
使用 PhaseSpaceParticle 完成实际三维粒子和相空间粒子的转换

作者：赵润晓
日期：2021年4月29日
"""

from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

# RunningParticle 类描述全局坐标系中的一个粒子

# 构造函数 RunningParticle()
# 参数为
# position   粒子位置，三维矢量，单位 [m, m, m]
# velocity   粒子速度，三位矢量，单位 [m/s, m/s, m/s]
# relativistic_mass 粒子相对论质量，又称为动质量，单位 kg， M=Mo/√(1-v^2/c^2)
# e          粒子电荷量，单位 C 库伦
# speed      粒子速率，单位 m/s
# distance   粒子运动距离，单位 m
# 下面随机创建一个粒子
running_particle_random = RunningParticle(
    position=P3.random(),
    velocity=P3.random().change_length(1),
    relativistic_mass=1,
    e=1,
    speed=1,
    distance=0
) # p=(0.3884444567695907, 0.22920931563733715, 0.5058699785343335),v=(0.6330208618895012, 0.6882505364810616, 0.35439495967945656),v0=1
print(running_particle_random)
# 注意：不应使用这个构造器来创建粒子，因为内部不检查 velocity 和 speed 的一致性
# 应该使用  ParticleFactory 来创建粒子，ParticleFactory 提供了丰富了创建粒子、粒子束的函数


# 函数 run_self_in_magnetic_field() 
# 此函数已经废弃，因为没有使用 Runge-Kutta 数值积分方法，误差过大
# 让粒子在恒强磁场中运动一小段距离
# 参数如下
# magnetic_field 磁场，看作恒定场，三维矢量 P3
# footstep 步长，默认 1 MM
# 被弃用，不应使用


# 函数 copy() 复制一个粒子，复制的粒子和原粒子没有依赖
# 即修改复制的粒子，不会改变原粒子
# 这个方法常用于复制多个理想粒子，然后分别修改参数，模拟束流
running_particle_random_copied = running_particle_random.copy()
running_particle_random_copied.speed=100
print(running_particle_random) # p=(0.12598114034167662, 0.4077843353758793, 0.15218549925472213),v=(0.7348323133200128, 0.4521171494230116, 0.5055804134837196),v0=1
print(running_particle_random_copied) # p=(0.12598114034167662, 0.4077843353758793, 0.15218549925472213),v=(0.7348323133200128, 0.4521171494230116, 0.5055804134837196),v0=100
# 可以看到，可以随意修改粒子速率，但是粒子速度不改变，这将导致不一致，因此一般情况下不应使用这些底层函数 

# 函数 compute_scalar_momentum()，计算粒子动量，即 speed * relativistic_mass
# 返回动量单位是 kg m/s
# 这个方法用于动量分散相关的计算
print(running_particle_random.compute_scalar_momentum()) # 1

# 函数 change_scalar_momentum(scalar_momentum)，改变粒子的标量动量
# 注意：真正改变的是粒子的速度和动质量
# 这个方法用于生成一组动量分散的粒子
# scalar_momentum 的单位是 kg m/s
running_particle_random.change_scalar_momentum(2)
print(running_particle_random) 
# p=(0.33316296168838744, 0.7938580738663331, 0.34008909721137903),v=(1.2025435401611133, 0.3651129458584246, 1.5558218313108225),v0=2.0

# 函数 get_natural_coordinate_system(y_direction)
# 获得粒子的自然坐标系，其中 z 轴方向即粒子速度方向，另外指定 y 轴方向
# x 轴方向由 y 轴和 z 轴方向确定
# 这个方法只用于相空间和实际三维空间转换
# 理想粒子的 natural_coordinate_system 即用来确定其他粒子的相空间坐标
running_particle_random.velocity = P3(1,1,0)
print(running_particle_random.get_natural_coordinate_system(y_direction=P3.z_direct())) 
# LOCATION=(0.05578965908696243, 0.9024614358976661, 0.17686810290473776), xi=(-0.7071067811865475, 0.7071067811865475, 0.0), yi=(0.0, -0.0, 0.9999999999999998), zi=(0.7071067811865475, 0.7071067811865475, 0.0)

# 函数 __str__() 和 __repr__()，将粒子转为字符串，调用 print() 函数时自动执行
# 仅返回粒子坐标、速度、速率
# 以下三行代码均返回
# p=(0.6257147199949195, 0.870379987066165, 0.140742191715013),v=(1.0, 1.0, 0.0),v0=2.0
# 注意：因为随意修改了粒子的速度和速率，所以两者不兼容
# 因此在实际使用 RunningParticle 时，不应直接调用本类的函数
print(running_particle_random)
print(running_particle_random.__str__())
print(running_particle_random.__repr__())

# 函数 to_numpy_array_data() 和 from_numpy_array_data(array)
# 完成粒子和 numpy 数组的转换
# 函数用于 GPU 加速粒子仿真
# numpy_array_data 是一个一维数组，长度 10，参数分别是
# px py pz vx vy vz    rm      e       speed   distance
#     位置     速度     动质量  电荷量   速率    运动距离
array = running_particle_random.to_numpy_array_data()
print(array)
# 输出如下
# [0.36754438 0.70788194 0.61466121 1.         1.         0.
#  1.         1.         2.         0.        ]
rp = RunningParticle.from_numpy_array_data(array)
print(rp) # p=(0.9720221164124271, 0.45095448716158015, 0.8361092775713064),v=(1.0, 1.0, 0.0),v0=2.0


# 函数 populate(other) 将粒子 other 的全部参数复制到调用者
# 复制双方无依赖
rp = RunningParticle(P3.origin(),P3.zeros(),0,0,0,0)
print(rp) 
# p=(0.0, 0.0, 0.0),v=(0.0, 0.0, 0.0),v0=0
rp.populate(running_particle_random)
print(rp) 
# p=(0.7590848416287488, 0.29220215565669594, 0.4094821302260654),v=(1.0, 1.0, 0.0),v0=2.0

# 函数 __sub__() 粒子"减法" 只用来显示两个粒子的差异
# 一般用于调试
print(running_particle_random-running_particle_random_copied)
# p=(0.0, 0.0, 0.0),v=(0.30413818368563283, 0.615853379245629, -0.6068012082706059),v0=-98.0

# 2021年5月1日 新增
# 函数 detailed_info() 获取粒子全部信息，返回字符串
print(running_particle_random.detailed_info())
# Particle[p=(0.40765062815998554, 0.0707454582037097, 0.15717268191231237), v=(1.0, 1.0, 0.0)], rm=1.0, e=1, speed=2.0, distance=0]