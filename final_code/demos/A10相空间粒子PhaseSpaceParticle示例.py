"""
CCT 建模优化代码
A10 相空间粒子 PhaseSpaceParticle 示例


作者：赵润晓
日期：2021年4月29日
"""

from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

# PhaseSpaceParticle 类以相空间坐标的形式表示粒子，即 x xp y yp z δ
# 利用理想粒子作为参照，可以快速地实现粒子的三维坐标到相空间坐标的转换
# 即 RunningParticle 和 PhaseSpaceParticle 之间的转换

# 类常量 XXP_PLANE 和 YYP_PLANE，分别表示 x-xp 平面和 y-yp 平面
# 常用于构建分布于某一平面的粒子，或者将 6 维相空间投影到某一平面

# PhaseSpaceParticle 的构造方法即指定 6 个坐标
pp = PhaseSpaceParticle(
    x=3.5*MM,xp=7.5*MRAD,
    y=2.5*MM,yp=5.5*MRAD,
    z=0.0,delta=0.05
)
print(pp) # x=0.0035,xp=0.0075,y=0.0025,yp=0.0055,z=0.0,d=0.05

# 函数 project_to_xxp_plane() 和 project_to_yyp_plane() 将相空间映射到 x-xp 平面和 y-yp 平面
# 可以传入 convert_to_mm 布尔值，修改单位，从 米-弧度 到 毫米-毫弧度
# 返回值是 P2
print("映射到 x-xp 平面",pp.project_to_xxp_plane())
print("映射到 y-yp 平面",pp.project_to_yyp_plane(convert_to_mm=True))
# 映射到 x-xp 平面 (0.0035, 0.0075)
# 映射到 y-yp 平面 (2.5, 5.5)

# 函数 project_to_plane() 将相空间映射到 x-xp 平面和 y-yp 平面
# 参数如下：
# plane_id 取 XXP_PLANE 和 YYP_PLANE，实现具体映射平面
# convert_to_mm 布尔值，修改单位，从 米-弧度 到 毫米-毫弧度
# 返回值是 P2
print("映射到 x-xp 平面",pp.project_to_plane(PhaseSpaceParticle.XXP_PLANE,convert_to_mm=True))
print("映射到 y-yp 平面",pp.project_to_plane(PhaseSpaceParticle.XXP_PLANE))
# 映射到 x-xp 平面 (3.5, 7.5)
# 映射到 y-yp 平面 (0.0035, 0.0075)

# 类函数 phase_space_particles_along_positive_ellipse_in_xxp_plane()
# 获取均匀分布于 x-xp 平面上正相椭圆上的多个粒子 PhaseSpaceParticles
# 参数如下
# xMax 相椭圆参数 x 最大值
# xpMax 相椭圆参数 xp 最大值
# delta 动量分散
# number 粒子数目
pps_xxp = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_xxp_plane(
    xMax=3.5*MM,xpMax=7.5*MRAD,delta=0.1,number=4
)
for pp in pps_xxp:
    print(pp)
# x=0.0035,xp=0,y=0,yp=0,z=0,d=0.1
# x=-6.544974912410392e-06,xp=0.007499986886714065,y=0,yp=0,z=0,d=0.1
# x=-0.003499999709767181,xp=-3.0543267124762926e-06,y=0,yp=0,z=0,d=0.1
# x=6.544974910946571e-06,xp=-0.007499986886714071,y=0,yp=0,z=0,d=0.1

# 类函数 phase_space_particles_along_positive_ellipse_in_yyp_plane()
# 获取均匀分布于 y-yp 平面上正相椭圆上的多个粒子 PhaseSpaceParticles
# 参数如下
# yMax 相椭圆参数 y 最大值
# ypMax 相椭圆参数 yp 最大值
# delta 动量分散
# number 粒子数目
pps_yyp = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_yyp_plane(
    yMax=3.5*MM,ypMax=7.5*MRAD,delta=0.1,number=32
)
# 去除注释查看绘图效果
# for pp in pps_yyp:
#     plt.plot(pp.y,pp.yp,"r.")
# plt.show()

# 类函数 phase_space_particles_along_positive_ellipse_in_plane()
# 和上两个函数类似，仅仅需要多传入一个参数 plane_id 确定生成的椭圆分布所属平面
# plane_id 取 XXP_PLANE 和 YYP_PLANE，确定平面
pps_xxp2 = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_plane(
    xMax=3.5*MM,xpMax=7.5*MRAD,delta=0.1,number=4,plane_id=PhaseSpaceParticle.XXP_PLANE
)
for pp in pps_xxp2:
    print(pp)
# x=0.0035,xp=0,y=0,yp=0,z=0,d=0.1
# x=-6.544974912410392e-06,xp=0.007499986886714065,y=0,yp=0,z=0,d=0.1
# x=-0.003499999709767181,xp=-3.0543267124762926e-06,y=0,yp=0,z=0,d=0.1
# x=6.544974910946571e-06,xp=-0.007499986886714071,y=0,yp=0,z=0,d=0.1


# 类函数 phase_space_particles_project_to_xxp_plane()
# 类函数 phase_space_particles_project_to_yyp_plane()
# 类函数 phase_space_particles_project_to_plane()
# 都是讲多个相空间粒子，映射到 x-xp / y-yp 平面，返回值是 P2 的数组
# 三个函数都可以传入布尔值 convert_to_mm，转换单位从 米-弧度 到 毫米-毫弧度
# 最后一个函数需要指定 plane_id 取 XXP_PLANE 和 YYP_PLANE，确定平面
p2_xxp = PhaseSpaceParticle.phase_space_particles_project_to_xxp_plane(pps_xxp,convert_to_mm=True)
for p2 in p2_xxp:
    print(p2)
# (3.5, 0.0)
# (-0.006544974912410392, 7.4999868867140655)
# (-3.499999709767181, -0.003054326712476293)
# (0.006544974910946571, -7.499986886714071

# 类函数 create_from_running_particle()
# 类函数 create_from_running_particles()
# 计算三维实际粒子 RunningParticle 的像空间坐标，返回 PhaseSpaceParticle
# 第一个函数将一个 RunningParticle 转为 PhaseSpaceParticle
# 第二个函数将 RunningParticle 数组转为 PhaseSpaceParticle 数组
# 根据束流光学的知识，还需要传入以下两个参数：
# ideal_particle    理想粒子，理想粒子位于相空间原点，是坐标变换的核心
# coordinate_system 理想粒子所在坐标系，确定纵向 z 方向，横向 x 和 y 方向
# 下面演示 create_from_running_particle() 函数的使用
# 创建一个动能 250MeV、位于原点、沿着 x 运动的理想粒子 ip
ip = ParticleFactory.create_proton(
    position=P3.origin(),
    direct=P3.x_direct(),
    kinetic_MeV=250
)
# 利用理想粒子构建坐标系，这里指定横向运动中 y 方向是全局坐标系的 z 方向（很多情况下都是如此）
cs = ip.get_natural_coordinate_system(y_direction=P3.z_direct())
# 创建一个动能 240MeV、位于(0,1,0)、沿着(100,-1,0)运动的粒子 p
p = ParticleFactory.create_proton(
    position=P3(0,1,0),
    direct=P3(100,-1,0),
    kinetic_MeV=240
)
# 求 p 的相空间坐标
p_pp = PhaseSpaceParticle.create_from_running_particle(
    ideal_particle=ip,
    coordinate_system=cs,
    running_particle=p
)
print(p_pp) # x=1.0,xp=-0.00985736125798635,y=0.0,yp=0.0,z=0.0,d=-0.02251054564650803


# 函数 convert_delta_from_momentum_dispersion_to_energy_dispersion()
# 函数 convert_delta_from_energy_dispersion_to_momentum_dispersion()
# 函数 convert_delta_from_momentum_dispersion_to_energy_dispersion_for_list()
# 函数 convert_delta_from_energy_dispersion_to_momentum_dispersion_for_list()
# 可以转换相空间坐标中的 delta/δ
# 完成 能量分散 和 动量分散 之间的转换
# 其中前两个方法面向单个粒子，即一个 PhaseSpaceParticle
# 其中后两个方法面向多个粒子，即 PhaseSpaceParticle 数组
# 为了完成转换，还需要输入中心动能 centerKineticEnergy_MeV，单位 MeV

# 下面举例
# 创建一个动量分散为 0.1 的粒子
pp = PhaseSpaceParticle(delta=0.1)
# 转为能量分散，设中心动能为 250 MeV
pp = PhaseSpaceParticle.convert_delta_from_momentum_dispersion_to_energy_dispersion(pp,250)
print(pp) # x=0.0,xp=0.0,y=0.0,yp=0.0,z=0.0,d=0.17896104739526544

# 创建一个能量分散为 0.1 的粒子
pp = PhaseSpaceParticle(delta=0.1)
# 转为动量分散，设中心动能为 250 MeV
pp = PhaseSpaceParticle.convert_delta_from_energy_dispersion_to_momentum_dispersion(pp,250)
print(pp) # x=0.0,xp=0.0,y=0.0,yp=0.0,z=0.0,d=0.055878081546501715

# 函数 __str__() 和 __repr__()，将 PhaseSpaceParticle 转为字符粗，在调用 print() 时自动执行
# 下面三行代码都打印
# x=0.0,xp=0.0,y=0.0,yp=0.0,z=0.0,d=0.055878081546501715
print(pp)
print(pp.__str__())
print(pp.__repr__())

# 函数 copy() 实现 PhaseSpaceParticle 的拷贝
# 拷贝结果和原对象无依赖关系

# 函数 getDelta() 得到粒子的 动量/能量分散，即 p.delta
# 写成函数是为了面向对象，不操作变量
