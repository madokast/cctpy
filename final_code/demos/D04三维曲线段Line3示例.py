"""
CCT 建模优化代码
三维曲线段

作者：赵润晓
日期：2021年4月27日
"""

from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *
from hust_sc_gantry import *

# 包含 Line3、RightHandSideLine3、FunctionLine3、TwoPointLine3、DiscretePointLine3 五个类
# 其中 Line3 是抽向基类，表示任意三维有方向的曲线

# 首先以 FunctionLine3 的示例，说明 Line3 的使用方法
# 建立一个螺线，构建 FunctionLine3 的具体参数说明间 FunctionLine3 类介绍
solenoid = FunctionLine3(
    p3_function=lambda θ:P3(math.sin(θ),math.cos(θ),0.1*θ),
    start=0.0,end=math.pi*10,delta_for_disperse=0.1
)

# 绘图
# Plot3.plot_line3(solenoid)
# Plot3.show()

# get_length() 获取曲线上长度
print(solenoid.get_length())
# 31.55966148641606

# point_at(s) 获取曲线 s 位置的点
print(solenoid.point_at(0)) 
print(solenoid.point_at(math.pi*2)) 
# (0.0, 1.0, 0.0)
#(-0.028568649543929417, 0.9985741979771121, 0.6254569242029041)

# direct_at(s) 获取曲线 s 位置处曲线切向方向
print(solenoid.direct_at(0))
print(solenoid.direct_at(math.pi*2))
# (0.9937962006082295, -0.04959830638780113, 0.09954456117794518)
# (0.9937962006082294, 0.04959830638780137, 0.09954456117794515)

# right_hand_side_point(s,d,plane_direct)
# 和 Line2 的方法 right_hand_side_point 类似
# 只不过这里的右手侧，在矢量 plane_direct 确定的平面内进行
# 即 s 点处的方向 direct_at(s) 和右手侧共同确定的平面，与 plane_direct 垂直
# 一般情况下 plane_direct 是全局坐标系的 z 方向
print(solenoid.right_hand_side_point(0,1,P3.z_direct()))
# (-0.04984588566069717, 0.0012430787810775445, 0.0)

# left_hand_side_point(s,d,plane_direct)
# 左手侧，见 right_hand_side_point 的描述
print(solenoid.left_hand_side_point(0,1,P3.z_direct()))
# (0.04984588566069717, 1.9987569212189225, 0.0)

# right_hand_side_line3(d,plane_direct)
# 右手侧点组成的曲线
right_line3 = solenoid.right_hand_side_line3(0.1,P3.z_direct())
# 绘图
# Plot3.plot_line3(right_line3,describe='k-')
# Plot3.plot_line3(solenoid,describe='r-')
# Plot3.show()

# disperse3d(step) 轨迹离散成散点，即 p3 数组
p3_list = solenoid.disperse3d(step=math.pi/4)
# 绘图
# Plot3.plot_p3s(p3_list,describe='k.')
# Plot3.plot_line3(solenoid)
# Plot3.show()

# disperse3d_with_distance()
# 同方法 disperse3d，每个离散点带有距离，返回值是 ValueWithDistance[P3] 的数组
p3d_list = solenoid.disperse3d_with_distance(step=math.pi/4)
print(p3d_list[0])
print(p3d_list[1])
print(p3d_list[2])
# (0.0:(0.0, 1.0, 0.0))
# (0.769747841132099:(0.6926953830535866, 0.7197367967550444, 0.07662421106316544))
# (1.539495682264198:(0.9981130843103152, 0.038268830662832244, 0.1532484221263309))

# ------------------------------------- #

# 类 RightHandSideLine3 
# 是 Line3.right_hand_side_line3() 返回类型，视为 Line3 即可

# FunctionLine3 利用函数表达式确定的三维曲线
# 一般用于任意线圈的构建、CCT 洛伦兹力的分析等

# 构建方法 (p3_function,start,end,delta_for_disperse)
# p3_function 曲线方程  p = p(s)
# start 曲线起点对应的自变量 s 值
# end 曲线终点对应的自变量 s 值
# delta_for_disperse 离散步长，对应 start to end

# 下面构造一个螺旋三角函数
rot_sin = FunctionLine3(
    p3_function=lambda t:P3(
        x = t,
        y = math.sin(t)*math.sin(10*t),
        z = math.sin(t)*math.cos(10*t)
    ),
    start=0.0,
    end=4*math.pi,
    delta_for_disperse=0.01
)
# Plot3.plot_line3(rot_sin)
# Plot3.set_center(P3.x_direct(6),cube_size=12)
# Plot3.show()

# FunctionLine3 中特有的方法是 point_at_p3_function(s)
# 他返回的是函数 p3_function 在 s 处的值，而不是曲线 s 位置处的点
print(rot_sin.point_at_p3_function(math.pi/2))
# (1.5707963267948966, 6.123233995736766e-16, -1.0)


# TwoPointLine3
# 两点确定的三维直线段，p0 -> p1
two_point = TwoPointLine3(P3.origin(),P3(1,1,0))
# Plot3.plot_line3(two_point)
# Plot3.show()

# TwoPointLine3 类中没有特有函数

# DiscretePointLine3 表示通过离散点构建的三维曲线
# 一般离散点取自实际粒子轨迹
bl = HUST_SC_GANTRY().create_first_bending_part_beamline()
track = bl.track_ideal_particle(215,footstep=10*MM)
print(type(track))
print(type(track[0]))
traj_line3 = DiscretePointLine3(track)
# 绘图
# Plot3.plot_line3(traj_line3)
# Plot3.plot_beamline(bl,describes=['k-'])
# Plot3.set_center(P3(2,1),cube_size=4)
# Plot3.show()





