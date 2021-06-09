"""
CCT 建模优化代码
A14 CCT磁铁示例


作者：赵润晓
日期：2021年5月1日
"""

from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))

from cctpy import *

# CCT 类表示一层连续的单线模型弯曲 CCT 磁铁
# 它是 Magnet 和 ApertureObject 的子类，所以父类的函数都可以使用

# 如果要会使用 CCT 类构建弯曲 CCT 磁铁，需要首先明白弯曲 CCT 线圈路径的二维曲线形式和三维曲线形式
# 只有先明确 CCT 线圈路径的二维曲线形式，才能方便、不易犯错的构建三维 CCT 磁铁

# 构建 CCT 有两种方法，一种是构造函数 CCT()
# 另一种是依靠设计轨道创建，CCT.create_cct_along()
# 第二种方法更简单

# 构造函数 CCT()，需要传入以下参数
# local_coordinate_system    CCT 局部坐标系
#    cct 的局部坐标系以偏转弧线段圆心为原点，
#    cct 绕线起点在 x 轴上，仅限于起点为(ξ=0,φ=0)时，如果不是则按实际映射
#    绕线时轴向朝 y 轴正方向运动，仅限于终点 φ > 起点 φ 时，如果不是，则朝 y 轴负方向
# big_r    大半径：偏转半径
# small_r  小半径（孔径的一半）
# bending_angle 偏转角度，角度制，必须是正数。
#    反向绕线的 CCT 按照 starting_point_in_ksi_phi_coordinate 和 end_point_in_ksi_phi_coordinate 确定。典型值 67.5
# tilt_angles 各极倾斜角，任意长度数组。典型值 [30,90,90,90]
# winding_number  匝数
# current  电流，电流方向按照 starting_point_in_ksi_phi_coordinate 到 end_point_in_ksi_phi_coordinate。负数则反向
# starting_point_in_ksi_phi_coordinate  CCT 起点，以 ksi_phi_coordinate 坐标系表示，类型为 P2
# end_point_in_ksi_phi_coordinate       CCT 终点，以 ksi_phi_coordinate 坐标系表示，类型为 P2
# disperse_number_per_winding    每匝线圈离散电流元数目，数字越大计算精度越高，默认 120

# 简单构造一个 30 度 30 匝的 CCT，就位于全局坐标系
cct = CCT(
    local_coordinate_system=LocalCoordinateSystem.global_coordinate_system(),
    big_r=1*M,
    small_r=30*MM,
    bending_angle=30,
    tilt_angles=[30],
    winding_number=30,
    current=1000,
    starting_point_in_ksi_phi_coordinate=P2(0,0),
    end_point_in_ksi_phi_coordinate=P2(2*30*math.pi,30/180*math.pi)
)
# 绘制 cct 三维分布
Plot3.plot_cct(cct)
Plot3.show()

# 绘制 ksi_phi_coordinate cct 二维曲线
Plot2.plot_cct_path2d(cct)
Plot2.show()

# 函数 phi_ksi_function(ksi) 完成 cct ksi->phi 的映射
print(cct.phi_ksi_function(2*math.pi)) # 0.01745329251994328
print(1/180*math.pi) # 0.017453292519943295 


# 内部类 BipolarToroidalCoordinateSystem 双极点坐标系
# 主要意义在于完成 CCT 二维曲线 (ξ,φ) 转为三维 (x,y,z)

# BipolarToroidalCoordinateSystem() 构造函数
# 参数为
# a       极点
# eta     η0
# big_r   
# small_r

# BipolarToroidalCoordinateSystem.convert(p2)
# CCT 二维曲线 (ξ,φ) 转为三维 (x,y,z)
# 注意：三维点 (x,y,z) 处于 CCT 的局部坐标系
print(cct.bipolar_toroidal_coordinate_system.convert(P2.origin())) # (1.030000000000002, 0.0, 0.0)

# BipolarToroidalCoordinateSystem.main_normal_direction_at(p2)
# 返回三维 CCT 曲线的法向量，位置由二维点 (ξ,φ) 确定，二维点会在内部转为三维点 (x,y,z)
# 即返回值 P3 在这点 (x,y,z) 垂直于圆环面
# 注意：法向量处于 CCT 的局部坐标系
print(cct.bipolar_toroidal_coordinate_system.main_normal_direction_at(P2.origin())) # (1.0, 0.0, 0.0)


# BipolarToroidalCoordinateSystem.__str__()
# BipolarToroidalCoordinateSystem.__repr__() 完成双极点坐标系到字符串转变
# 调用 print() 自动执行
print(cct.bipolar_toroidal_coordinate_system) # BipolarToroidalCoordinateSystem a(0.9995498987044118)eta(4.199480001904374)R(1.0)r(0.03)



# magnetic_field_at() 见 Magnet

# is_out_of_aperture() 见 ApertureObject


# 函数 __str__() 和 __repr__() 完成 CCT 到字符串转变
print(cct) # CCT: local_coordinate_system(LOCATION=(0.0, 0.0, 0.0), xi=(1.0, 0.0, 0.0), yi=(0.0, 1.0, 0.0), zi=(0.0, 0.0, 1.0))big_r(1.0)small_r(0.03)bending_angle(30.0)tilt_angles([30.0])winding_number(30)current(1000.0)starting_point_in_ksi_phi_coordinate((0.0, 0.0))end_point_in_ksi_phi_coordinate((188.49555921538757, 0.5235987755982988))disperse_number_per_winding(120)


# 类函数 create_cct_along() 创建一个沿着设计轨道的 CCT
# 参数如下：
# trajectory   设计轨道，创建的 cct 将位于轨道上
# s             cct起点所处 设计轨道的位置
# big_r    大半径：偏转半径
# small_r  小半径（孔径的一半）
# bending_angle 偏转角度，角度制，必须是正数。
#    反向绕线的 CCT 按照 starting_point_in_ksi_phi_coordinate 和 end_point_in_ksi_phi_coordinate 确定。典型值 67.5
# tilt_angles 各极倾斜角，任意长度数组。典型值 [30,90,90,90]
# winding_number  匝数
# current  电流，电流方向按照 starting_point_in_ksi_phi_coordinate 到 end_point_in_ksi_phi_coordinate。负数则反向
# starting_point_in_ksi_phi_coordinate  CCT 起点，以 ksi_phi_coordinate 坐标系表示，类型为 P2
# end_point_in_ksi_phi_coordinate       CCT 终点，以 ksi_phi_coordinate 坐标系表示，类型为 P2
# disperse_number_per_winding    每匝线圈离散电流元数目，数字越大计算精度越高，默认 120

# 下面举例，首先建立设计轨道，前后偏移段 1 米，中间顺时针圆弧
# 圆弧半径 1 米，弧度 30 deg
traj = (
    Trajectory.set_start_point(P2.origin())
    .first_line(direct=P2.x_direct(),length=1)
    .add_arc_line(radius=1,clockwise=True,angle_deg=30)
    .add_strait_line(length=1)
)
# 依靠 traj 创建 CCT
# 因为圆弧顺时针，所以 end_point_in_ksi_phi_coordinate 中 phi = -30 deg
# 这是因为为了创建顺时针旋转的 CCT，建立的局部坐标系 y 轴方向是 CCT起点方向的负向
# 看不懂没关系，后面使用 Beamline 更加无脑
cct = CCT.create_cct_along(
    trajectory=traj,
    s=1,
    big_r=1,
    small_r=60*MM,
    bending_angle=30,
    tilt_angles=[90,30],
    winding_number=30,
    current=1000,
    starting_point_in_ksi_phi_coordinate=P2(0,0),
    end_point_in_ksi_phi_coordinate=P2(30*2*math.pi,-30/180*math.pi)
)
# Plot3.plot_cct(cct)
# # 绘制 CCT 的局部坐标系
# Plot3.plot_local_coordinate_system(cct.local_coordinate_system,axis_lengths=[1,0.5,0.1],describe='b-')
# Plot3.plot_line2(traj)
# Plot3.show()

# 函数 global_path3() 获取 CCT 路径点，以全局坐标系的形式。主要目的是为了 CUDA 计算
global_path3 = cct.global_path3()
# Plot3.plot_line2(traj)
# Plot3.plot_p3s(global_path3)
# Plot3.show()


# 函数 global_current_elements_and_elementary_current_positions()
# 获取全局坐标系下的
# 电流元 (miu0/4pi) * current * (p[i+1] - p[i])
# 和
# 电流元的位置 (p[i+1]+p[i])/2
# 主要目的是为了 CUDA 计算
# 返回值是两个 numpy 数组
# 内部使用了 flatten()，将 n*3 的二维数组，转为一维

# p2_function(ksi)
# 完成 ksi -> P2(ksi,phi) 的映射


# p3_function(ksi)
# 完成 ksi -> P3(x,y,z) 的映射，映射结果显示的是局部坐标系中 cct 线圈


# conductor_length() 求 CCT 线圈导体长度
# line_number 表示导线数目
# disperse_number_per_winding 计算是，一匝线圈分段数目
print(cct.conductor_length(line_number=1,disperse_number_per_winding=3600)) # 17.443395580963664
print(cct.conductor_length(line_number=1,disperse_number_per_winding=360)) # 17.442809870531804

# 类函数 as_cct() 将任意的对象“转为” cct，实际上什么也没做
# 但是 IDE 就能根据返回值做代码提示
anything = 123
anything = CCT.as_cct(anything)
print(anything) # 123 实际上什么也没做

# 类函数 calculate_a() 计算极角 a
# 参数为 big_r 和 small_r
# 可以看到 a ≈ big_r if big_r>>small_r
print(CCT.calculate_a(100,1)) # 99.99499987499375


# 类函数 calculate_eta() 计算 η
# 参数为 big_r 和 small_r

# 类函数 calculate_cheta() 计算 cosh(η)
# 参数为 big_r 和 small_r

# 类函数 calculate_sheta() 计算 sinh(η)
# 参数为 big_r 和 small_r