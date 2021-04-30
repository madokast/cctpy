"""
CCT 建模优化代码
设计轨道 trajectory

作者：赵润晓
日期：2021年4月29日
"""

from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

# Trajectory 类表示二维设计轨道，由有方向的直线段和圆弧段组成
# 它的使用方法非常简单，例如构建一个圆角正方形
# 设定圆角半径 1，四条边去除圆角部分外的直线段长度 10
# 构建方法即设定轨迹起点和起点方向，返回不断地接续直线段和圆弧段
square = (Trajectory
    # 设定起点
    .set_start_point(P2.origin()) 
    # 设定第一条直线度（Trajectory 的开头必须是直线段，这样的限定不会给轨迹设定造成阻碍）
    .first_line(direct=P2.x_direct(),length=10)
    # 接续圆弧段
    .add_arc_line(radius=1,clockwise=True,angle_deg=90)
    # 接续直线段
    .add_strait_line(length=10)
    .add_arc_line(radius=1,clockwise=True,angle_deg=90)
    .add_strait_line(length=10)
    .add_arc_line(radius=1,clockwise=True,angle_deg=90)
    .add_strait_line(length=10)
    .add_arc_line(radius=1,clockwise=True,angle_deg=90)
)
# 去除注释显示绘图效果
# plt.plot(*P2.extract(square.disperse2d()))
# plt.axis("equal")
# plt.show()

# 下面详细描述 Trajectory 的各个函数

# 构造器 Trajectory(first_line) 传入第一条二维曲线段
# 不推荐使用，建议按照上文的 set_start_point 和 first_line 方法
# 举例如下
traj = Trajectory(StraightLine2(
    length=10,
    direct=P2.x_direct(),
    start_point=P2(1,1)
))
print(traj) # 01 直线段[起点(1.0, 1.0)，方向(1.0, 0.0)，长度10.0]

# 函数 add_line2() 尾接任意二维曲线，不判断是否和当前轨迹相接、相切
# 因为“不判断是否和当前轨迹相接、相切”，所以不推荐使用

# 函数 add_strait_line(length) 尾接直线段，长度 length
# 尾接的直线方向和接点满足相切

# 函数 add_arc_line(r, c, a) 尾接弧线
# 参数意义如下：
# r 半径
# c 是否顺时针，布尔值，True 表示顺时针，False 表示逆时针
# a 弧角，单位为角度制


# 内部类 __TrajectoryBuilder 用于便捷的构造函数
# set_start_point() 和 first_line()
# 这两个函数应联合使用，用于指定轨迹的起点，和初始直线段
# 下面设置了一条直线段，起点为原点，
straight_line = (Trajectory
    .set_start_point(P2.origin())
    .first_line(direct=P2.x_direct(),length=1))
print(straight_line) # # 01 直线段[起点(0.0, 0.0)，方向(1.0, 0.0)，长度1.0]

# 函数 get_line2_list() 用于获得 trajectory 内部的曲线数组
# 即 add_xxx() 函数都是在 trajectory 内添加数组
# 以 square 为例
lines = square.get_line2_list()
for line in lines:
    print("square 内的曲线段有",line)
# 打印信息如下
# square 内的曲线段有 直线段[起点(0.0, 0.0)，方向(1.0, 0.0)，长度10.0]
# square 内的曲线段有 弧线段[起点(10.0, 0.0)，方向(1.0, 0.0)，顺时针，半径1，角度1.5707963267948966]
# square 内的曲线段有 直线段[起点(11.0, -1.0)，方向(6.123233995736766e-17, -1.0)，长度10.0]
# square 内的曲线段有 弧线段[起点(11.0, -11.0)，方向(6.123233995736766e-17, -1.0)，顺时针，半径1，角度1.5707963267948966]
# square 内的曲线段有 直线段[起点(10.0, -12.0)，方向(-1.0, -1.2246467991473532e-16)，长度10.0]
# square 内的曲线段有 弧线段[起点(-3.6739403974420594e-16, -12.000000000000002)，方向(-1.0, 1.224646799147353e-16)，顺时针，半径1，角度1.5707963267948966]
# square 内的曲线段有 直线段[起点(-1.0000000000000002, -11.000000000000002)，方向(6.123233995736766e-17, 1.0)，长度10.0]
# square 内的曲线段有 弧线段[起点(-0.9999999999999996, -1.0000000000000016)，方向(6.123233995736766e-17, 1.0)，顺时针，半径1，角度1.5707963267948966]

# 函数 as_aperture_objrct_on_last 给 trajectory 轨迹中最后一段添加孔径信息
# 注意：这个信息只用于绘图，无其他意义
# 孔径信息即以直线段、曲线段的形式保存，在绘图时显示
# 还是用 square 举例，在圆角处加上孔径
square_aper = (Trajectory
    .set_start_point(P2.origin()) 
    .first_line(direct=P2.x_direct(),length=10)
    .add_arc_line(radius=1,clockwise=True,angle_deg=90)
    .as_aperture_objrct_on_last(aperture_radius=0.3)
    .add_strait_line(length=10)
    .add_arc_line(radius=1,clockwise=True,angle_deg=90)
    .as_aperture_objrct_on_last(aperture_radius=0.3)
    .add_strait_line(length=10)
    .add_arc_line(radius=1,clockwise=True,angle_deg=90)
    .as_aperture_objrct_on_last(aperture_radius=0.3)
    .add_strait_line(length=10)
    .add_arc_line(radius=1,clockwise=True,angle_deg=90)
    .as_aperture_objrct_on_last(aperture_radius=0.3)
)
# 去除注释显示绘图效果
# plt.plot(*P2.extract(square_aper.disperse2d()))
# for aperture_objrct in square_aper.get_aperture_objrcts(): # get_aperture_objrcts 获取内部的孔径信息
#     plt.plot(*P2.extract(aperture_objrct.disperse2d()))
# plt.axis("equal")
# plt.show()

# 函数 get_aperture_objrcts() 获取内部的孔径信息，即 lines 数组
first_aperture_objrct = square_aper.get_aperture_objrcts()[0]
print("函数 get_aperture_objrcts() 获取内部的孔径信息",type(first_aperture_objrct)) # <class 'packages.lines.ArcLine2'>

# 函数 __str__() 和 __repr__()，将对象转为字符串
# 以下均打印打印
# Trajectory:
# 01 直线段[起点(0.0, 0.0)，方向(1.0, 0.0)，长度10.0]
# 02 弧线段[起点(10.0, 0.0)，方向(1.0, 0.0)，顺时针，半径1，角度1.5707963267948966]
# 03 直线段[起点(11.0, -1.0)，方向(6.123233995736766e-17, -1.0)，长度10.0]
# 04 弧线段[起点(11.0, -11.0)，方向(6.123233995736766e-17, -1.0)，顺时针，半径1，角度1.5707963267948966]
# 05 直线段[起点(10.0, -12.0)，方向(-1.0, -1.2246467991473532e-16)，长度10.0]
# 06 弧线段[起点(-3.6739403974420594e-16, -12.000000000000002)，方向(-1.0, 1.224646799147353e-16)，顺时针，半径1，角度1.5707963267948966]
# 07 直线段[起点(-1.0000000000000002, -11.000000000000002)，方向(6.123233995736766e-17, 1.0)，长度10.0]
# 08 弧线段[起点(-0.9999999999999996, -1.0000000000000016)，方向(6.123233995736766e-17, 1.0)，顺时针，半径1，角度1.5707963267948966]
print(square)
print(square.__str__())
print(square.__repr__())


# 函数 __cctpy__() 时彩蛋，打印 CCTPY 五个字母
cctpy = Trajectory.__cctpy__()
# 去除注释显示绘图效果
# for line in cctpy:
#     plt.plot(*P2.extract(line.disperse2d()),'r-')
# plt.show()