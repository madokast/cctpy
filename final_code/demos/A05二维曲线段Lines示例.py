"""
CCT 建模优化代码
二维曲线段

作者：赵润晓
日期：2021年4月27日
"""

from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

# 包含 Line2、StraightLine2、ArcLine2、Trajectory 四个类

# Line2 是抽向基类，表示二维 xy 平面任意一条有方向的连续曲线段，可以是直线、圆弧、或它们的组合
# 这个类主要用于构建理想轨道，理想轨道的用处在于
# 1. 更容易确定磁铁元件的位置
# 2. 用于构建参考粒子
# 3. 用于研究理想轨道上的磁场分布

# 注：因为 Line2 的用处在于构建理想轨道，理想轨道一般就以全局坐标系进行构建，所以不涉及坐标变换

# 使用一条有直线段和一条有向圆弧，来说明 Line2 的函数
# straight_line 表示一条有方向的直线段，长度 1 米，方向平行于 x 轴正向，起点为原点
straight_line = StraightLine2(
    length=1*M, # 长度 1 米，米为标准单位，*M 可不写
    direct=P2.x_direct(), # 方向平行于 x 轴正向
    start_point=P2.origin() # 起点为原点
)
# arc_line 表示一条有方向的圆弧，圆弧的弧心为原点，半径 1 米
# 以弧心建立极坐标系（极点为弧心，极轴平行于 x 轴正方向），则圆弧起点对应极角为 0 弧度
# 圆弧方向为逆时针，旋转 180 度
arc_line = ArcLine2(
    starting_phi=0, # 
    center=P2.origin(), # 圆弧的弧心为原点
    radius=1*M, # 半径 1 米，米为标准单位，*M 可不写
    total_phi=math.pi, # 旋转 180 度
    clockwise=False # 圆弧方向为逆时针
)

# 下面以 straight_line 和 arc_line 为例说明 Line2 的函数使用方法

# 函数 get_length() 返回二维曲线的长度
print("straight_line 的长度为",straight_line.get_length()) # 1.0
print("arc_line 的长度为",arc_line.get_length()) # 3.141592653589793

# 函数 point_at(s) 返回二维曲线的 s 位置的点坐标
# 所谓 s 位置，即以曲线起点出发（曲线有方向，有长度，所以必存在起点），沿曲线运动 s 长度后，所在点的坐标
# s = 0 时，即曲线的起点
print("straight_line 的起点为",straight_line.point_at(0)) # (0.0, 0.0)
print("arc_line 的起点为",arc_line.point_at(0)) # (1.0, 0.0)
# s = get_length() 时，即曲线的终点
print("straight_line 的终点为",straight_line.point_at(straight_line.get_length())) # (1.0, 0.0)
print("arc_line 的终点为",arc_line.point_at(arc_line.get_length())) # (-1.0, 1.2246467991473532e-16)
# 对于任意 [0,get_length()] 位置，都可以计算
print("straight_line.point_at(0.5) 为",straight_line.point_at(0.5)) # (0.5, 0.0)
print("arc_line.point_at(math.pi/2) 为",arc_line.point_at(math.pi/2)) # (6.123233995736766e-17, 1.0)

# 函数 direct_at(s) 返回二维曲线的 s 位置的切向
# s = 0 时，即曲线的起点位置处切向
print("straight_line 的起点处切向为",straight_line.direct_at(0)) # (1.0, 0.0)
print("arc_line 的起点处切向为",arc_line.direct_at(0)) # (6.123233995736766e-17, 1.0)
# s = get_length() 时，即曲线的终点处切向
print("straight_line 的终点处切向为",straight_line.direct_at(straight_line.get_length())) # (1.0, 0.0)
print("arc_line 的终点处切向为",arc_line.direct_at(arc_line.get_length())) # (-1.8369701987210297e-16, -1.0)
# 对于任意 [0,get_length()] 位置，都可以计算
print("straight_line.direct_at(0.5) 为",straight_line.direct_at(0.5)) # (1.0, 0.0)
print("arc_line.direct_at(math.pi/2) 为",arc_line.direct_at(math.pi/2)) # (-1.0, 1.2246467991473532e-16)

# 函数 right_hand_side_point(s,d) 和  left_hand_side_point(s,d) 用来计算曲线 s 位置处，右手侧/左手侧 d 位置处点坐标
# 可以用人沿着曲线运动来直观的理解，假设人沿着曲线正方向运动，先运动 s 距离，然后他右手侧/左手侧 d 位置处的点，即函数的返回值
# straight_line 起点位置，右手侧 1 米位置的点为（很明显是 (0,-1)）
print("straight_line 起点位置，右手侧 1 米位置的点为",straight_line.right_hand_side_point(0,1)) # (6.123233995736766e-17, -1.0)
# arc_line 终点位置，左手侧 1 米位置的点为（很明显是原点）
print("arc_line 终点位置，左手侧 1 米位置的点为",arc_line.left_hand_side_point(arc_line.get_length(),1)) # (0.0, 2.465190328815662e-32)

# 函数 point_at_start() 和 point_at_end() 返回曲线起点和终点坐标
print("straight_line 的起点为",straight_line.point_at_start()) # (0.0, 0.0)
print("straight_line 的终点为",straight_line.point_at_end()) # (1.0, 0.0)

# 函数 direct_at_start() 和 direct_at_end() 返回曲线起点和终点处的切向
print("arc_line 的起点处的切向为",arc_line.direct_at_start()) # (6.123233995736766e-17, 1.0)
print("arc_line 的终点处的切向为",arc_line.direct_at_end()) # (-1.8369701987210297e-16, -1.0)

# 函数 __add__()，可以实现曲线的平移，使用 + 运算符，配合二维矢量实现
# 将 straight_line 向上平移 0.5 米
straight_line_up05 = straight_line+P2(y=0.5)
print("straight_line 向上平移 0.5 米，起点为",straight_line_up05.point_at_start()) # (0.0, 0.5)

# disperse2d(step) 将二维曲线离散成连续的二维点，step 为离散点步长，默认 1 毫米。返回值为二维点的数组。
# straight_line 按照 step=0.2 米离散
# 返回 [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (0.6000000000000001, 0.0), (0.8, 0.0), (1.0, 0.0)]
print("straight_line 按照 step=0.2 米离散",straight_line.disperse2d(0.2))
# arc_line 按照默认 step 离散，查看离散后点数组的长度，和前 2 项
print("arc_line 按照默认 step 离散，查看离散后点数组的长度",len(arc_line.disperse2d())) # 3143
print("arc_line 按照默认 step 离散，查看前 2 项",arc_line.disperse2d()[:2]) # [(1.0, 0.0), (0.9999995001296789, 0.0009998701878188414)]

# disperse2d_with_distance() 将二维曲线离散成连续的二维点，其中二维点带有其所在位置（距离）
# 带有位置/距离，是为了方便磁场计算时。得到磁场某分量按照位置/距离的分布
# straight_line 按照 step=0.5 米离散
# 返回 [(0.0:(0.0, 0.0)), (0.25:(0.25, 0.0)), (0.5:(0.5, 0.0)), (0.75:(0.75, 0.0)), (1.0:(1.0, 0.0))]
print("straight_line 按照 step=0.2 米离散",straight_line.disperse2d_with_distance(0.25))

# disperse3d() 和 disperse3d_with_distance()和上面两个方法类似，返回的点为三维点
# 因此可以传入二维到三维的转换的 lambda 函数，p2_t0_p3
# arc_line 按照默认 step 离散，p2_t0_p3 将转换的点 z 方向平移 0.7 米，查看前 2 项
print(
    "arc_line 按照默认 step 离散，p2_t0_p3 将转换的点 z 方向平移 0.7 米，查看前 2 项",
    arc_line.disperse3d_with_distance(p2_t0_p3=lambda p2:p2.to_p3()+P3(z=0.7))[:2]
)# [(0.0:(1.0, 0.0, 0.7)), (0.0009998703544206852:(0.9999995001296789, 0.0009998701878188414, 0.7))]

# 函数 __str__()，将二维曲线转为字符串，在 print() 时自动调用
print(straight_line) # 直线段，起点(0.0, 0.0)，方向(1.0, 0.0)，长度1.0
print(straight_line.__str__()) # 直线段，起点(0.0, 0.0)，方向(1.0, 0.0)，长度1.0
print(arc_line) # 弧线段，起点(1.0, 0.0)，方向(6.123233995736766e-17, 1.0)，顺时针False，半径1.0，角度3.141592653589793
print(arc_line.__str__()) # 弧线段，起点(1.0, 0.0)，方向(6.123233995736766e-17, 1.0)，顺时针False，半径1.0，角度3.141592653589793



# -------------------------------------------------- #

# StraightLine2 表示二维有向直线段。构造方法为指定长度、方向、起点
# straight_line 表示一条有方向的直线段，长度 1 米，方向平行于 x 轴正向，起点为原点
straight_line = StraightLine2(
    length=1*M, # 长度 1 米，米为标准单位，*M 可不写
    direct=P2.x_direct(), # 方向平行于 x 轴正向
    start_point=P2.origin() # 起点为原点
)

# 所有基类 Line2 的函数都可以使用，这里介绍 StraightLine2 特有的一些函数

# 函数 position_of(point) 求点 point 相对于该直线段的方位
# 因为直线段是有方向的，所以可以确定 point 在其左侧还是右侧
# 返回值有三种
#    1  point 在直线段的右侧
#   -1  point 在直线段的左侧
#    0  point 在直线段所在直线上
# 示例如下，其中虚线表示直线段，--> 表示直线段的方向
# 符号 % & 和 $ 表示三个点
#     %
# --------------&---->
#         $
# 如上图，对于点 % ，在直线左侧，返回 -1
# 对于点 & 在直线上，返回 0
# 对于点 $，在直线右侧，返回 1
# 使用 straight_line 举例
print("点 (0.5, 0) 在straight_line上，所以 position_of 返回",straight_line.position_of(P2(x=0.5))) # 0
print("点 (-10, 0) 在straight_line所在直线上，所以 position_of 返回",straight_line.position_of(P2(x=-10))) # 0
print("点 (0.5, 5) 在straight_line的左侧，所以 position_of 返回",straight_line.position_of(P2(0.5, 5))) # -1
print("点 (-10, -1) 在straight_line的右侧，所以 position_of 返回",straight_line.position_of(P2(-10, -1))) # 1

# 函数 straight_line_equation() 返回这条直线段所在直线的方程
# 方程形式为 Ax + By + C = 0，返回 (A,B,C) 形式的元组
# 注意结果不唯一，不能用于比较
print("straight_line 所在直线方程系数为：",straight_line.straight_line_equation()) # (0.0, 1.0, -0.0)


# 类函数 intersecting_point(pa,va,pb,vb) 求两条直线 a 和 b 的交点
# 参数意义如下
# pa 直线 a 上的一点
# va 直线 a 方向
# pb 直线 b 上的一点
# vb 直线 b 方向
# 返回值有三个， cp ka kb
# cp 交点
# ka 交点在直线 a 上的位置，即 cp = pa + va * ka 
# kb 交点在直线 b 上的位置，即 cp = pb + vb * kb
# 下面举个例子，直线 a 有 pa=(0,0) va=(1,1)，直线 b 有 pb=(1,1) vb=(1.-1)
# 很明现交点是 (1,1)
print("求两条直线 a 和 b 的交点:",StraightLine2.intersecting_point(
    pa = P2.origin(),
    va = P2(1,1),
    pb = P2(1,1),
    vb = P2(1,-1)
)) # ((1.0, 1.0), 1.0, 0.0)

# 类函数 is_on_right(view_point, view_direct, viewed_point)
# 查看点 viewed_point 是不是在右边，观察点为 view_point 观测方向为 view_direct
#  返回值
#  1  在右侧
#  0  在正前方或者正后方
#  -1 在左侧
# 这个函数和 position_of() 很类似
# 查看 y 轴右侧的一个点
print("查看 y 轴右侧的一个点",StraightLine2.is_on_right(
    view_point=P2.origin(),
    view_direct=P2.y_direct(),
    viewed_point=P2(1,1)
)) # 1

# 类函数 calculate_k_b(p1,p2) 计算过两点的直线方程
# 使用 y = kx + d 表示，返回 (k,d) 元组
# 使用点 (0,1) 和点 (2,5) 举例子
print("类函数 calculate_k_b(p1,p2) 计算过两点的直线方程",
    StraightLine2.calculate_k_b(
        p1 = P2(0,1),
        p2 = P2(2,5),
)) # (2.0, 1.0)

# -------------------------------------------------- #

# ArcLine2 表示二维有向圆弧段。构造方法为指定弧心、半径、起点弧度、弧角、方向
# 弧心     表示圆弧所在圆的圆心
# 半径     圆弧的半径
# 起点弧度 以弧心建立建立极坐标系（极点为弧心，极轴平行于 x 轴正方向），圆弧起点对应的弧度
# 弧角     圆弧角度，如 1/4 圆的圆弧角度 90 度，半圆角度 180 度。注意单位要转为弧度制
# 方向     有向圆弧段的方向，布尔值，True 为顺时针，False 为逆时针
# 下面以举例例子说明 起点弧度 的意义
# 45 度圆弧，起点分别设为 30 45 60 度，并绘图（为了显示区分，半径设为不同值）
DEG = 1/180*math.pi # 简化弧度制计算
ORI = P2.origin()
arc45_start30 = ArcLine2(starting_phi=30*DEG,center=ORI,radius=10,total_phi=45*DEG,clockwise=False)
arc45_start45 = ArcLine2(starting_phi=45*DEG,center=ORI,radius=11,total_phi=45*DEG,clockwise=False)
arc45_start60 = ArcLine2(starting_phi=60*DEG,center=ORI,radius=12,total_phi=45*DEG,clockwise=False)
# 去除注释显示绘图效果
# plt.plot(*P2.extract(arc45_start30.disperse2d()))
# plt.plot(*P2.extract(arc45_start45.disperse2d()))
# plt.plot(*P2.extract(arc45_start60.disperse2d()))
# plt.axis("equal") # 让 x 轴和 y 轴坐标比例相同
# plt.show()

# 下面以举例例子说明 方向 的意义
# 建立两个方向相反的圆弧
arc45_clockwise = ArcLine2(starting_phi=30*DEG,center=ORI,radius=10,total_phi=45*DEG,clockwise=True)
arc45_anticlockwise = ArcLine2(starting_phi=30*DEG,center=ORI,radius=11,total_phi=45*DEG,clockwise=False)
# 去除注释显示绘图效果
# plt.plot(*P2.extract(arc45_clockwise.disperse2d()))
# plt.plot(*P2.extract(arc45_anticlockwise.disperse2d()))
# plt.axis("equal") # 让 x 轴和 y 轴坐标比例相同
# plt.show()

# ArcLine2 提供另外一种创建圆弧的类方法 ArcLine2.create()，这个方法特别适合束线设计
# 需要的参数如下
# 起点         圆弧的起点（因为 ArcLine2 表示二维有向圆弧段，所以存在起点）
# 起点切向     圆弧的起点处切向向量（因为 ArcLine2 表示二维有向圆弧段，所以存在起点）
# 半径         圆弧半径
# 方向         布尔值，顺时针还是逆时针。True 为顺时针，False 为逆时针
# 弧角        圆弧角度，如 1/4 圆的圆弧角度 90 度，半圆角度 180 度。注意单位是角度制，和构造方法不同
# 举个例子，需要在 (0,100) 位置和 y 方向作为圆弧的起始，半径 10 弧度 90 度的顺时针圆弧
arc90 = ArcLine2.create(
    start_point=P2(0,100),
    start_direct=P2.y_direct(),
    radius=10,
    clockwise=True,
    total_deg=90
)
# 去除注释显示绘图效果
# plt.plot(*P2.extract(arc90.disperse2d()))
# plt.axis("equal") # 让 x 轴和 y 轴坐标比例相同
# plt.show()

# 类函数 unit_circl(phi) 获取极坐标(r=1.0,phi=phi)的点的直角坐标(x,y)
# 例如 45 度
print("unit_circl(45) =",ArcLine2.unit_circle(45*DEG)) # (0.7071067811865476, 0.7071067811865476)

# 函数 __str__() 和 __repr__() 将圆弧转为字符串，在调用 print() 时自动执行
# 下面三条语句均打印：
# 弧线段[起点(0.0, 100.0)，方向(6.123233995736766e-17, 1.0)，顺时针，半径10，角度1.5707963267948966]
print(arc90)
print(arc90.__str__())
print(arc90.__repr__())