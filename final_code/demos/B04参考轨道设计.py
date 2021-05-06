"""
CCT 建模优化代码
参考轨道设计

作者：赵润晓
日期：2021年5月3日
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

# 设计一段机架竖线轨道（HUST-PTF 机架）
traj = ( # 使用括号，方便书写链式语法
    Trajectory
        # 设计轨道的起点，一般就设为原点
        .set_start_point(start_point=P2.origin())
        # 添加第一条直线段，需要给出直线的方向和长度
        .first_line(direct=P2.x_direct(), length=3.1)
        # 添加一段圆弧，半径0.746，逆时针，偏转角度57度
        .add_arc_line(radius=0.746, clockwise=False, angle_deg=57)
        # 添加一条直线段，只需要给出长度 4.3，因为必须和之前的轨迹相切
        .add_strait_line(length=4.3)
        # 继续尾接圆弧，此时为顺时针
        .add_arc_line(radius=0.746, clockwise=True, angle_deg=57)
        .add_strait_line(length=0.684)
        .add_arc_line(radius=0.7854, clockwise=True, angle_deg=90)
        .add_strait_line(length=3.5)
)

# 查看 ISOC 位置，即轨道末尾
end = traj.point_at_end()
print('ISOC 位置',end)
# ISOC 位置 (8.162644337939188, 0.00028200192290395165)

# 查看束线总长度
total_length = traj.get_length()
print('束线总长度',total_length)
# 束线总长度 14.302001244130768

# 绘图
Plot2.plot(traj)
# 设置横纵坐标比例相同
Plot2.equal()
# 设置绘图信息
Plot2.info(
    x_label="x/m", # x 轴标签
    y_label="y/m", # y 轴标签
    title="HUST-PTF GANTRY Trajectory", # 标题
    font_size=32 # 字体大小
)
# 展示图片
Plot2.show()