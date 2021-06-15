"""
必须参数

DL2 = 2.6 m

QS3 的 QS 参数需要改变，因为比例问题

前段需要简化

中间部分不需要增加磁铁

长度控制更简单一些
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path, system
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from work.A01run import *
from cctpy import *

GAP1 = 0.45
GAP2 = 0.45
DL2 = 2.6

QS1_LEN = 0.27
QS2_LEN = 0.27


# Q1_LEN = 0.27
# Q2_LEN = 0.27
# GAP_Q12 = 0.45

DL1 = 1.633452377915599
print(DL1)


traj = (
    Trajectory.set_start_point().first_line(length=DL1)
    .add_arc_line(radius=0.95,clockwise=False,angle_deg=22.5)
    .add_strait_line(GAP1)
    .add_strait_line(QS1_LEN)
    .add_strait_line(GAP1)
    .add_arc_line(radius=0.95,clockwise=False,angle_deg=22.5)
    .add_strait_line(DL1)
    .add_strait_line(DL2)
    .add_arc_line(radius=0.95,clockwise=True,angle_deg=67.5)
    .add_strait_line(GAP2)
    .add_strait_line(QS2_LEN)
    .add_strait_line(GAP2)
    .add_arc_line(radius=0.95,clockwise=True,angle_deg=67.5)
    .add_strait_line(0.45)
    .add_strait_line(0.27)
    .add_strait_line(DL2-0.45-0.27)
)

print(traj.point_at_end())

Plot2.plot(traj)
Plot2.equal()
Plot2.show()