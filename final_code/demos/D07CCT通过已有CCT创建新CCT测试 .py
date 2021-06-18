"""
CCT 建模优化代码
CCT通过已有CCT创建新CCT测试

作者：赵润晓
日期：2021年6月17日
"""



from math import pi
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *


# 先建立一个最简单的 CCT
cct = CCT(
    local_coordinate_system=LocalCoordinateSystem.global_coordinate_system(),
    big_r=1.0,
    small_r=100*MM,
    bending_angle=45,
    tilt_angles=[30],
    winding_number=45,
    current=10000,
    starting_point_in_ksi_phi_coordinate=P2.origin(),
    end_point_in_ksi_phi_coordinate=P2(x=45*2*pi, y=BaseUtils.angle_to_radian(-45))
)

Plot2.plot_cct_path3d_in_2d(cct,describe='y--')

cct2 = CCT.create_by_existing_cct(
    existing_cct=cct,
    winding_number=10,
    end_point_in_ksi_phi_coordinate=P2(x=10*2*pi, y=BaseUtils.angle_to_radian(-45))
)

Plot2.plot_cct_path3d_in_2d(cct2,describe='r-')


Plot2.equal()
Plot2.show()


