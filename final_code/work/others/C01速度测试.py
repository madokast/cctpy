from math import pi
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
sys.path.append(path.dirname(path.dirname(
    path.abspath(path.dirname(__file__)))))
from cctpy import *

cct = CCT(
    local_coordinate_system=LocalCoordinateSystem.global_coordinate_system(),
    big_r=1.0,
    small_r=100*MM,
    bending_angle=45.0,
    tilt_angles=[90,90,90],
    winding_number=15,
    current=10000,
    starting_point_in_ksi_phi_coordinate=P2(),
    end_point_in_ksi_phi_coordinate=P2(15*2*pi,BaseUtils.angle_to_radian(45)),
    disperse_number_per_winding=360
)


t = BaseUtils.Timer()

for _ in range(10000):

    sum = P3.zeros()
    for i in range(30*1000):
        m = cct.magnetic_field_at(P3(x=i/10000.0))
        sum += m

    print(t.period())
    t.reset()