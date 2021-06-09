"""
束线元件位置

作者：赵润晓
日期：2021年4月24日
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from hust_sc_gantry import HUST_SC_GANTRY
from cctpy import *

g = HUST_SC_GANTRY(
    DL1=900.78*MM,
    GAP1=430.15*MM,
    GAP2=370.82*MM,
    qs1_length=234.01*MM,
    qs1_aperture_radius=60 * MM,
    qs1_gradient=0.0,
    qs1_second_gradient=0.0,
    qs2_length=200.14*MM,
    qs2_aperture_radius=60 * MM,
    qs2_gradient=0.0,
    qs2_second_gradient=0.0,

    DL2=2350.11*MM,
    GAP3=431.88*MM,
    qs3_length=243.79*MM,
)

traj = (
    Trajectory
    .set_start_point(P2.origin())
    .first_line(direct=P2.x_direct(), length=g.DL1))

# AGCT12
traj=traj.add_arc_line(radius=g.cct12_big_r, clockwise=False, angle_deg=22.5).as_aperture_objrct_on_last(140.5 * MM - 20 * MM)
Plot2.plot_p2(traj.get_last_line2().point_at_middle(),'k.')
print('CCT1',traj.get_last_line2().point_at_middle()/MM)

traj=traj.add_strait_line(length=g.GAP1)

# qs1
traj=traj.add_strait_line(length=g.qs1_length).as_aperture_objrct_on_last(60*MM)
Plot2.plot_p2(traj.get_last_line2().point_at_middle(),'k.')
print('qs1',traj.get_last_line2().point_at_middle()/MM)

traj=traj.add_strait_line(length=g.GAP2)

# qs2
traj=traj.add_strait_line(length=g.qs2_length).as_aperture_objrct_on_last(60*MM)
Plot2.plot_p2(traj.get_last_line2().point_at_middle(),'k.')
print('qs2',traj.get_last_line2().point_at_middle()/MM)


traj=traj.add_strait_line(length=g.GAP2)

# qs1
traj=traj.add_strait_line(length=g.qs1_length).as_aperture_objrct_on_last(60*MM)
Plot2.plot_p2(traj.get_last_line2().point_at_middle(),'k.')
print('qs1',traj.get_last_line2().point_at_middle()/MM)

traj=traj.add_strait_line(length=g.GAP1)

# cct12
traj=traj.add_arc_line(radius=g.cct12_big_r, clockwise=False, angle_deg=22.5).as_aperture_objrct_on_last(140.5 * MM - 20 * MM)
Plot2.plot_p2(traj.get_last_line2().point_at_middle(),'k.')
print('CCT1',traj.get_last_line2().point_at_middle()/MM)

traj=traj.add_strait_line(length=g.DL1)
traj=traj.add_strait_line(length=g.DL2)

# cct345
traj=traj.add_arc_line(radius=g.cct345_big_r, clockwise=True, angle_deg=67.5).as_aperture_objrct_on_last(140.5 * MM)
Plot2.plot_p2(traj.get_last_line2().point_at_middle(),'k.')
print('CCT2',traj.get_last_line2().point_at_middle()/MM)

traj=traj.add_strait_line(length=g.GAP3)

# qs3
traj=traj.add_strait_line(length=g.qs3_length).as_aperture_objrct_on_last(60*MM)
Plot2.plot_p2(traj.get_last_line2().point_at_middle(),'k.')
print('qs3',traj.get_last_line2().point_at_middle()/MM)

traj=traj.add_strait_line(length=g.GAP3)

# cct345
traj=traj.add_arc_line(radius=g.cct345_big_r, clockwise=True, angle_deg=67.5).as_aperture_objrct_on_last(140.5 * MM)
Plot2.plot_p2(traj.get_last_line2().point_at_middle(),'k.')
print('CCT2',traj.get_last_line2().point_at_middle()/MM)

traj=traj.add_strait_line(length=g.DL2)


print(traj.point_at_end()/MM)

Plot2.plot(traj)

for line2 in traj.get_line2_list():
    if isinstance(line2, ArcLine2):
        arc = ArcLine2.as_arc_line2(line2)
        Plot2.plot_p2s([
            arc.point_at_end(), arc.center, arc.point_at_start()
        ], describe='r--')

Plot2.equal()
Plot2.show()
