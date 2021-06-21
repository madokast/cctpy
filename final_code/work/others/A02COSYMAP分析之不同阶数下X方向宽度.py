""""
A02COSYMAP分析之不同阶数下X方向宽度
"""
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
sys.path.append(path.dirname(path.dirname(
    path.abspath(path.dirname(__file__)))))

from cctpy import *


# COSY_MAP:str = COSY_MAP_手动优化至伪二阶
COSY_MAP:str = COSY_MAP_廖益诚五阶光学优化

map = CosyMap(COSY_MAP)

num = 64
plane_id = PhaseSpaceParticle.XXP_PLANE

# 粒子
pps0 = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_plane(
    xMax=3.5*MM,xpMax=7.5*MM,delta=0.0,number=num,plane_id=plane_id
)


pps5 = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_plane(
    xMax=3.5*MM,xpMax=7.5*MM,delta=0.05,number=num,plane_id=plane_id
)

ppsm5 = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_plane(
    xMax=3.5*MM,xpMax=7.5*MM,delta=-0.05,number=num,plane_id=plane_id
)

pps10 = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_plane(
    xMax=3.5*MM,xpMax=7.5*MM,delta=0.10,number=num,plane_id=plane_id
)

ppsm10 = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_plane(
    xMax=3.5*MM,xpMax=7.5*MM,delta=-0.10,number=num,plane_id=plane_id
)

xw0s:List[P2] = []
xw5s:List[P2] = []
xwm5s:List[P2] = []
xw10s:List[P2] = []
xwm10s:List[P2] = []

for order in range(1,10):
    pps0d =  map.apply_phase_space_particles(pps0,order)
    pps5d =  map.apply_phase_space_particles(pps5,order)
    ppsm5d =  map.apply_phase_space_particles(ppsm5,order)
    pps10d =  map.apply_phase_space_particles(pps10,order)
    ppsm10d =  map.apply_phase_space_particles(ppsm10,order)

    xw0 = BaseUtils.Statistic().add_all([p.x for p in pps0d]).half_width()/MM
    xw5 = BaseUtils.Statistic().add_all([p.x for p in pps5d]).half_width()/MM
    xw10 = BaseUtils.Statistic().add_all([p.x for p in pps10d]).half_width()/MM
    xwm5 = BaseUtils.Statistic().add_all([p.x for p in ppsm5d]).half_width()/MM
    xwm10 = BaseUtils.Statistic().add_all([p.x for p in ppsm10d]).half_width()/MM

    xw0s.append(P2(order,xw0))
    xw5s.append(P2(order,xw5))
    xw10s.append(P2(order,xw10))
    xwm5s.append(P2(order,xwm5))
    xwm10s.append(P2(order,xwm10))

Plot2.plot_p2s(xwm10s,describe='g-')
Plot2.plot_p2s(xwm5s,describe='b-')
Plot2.plot_p2s(xw0s,describe='r-')
Plot2.plot_p2s(xw5s,describe='k-')
Plot2.plot_p2s(xw10s,describe='y-')

Plot2.plot_p2s(xwm10s,describe='go')
Plot2.plot_p2s(xwm5s,describe='bo')
Plot2.plot_p2s(xw0s,describe='ro')
Plot2.plot_p2s(xw5s,describe='ko')
Plot2.plot_p2s(xw10s,describe='yo')

Plot2.info("Order of Beam Optics","Beam Spot Width in X-direction/mm","")
Plot2.legend(
    "dp/p=-10%",
    "dp/p=-5%",
    "dp/p=0",
    "dp/p=5%",
    "dp/p=10%",
)

Plot2.show()

