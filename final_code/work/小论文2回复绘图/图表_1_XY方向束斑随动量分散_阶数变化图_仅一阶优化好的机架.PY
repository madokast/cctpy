""""
绘制X、Y两个方向束斑大小，随阶数、动量分散变化图。（两个方向 * 动量分散取 8% 0 -7%，共6条线）
然后取一阶和五阶，绘制束斑图
"""
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
sys.path.append(path.dirname(path.dirname(
    path.abspath(path.dirname(__file__)))))

from cctpy import *


#-----------------------------       修改机架 COSY MAP    --------------------
COSY_MAP:str = COSY_MAP_手动优化至伪二阶
# COSY_MAP:str = COSY_MAP_廖益诚五阶光学优化
#-----------------------------------------------------------------------------

map = CosyMap(COSY_MAP)

num = 64

# 粒子
pps0x = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_plane(
    xMax=3.5*MM,xpMax=7.5*MM,delta=0.0,number=num,plane_id=PhaseSpaceParticle.XXP_PLANE
)


pps8x = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_plane(
    xMax=3.5*MM,xpMax=7.5*MM,delta=0.08,number=num,plane_id=PhaseSpaceParticle.XXP_PLANE
)

ppsm7x = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_plane(
    xMax=3.5*MM,xpMax=7.5*MM,delta=-0.07,number=num,plane_id=PhaseSpaceParticle.XXP_PLANE
)

pps0y = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_plane(
    xMax=3.5*MM,xpMax=7.5*MM,delta=0.0,number=num,plane_id=PhaseSpaceParticle.YYP_PLANE
)


pps8y = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_plane(
    xMax=3.5*MM,xpMax=7.5*MM,delta=0.08,number=num,plane_id=PhaseSpaceParticle.YYP_PLANE
)

ppsm7y = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_plane(
    xMax=3.5*MM,xpMax=7.5*MM,delta=-0.07,number=num,plane_id=PhaseSpaceParticle.YYP_PLANE
)





xw0s:List[P2] = []
xw8s:List[P2] = []
xwm7s:List[P2] = []

yw0s:List[P2] = []
yw8s:List[P2] = []
ywm7s:List[P2] = []


for order in range(1,10):
    pps0xd =  map.apply_phase_space_particles(pps0x,order)
    pps8xd =  map.apply_phase_space_particles(pps8x,order)
    ppsm7xd =  map.apply_phase_space_particles(ppsm7x,order)

    pps0yd =  map.apply_phase_space_particles(pps0y,order)
    pps8yd =  map.apply_phase_space_particles(pps8y,order)
    ppsm7yd =  map.apply_phase_space_particles(ppsm7y,order)


    xw0 = BaseUtils.Statistic().add_all([p.x for p in pps0xd]).half_width()/MM
    xw8 = BaseUtils.Statistic().add_all([p.x for p in pps8xd]).half_width()/MM
    xwm7 = BaseUtils.Statistic().add_all([p.x for p in ppsm7xd]).half_width()/MM


    yw0 = BaseUtils.Statistic().add_all([p.y for p in pps0yd]).half_width()/MM
    yw8 = BaseUtils.Statistic().add_all([p.y for p in pps8yd]).half_width()/MM
    ywm7 = BaseUtils.Statistic().add_all([p.y for p in ppsm7yd]).half_width()/MM

    xw0s.append(P2(order,xw0))
    xw8s.append(P2(order,xw8))
    xwm7s.append(P2(order,xwm7))

    yw0s.append(P2(order,yw0))
    yw8s.append(P2(order,yw8))
    ywm7s.append(P2(order,ywm7))

Plot2.plot_p2s(xw0s,describe='g-')
Plot2.plot_p2s(xw8s,describe='b-')
Plot2.plot_p2s(xwm7s,describe='r-')

Plot2.plot_p2s(yw0s,describe='g--')
Plot2.plot_p2s(yw8s,describe='b--')
Plot2.plot_p2s(ywm7s,describe='r--')

Plot2.plot_p2s(xw0s,describe='go')
Plot2.plot_p2s(xw8s,describe='bo')
Plot2.plot_p2s(xwm7s,describe='ro')

Plot2.plot_p2s(yw0s,describe='go')
Plot2.plot_p2s(yw8s,describe='bo')
Plot2.plot_p2s(ywm7s,describe='ro')

Plot2.info("Order of Beam Optics","Beam Spot Width in X/Y-direction/mm","")
Plot2.legend(
    "x dp/p=0",
    "x dp/p=8%",
    "x dp/p=-7%",
    "y dp/p=0",
    "y dp/p=8%",
    "y dp/p=-7%",
)
Plot2.ylim(2.5,4.5)
Plot2.xlim(-1,15)
Plot2.show()

