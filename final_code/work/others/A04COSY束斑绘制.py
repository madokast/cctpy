""""
COSY 束斑绘制

2021年6月23日
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

delta = 0.08
num = 5000
order = 1

# 生成 x xp y yp 四维椭球粒子
pps = []

for _ in range(num):
    r = BaseUtils.Random.uniformly_distributed_in_hypereellipsoid(
        axes=BaseUtils.list_multiply([3.5,7.5,3.5,7.5],MM)
        )
    pps.append(PhaseSpaceParticle(
        x = r[0],
        xp = r[1],
        y = r[2],
        yp = r[3],
        delta=delta
    ))

ppsd = map.apply_phase_space_particles(pps,order)

ps = []

for pp in ppsd:
    ps.append(P2(
        x = pp.x/MM,
        y = pp.y/MRAD,
    ))

Plot2.plot_p2s(ps,describe='r.')
Plot2.info("x/mm","y/mm",f"dp/p={int(delta*100)}%")
Plot2.xlim(-4.5,+4.5)
Plot2.ylim(-4.5,+4.5)
Plot2.show()


