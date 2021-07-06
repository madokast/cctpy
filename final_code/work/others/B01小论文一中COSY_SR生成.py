from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
sys.path.append(path.dirname(path.dirname(
    path.abspath(path.dirname(__file__)))))
from cctpy import *

pps = ParticleFactory.distributed_particles(
    x = 3.5*MM, xp = 7.5*MM, y = 0, yp = 0, delta=0.08,
    number = 80, distribution_area= ParticleFactory.DISTRIBUTION_AREA_EDGE,
    x_distributed=True, xp_distributed=True, delta_distributed = True
)

s = SR.to_cosy_sr(pps)

print(s)