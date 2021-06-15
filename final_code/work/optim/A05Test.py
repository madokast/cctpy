"""
CCT 建模优化代码

作者：赵润晓
日期：2021年5月21日
"""

from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
sys.path.append(path.dirname(path.dirname(path.abspath(path.dirname(__file__)))))
import time
import numpy as np
from work.optim.A05run import create_gantry_beamline
from hust_sc_gantry import beamline_phase_ellipse_multi_delta
from packages.constants import MM

params = [
    3.5302, 	5.5552, 	-42.9943, 	-19.1617,
    94.9830,	75.8091,	80.5371,
    87.3605,	81.8937,	95.5016,
    -9613.5385,	10969.1606,
    24.0000,	24.0000,
    2.9978,	-92.7398,	85.1549,	88.9347,
    88.3576,	78.3727,	86.5075,
    95.0290,	9147.5885,	-7582.9665,
    22.0000 	, 39.0000,	32.0000
]

bl = create_gantry_beamline(params)

if __name__ == "__main__":

    beamline_phase_ellipse_multi_delta(bl,particle_number=8,foot_step=10*MM)
