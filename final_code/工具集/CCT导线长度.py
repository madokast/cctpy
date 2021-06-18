"""
CCT 导线长度
"""

import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)

from cctpy import *

beamline = HUST_SC_GANTRY(
        agcct345_inner_small_r=83 * MM + 17 * MM,
        agcct345_outer_small_r=83 * MM + 15 * MM+ 17 * MM,
        dicct345_inner_small_r=83 * MM + 15 * MM * 2+ 17 * MM,
        dicct345_outer_small_r=83 * MM + 15 * MM * 3+ 17 * MM,
).create_second_bending_part_beamline()

cond_length = 0

for magnet in beamline.get_magnets():
    if isinstance(magnet,CCT):
        cct = CCT.as_cct(magnet)
        length = cct.conductor_length(line_number=2*8,disperse_number_per_winding=3600)
        print(f"长度 = {length} 米")
        cond_length += length

print(f"cct 导线总长度为 {cond_length} 米" )