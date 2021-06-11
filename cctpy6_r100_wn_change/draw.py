from run import create_beamline,second_bending_part_start_point,second_bending_part_start_direct
from cctpy import *

param = []

bl = create_beamline(param,second_bending_part_start_point,second_bending_part_start_direct)


beamline_phase_ellipse_multi_delta(
    bl,8,[-0.05,0,0.05],
)