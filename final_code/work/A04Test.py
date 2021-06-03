# -5.0 	-10.0 	-98.2 	-96.5 	103.0 	102.8 	81.3 	71.0 	60.1 	85.9 	-7267.0 	10599.9 	25.0 	19.0 
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *
from work.A04run import *
from work.A04geatpy_problem import *
from hust_sc_gantry import beamline_phase_ellipse_multi_delta

if __name__ == '__main__':
    BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main()
    
    bl = create_gantry_beamline([
        -5.0,-10.0,-98.2,-96.5,103.0,102.8,81.3,71.0,60.1,85.9,-7267.0,10599.9,25.0,19.0 
    ])

    beamline_phase_ellipse_multi_delta(
        bl,8,[-0.05,0,0.05]
    )

