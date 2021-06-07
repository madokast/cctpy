from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))

from hust_sc_gantry import beamline_phase_ellipse_multi_delta
from work.A04geatpy_problem import *
from work.A04run import *
from cctpy import *


if __name__ == '__main__':
    BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main()

    bl = create_gantry_beamline([
        # -5.0,-10.0,-98.2,-96.5,103.0,102.8,81.3,71.0,60.1,85.9,-7267.0,10599.9,25.0,19.0
        # 3.6,	7.8 	, 30.4, 	-70.8,	83.1 	, 88.4,	85.1 	, 90.8,	76.5 	, 82.2,	- 14216.6,	10375.6,	17.0,	16.0
# -1.6 ,	8.3 ,	57.7 ,	-29.9 ,	89.9 	,76.2 ,	90.3 ,	92.1 ,	82.1 ,	90.3 ,	-9486.8 ,	10766.7 ,	23.0 ,	22.0 
3.6, 	7.8 ,	30.4 ,	-70.8 ,	83.1 ,	88.4 ,	85.1 	,90.8 ,	76.5 	,82.2 	,-14216.6 ,	10375.6 ,	17.0 ,	16.0 


    ])

    beamline_phase_ellipse_multi_delta(
        bl, 8, [-0.05, 0, 0.05]
    )
