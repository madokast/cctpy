"""
CCT 建模优化代码

作者：赵润晓
日期：2021年4月24日
"""

import multiprocessing  # 多线程计算
import time  # 统计计算时长
from typing import Callable, Dict, Generic, Iterable, List, NoReturn, Optional, Tuple, TypeVar, Union
import matplotlib.pyplot as plt
import math
import random  # 随机数
import sys
import os  # 查看CPU核心数
import numpy
from scipy.integrate import solve_ivp  # since v0.1.1 ODE45
import warnings  # 提醒方法过时



from packages.constants import M, MM, LIGHT_SPEED, RAD, MRAD, J, eV, MeV, MeV_PER_C, T, V
from packages.base_utils import BaseUtils
from packages.point import P2, P3, ValueWithDistance
from packages.local_coordinate_system import LocalCoordinateSystem
from packages.line2s import Line2, StraightLine2, ArcLine2
from packages.line3s import Line3, RightHandSideLine3, FunctionLine3, TwoPointLine3, DiscretePointLine3
from packages.trajectory import Trajectory
from packages.magnets import ApertureObject, Magnet, UniformMagnet, LocalUniformMagnet, CombinedMagnet, QS, Q
from packages.cct import CCT, Wire, AGCCT_CONNECTOR
from packages.particles import Protons, RunningParticle, ParticleRunner, ParticleFactory, PhaseSpaceParticle
from packages.beamline import Beamline
from packages.plot import Plot2, Plot3
from packages.function_part import Function_Part
from packages.gpu_accelerator import GPU_ACCELERATOR
from opera_utils import Brick8,Brick8s, OperaConductorScript, OperaFieldTableMagnet
from cosy_utils import CosyMap, MagnetSlicer, SR, COSY_MAP_手动优化至伪二阶,COSY_MAP_廖益诚五阶光学优化
from hust_sc_gantry import HUST_SC_GANTRY, beamline_phase_ellipse_multi_delta


if __name__ == "__main__":
    LOGO = Plot3.__logo__
    LOGO()

    if True:
        BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main()

        data = [5.546, -57.646, 87.426, 92.151, 91.668, 94.503, 	72.425,	82.442,	9445.242 	,
                -5642.488,	25.000,	40.000, 	34.000
                ]

        gantry = HUST_SC_GANTRY(
            qs3_gradient=data[0],
            qs3_second_gradient=data[1],
            dicct345_tilt_angles=[30, data[2], data[3], data[4]],
            agcct345_tilt_angles=[data[5], 30, data[6], data[7]],
            dicct345_current=data[8],
            agcct345_current=data[9],
            agcct3_winding_number=data[10],
            agcct4_winding_number=data[11],
            agcct5_winding_number=data[12],
            agcct3_bending_angle=-67.5*(data[10])/(data[10]+data[11]+data[12]),
            agcct4_bending_angle=-67.5*(data[11])/(data[10]+data[11]+data[12]),
            agcct5_bending_angle=-67.5*(data[12])/(data[10]+data[11]+data[12]),

            DL1=0.9007765,
            GAP1=0.4301517,
            GAP2=0.370816,
            qs1_length=0.2340128,
            qs1_aperture_radius=60 * MM,
            qs1_gradient=0.0,
            qs1_second_gradient=0.0,
            qs2_length=0.200139,
            qs2_aperture_radius=60 * MM,
            qs2_gradient=0.0,
            qs2_second_gradient=0.0,

            DL2=2.35011,
            GAP3=0.43188,
            qs3_length=0.24379,

            agcct345_inner_small_r=83 * MM,
            agcct345_outer_small_r=98 * MM,  # 83+15
            dicct345_inner_small_r=114 * MM,  # 83+30+1
            dicct345_outer_small_r=130 * MM,  # 83+45 +2
        )
        bl_all = gantry.create_beamline()

        f = gantry.first_bending_part_length()

        sp = bl_all.trajectory.point_at(f)
        sd = bl_all.trajectory.direct_at(f)

        bl = gantry.create_second_bending_part(sp, sd)

        ga =  GPU_ACCELERATOR()

        for delta in BaseUtils.linspace(-0.1,0.1,21):
            pps = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_xxp_plane(
                xMax=3.5*MM,xpMax=7.5*MM,delta=delta,number=32
            )

        beamline_phase_ellipse_multi_delta(
            bl, 32, BaseUtils.linspace(-0.1,0.1,21), describles=['r-'], foot_step=20*MM)