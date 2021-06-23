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
