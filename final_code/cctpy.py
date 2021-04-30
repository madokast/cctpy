"""
CCT 建模优化代码

作者：赵润晓
日期：2021年4月24日
"""

import multiprocessing  # since v0.1.1 多线程计算
import time  # since v0.1.1 统计计算时长
from typing import Callable, Dict, Generic, Iterable, List, NoReturn, Optional, Tuple, TypeVar, Union
import matplotlib.pyplot as plt
import math
import random  # since v0.1.1 随机数
import sys
import os  # since v0.1.1 查看CPU核心数
import numpy
from scipy.integrate import solve_ivp  # since v0.1.1 ODE45
import warnings  # since v0.1.1 提醒方法过时
from packages.constants import *
from packages.base_utils import BaseUtils
from packages.point import P2, P3, ValueWithDistance
from packages.local_coordinate_system import LocalCoordinateSystem
from packages.lines import Line2, StraightLine2, ArcLine2
from packages.trajectory import Trajectory
from packages.magnets import *
from packages.particles import *
