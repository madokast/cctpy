"""
CCT 建模优化代码

作者：赵润晓
日期：2021年4月24日
"""


from packages.constants import *
from packages.base_utils import BaseUtils
from packages.point import P2, P3, ValueWithDistance
from packages.local_coordinate_system import LocalCoordinateSystem
from packages.line2s import Line2, StraightLine2, ArcLine2
from packages.trajectory import Trajectory
from packages.magnets import *
from packages.cct import *
from packages.particles import *
from packages.beamline import *
from packages.plot import *
from packages.function_part import *
try:
    from packages.gpu_accelerator import GPU_ACCELERATOR
except Exception as e:
    print("导入 GPU_ACCELERATOR 类出现异常：", e)
    print("可能是没有安装好 pycuda，出现此异常不影响 ccpty 核心功能的使用")


if __name__ == "__main__":
    Plot3.__logo__()
