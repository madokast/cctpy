# -*- coding: utf-8 -*-

"""
#define X 0
#define Y 1
#define Z 2

"""

import numpy as np

import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
    #define X 0
    #define Y 1
    #define Z 2
    
    __device__ float g;

    __global__ void global(float *a, float* b){
        g = *a;
        g += 1.0f;
        *b = g;
    }
""")

f = mod.get_function("global")


def use_global(src: np.ndarray, des: np.ndarray) -> None:
    """
    v = -v
    Parameters
    ----------
    v 三维矢量

    Returns None
    -------

    """
    f(drv.In(src), drv.Out(des), block=(1, 1, 1), grid=(1, 1))
