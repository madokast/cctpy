# -*- coding: utf-8 -*-

"""
#define X 0
#define Y 1
#define Z 2

__device__ __forceinline__ void add3d_local(float *a_local, float *b)
{
    a_local[X] += b[X];
    a_local[Y] += b[Y];
    a_local[Z] += b[Z];
}
"""

import numpy as np

import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
    #define X 0
    #define Y 1
    #define Z 2
    
    __device__ __forceinline__ void add3d_local(float *a_local, float *b)
    {
        a_local[X] += b[X];
        a_local[Y] += b[Y];
        a_local[Z] += b[Z];
    }

    __global__ void add_helper(float *a_local, float *b){
        add3d_local(a_local, b);
    }
""")

f = mod.get_function("add_helper")


def add3d_local(a_local: np.ndarray, b: np.ndarray) -> None:
    """
    a_local += b
    Parameters
    ----------
    a_local 三维矢量，原地加法
    b 三维矢量

    Returns None
    -------

    """
    f(drv.InOut(a_local), drv.In(b), block=(1, 1, 1), grid=(1, 1))
