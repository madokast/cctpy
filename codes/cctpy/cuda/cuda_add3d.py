# -*- coding: utf-8 -*-

"""
#define X 0
#define Y 1
#define Z 2

__device__ __forceinline__ void add3d(float *a, float *b, float* ret)
{
    ret[X] = a[X] + b[X];
    ret[Y] = a[Y] + b[Y];
    ret[Z] = a[Z] + b[Z];
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
    
    __device__ __forceinline__ void add3d(float *a, float *b, float* ret)
    {
        ret[X] = a[X] + b[X];
        ret[Y] = a[Y] + b[Y];
        ret[Z] = a[Z] + b[Z];
    }

    __global__ void add_helper(float *a, float *b, float* ret){
        add3d(a, b, ret);
    }
""")

f = mod.get_function("add_helper")


def add3d(a: np.ndarray, b: np.ndarray, result: np.ndarray) -> None:
    """
    a_local += b
    Parameters
    ----------
    a_local 三维矢量，原地加法
    b 三维矢量

    Returns None
    -------

    """
    f(drv.In(a), drv.In(b), drv.Out(result), block=(1, 1, 1), grid=(1, 1))
