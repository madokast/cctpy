# -*- coding: utf-8 -*-

"""
__device__ __forceinline__ void dot_a_v(float *a, float *v)
{
    v[X] *= *a;
    v[Y] *= *a;
    v[Z] *= *a;
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
    
    __device__ __forceinline__ void dot_a_v(float *a, float *v)
    {
        v[X] *= *a;
        v[Y] *= *a;
        v[Z] *= *a;
    }

    __global__ void dot_helper(float *a, float *v){
        dot_a_v(a, v);
    }
""")

f = mod.get_function("dot_helper")


def dot_a_v(a: np.ndarray, v: np.ndarray) -> None:
    """
    v *= a
    Parameters
    ----------
    a 标量
    v 三维矢量

    Returns None
    -------

    """
    f(drv.In(a), drv.InOut(v), block=(1, 1, 1), grid=(1, 1))
