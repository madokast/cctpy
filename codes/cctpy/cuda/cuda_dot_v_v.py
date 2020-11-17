# -*- coding: utf-8 -*-

"""
#define X 0
#define Y 1
#define Z 2

__device__ __forceinline__ void dot_v_v(float *v1, float *v2,float* ret)
{
    *ret = v1[X]*v2[X] + v1[Y]*v2[Y] + v1[Z]*v2[Z];
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
    
    __device__ __forceinline__ void dot_v_v(float *v1, float *v2,float* ret)
    {
        *ret = v1[X]*v2[X] + v1[Y]*v2[Y] + v1[Z]*v2[Z];
    }

    __global__ void dot_helper(float *v1, float *v2,float* ret){
        dot_v_v(v1, v2, ret);
    }
""")

f = mod.get_function("dot_helper")


def dot_v_v(v1: np.ndarray, v2: np.ndarray, result: np.ndarray) -> None:
    """
    result = v1 inner v2
    Parameters
    ----------
    v1 三维矢量
    v2 三维矢量
    result 点乘/内积的解，注意是 [] 型 ndarrry 变量

    Returns v1 inner v2
    -------

    """
    f(drv.In(v1), drv.In(v2), drv.Out(result), block=(1, 1, 1), grid=(1, 1))
