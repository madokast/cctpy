# -*- coding: utf-8 -*-

"""
#define X 0
#define Y 1
#define Z 2
    __device__ __forceinline__ void len3d(float *v, float *len)
    {
        *len = sqrtf(v[X]*v[X]+v[Y]*v[Y]+v[Z]*v[Z]);
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
    
    __device__ __forceinline__ void len3d(float *v, float *len)
    {
        *len = sqrtf(v[X]*v[X]+v[Y]*v[Y]+v[Z]*v[Z]);
    }

    __global__ void len3d_helper(float *v, float *len){
        len3d(v, len);
    }
""")

f = mod.get_function("len3d_helper")


def len3d(v: np.ndarray, length: np.ndarray) -> None:
    """
    返回矢量长度
    Parameters
    ----------
    v 矢量
    length 长度

    Returns 空
    -------

    """
    f(drv.In(v), drv.InOut(length), block=(1, 1, 1), grid=(1, 1))
