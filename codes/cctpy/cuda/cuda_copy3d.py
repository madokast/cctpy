# -*- coding: utf-8 -*-

"""
#define X 0
#define Y 1
#define Z 2
__device__ __forceinline__ void copy3d(float *src, float* des)
{
    des[X] = src[X];
    des[Y] = src[Y];
    des[Z] = src[Z];
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
    
    __device__ __forceinline__ void copy3d(float *src, float* des)
    {
        des[X] = src[X];
        des[Y] = src[Y];
        des[Z] = src[Z];
    }

    __global__ void copy_helper(float *src, float* des){
        copy3d(src, des);
    }
""")

f = mod.get_function("copy_helper")


def copy3d(src: np.ndarray, des: np.ndarray) -> None:
    """
    des = src
    Parameters
    ----------
    src 源
    des 目的地

    Returns None
    -------

    """
    f(drv.In(src), drv.InOut(des), block=(1, 1, 1), grid=(1, 1))
