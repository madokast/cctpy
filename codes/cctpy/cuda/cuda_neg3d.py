# -*- coding: utf-8 -*-

"""
#define X 0
#define Y 1
#define Z 2
__device__ __forceinline__ void neg3d(float *v)
{
    v[X] = -v[X];
    v[Y] = -v[Y];
    v[Z] = -v[Z];
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
    
    __device__ __forceinline__ void neg3d(float *v)
    {
        v[X] = -v[X];
        v[Y] = -v[Y];
        v[Z] = -v[Z];
    }

    __global__ void neg_helper(float *v){
        neg3d(v);
    }
""")

f = mod.get_function("neg_helper")


def neg3d(v: np.ndarray) -> None:
    """
    v = -v
    Parameters
    ----------
    v 三维矢量

    Returns None
    -------

    """
    f(drv.InOut(v), block=(1, 1, 1), grid=(1, 1))
