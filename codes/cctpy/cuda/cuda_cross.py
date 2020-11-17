# -*- coding: utf-8 -*-

"""
#define X 0
#define Y 1
#define Z 2

__device__ __forceinline__ void cross3d(float *a, float *b, float *ret)
{
    ret[X] = a[Y] * b[Z] - a[Z] * b[Y];
    ret[Y] = -a[X] * b[Z] + a[Z] * b[X];
    ret[Z] = a[X] * b[Y] - a[Z] * b[X];
}
"""

import numpy as np

import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

from cctpy.baseutils import Vectors

mod = SourceModule("""
    #include <stdio.h>
    #define X 0
    #define Y 1
    #define Z 2
    
    __device__ __forceinline__ void cross3d(float *a, float *b, float *ret)
    {
        ret[X] = a[Y] * b[Z] - a[Z] * b[Y];
        ret[Y] = -a[X] * b[Z] + a[Z] * b[X];
        ret[Z] = a[X] * b[Y] - a[Y] * b[X];
    }
    
    __global__ void cross_helper(float *a, float *b, float *ret){
        cross3d(a, b, ret);
        printf("%f",ret[Z]);
    }
""")

f = mod.get_function("cross_helper")


def cross(a: np.ndarray, b: np.ndarray, result: np.ndarray) -> None:
    f(drv.In(a), drv.In(b), drv.Out(result), block=(1, 1, 1), grid=(1, 1))
