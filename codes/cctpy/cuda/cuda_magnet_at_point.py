# -*- coding: utf-8 -*-

"""
dB 函数 CUDA
"""
import time

import numpy as np

import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

from cctpy.baseutils import Vectors

mod = SourceModule("""
// cct_for_cuda.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include<stdio.h>
#include <math.h> // CUDA IGNORE
#define MM 0.001f
#define DIM 3
#define PI 3.1415927f
#define X 0
#define Y 1
#define Z 2

float sqrtf(float a) {
    return (float)sqrt(a);
}

__device__ __forceinline__ void add3d(float* a, float* b, float* ret)
{
    ret[X] = a[X] + b[X];
    ret[Y] = a[Y] + b[Y];
    ret[Z] = a[Z] + b[Z];
}

__device__ __forceinline__ void add3d_local(float* a_local, float* b)
{
    a_local[X] += b[X];
    a_local[Y] += b[Y];
    a_local[Z] += b[Z];
}

__device__ __forceinline__ void sub3d(float* a, float* b, float* ret)
{
    ret[X] = a[X] - b[X];
    ret[Y] = a[Y] - b[Y];
    ret[Z] = a[Z] - b[Z];
}

__device__ __forceinline__ void copy3d(float* src, float* des)
{
    des[X] = src[X];
    des[Y] = src[Y];
    des[Z] = src[Z];
}

__device__ __forceinline__ void cross3d(float* a, float* b, float* ret)
{
    ret[X] = a[Y] * b[Z] - a[Z] * b[Y];
    ret[Y] = -a[X] * b[Z] + a[Z] * b[X];
    ret[Z] = a[X] * b[Y] - a[Y] * b[X];
}

__device__ __forceinline__ void dot_a_v(float a, float* v)
{
    v[X] *= a;
    v[Y] *= a;
    v[Z] *= a;
}

__device__ __forceinline__ float dot_v_v(float* v1, float* v2)
{
    return v1[X] * v2[X] + v1[Y] * v2[Y] + v1[Z] * v2[Z];
}

__device__ __forceinline__ float len3d(float* v)
{
    return sqrtf(v[X] * v[X] + v[Y] * v[Y] + v[Z] * v[Z]);
}

__device__ __forceinline__ void neg3d(float* v)
{
    v[X] = -v[X];
    v[Y] = -v[Y];
    v[Z] = -v[Z];
}

// 注意，这里计算的不是电流元的磁场，还需要乘以 电流 和 μ0/4π (=1e-7)
__device__  __forceinline__ void dB(float *p0, float *p1, float *p, float *ret)
{
    float p01[DIM];
    float r[DIM];
    float rr;

    sub3d(p1, p0, p01); // p01 = p1 - p0

    add3d(p0, p1, r); // r = p0 + p1

    dot_a_v(0.5,r); // r = (p0 + p1)/2

    sub3d(p, r, r); // r = p - r

    rr = len3d(r); // rr = len(r)

    cross3d(p01, r, ret); // ret = p01 x r

    rr = 1.0 / rr / rr / rr; // changed

    dot_a_v(rr, ret); // rr . (p01 x r)
}

// line = float[length][3]。计算导线 line 在 p 点产生的磁场，length 表示 line 中点数目，返回值放到 ret
 __device__ void  magnet_at_point(float **line, int length, float current, float *p, float *ret)
{
    int i;
    float db[3];

    ret[X] = 0.0f;
    ret[Y] = 0.0f;
    ret[Z] = 0.0f;
    
    printf("len==%d -- cuda",length);

    for (i = 0; i < length - 1; i++)
    {
        dB(line[i], line[i + 1], p, db);
        add3d_local(ret, db);
    }

    dot_a_v(current * 1e-7, ret);
}

// float **line, int length, float current, float *p, float *ret
__global__ void magnet_at_point_help(float *line, int* length, float* current, float *p, float *ret)
{
    int i;
    int len = *length;
    
    float** lines = (float**)malloc(len*sizeof(float*));
    
    // printf("len=%d --cuda",len);
    // printf("len=%f --cuda",*current);
    
    for(i=0;i<len;i++){
        lines[i] = (float*)malloc(DIM*sizeof(float));
        lines[i][X] = line[i*DIM+X];
        lines[i][Y] = line[i*DIM+Y];
        lines[i][Z] = line[i*DIM+Z];
    }
    magnet_at_point(lines, len, *current, p, ret);
    
    for(i=0;i<len;i++){
        free(lines[i]);
    }
    
    free(lines);
}
""")

f = mod.get_function("magnet_at_point_help")


# float **line, int length, float current, float *p, float *ret
def magnet_at_point(line: np.ndarray, length: np.ndarray, current: np.ndarray, p: np.ndarray, result: np.ndarray) -> None:
    # s = time.time()
    f(drv.In(line), drv.In(length), drv.In(current), drv.In(p), drv.Out(result), block=(1, 1, 1), grid=(1, 1))
    # e = time.time()
    # print(f"time={e-s}")
