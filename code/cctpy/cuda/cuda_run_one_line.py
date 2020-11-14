# -*- coding: utf-8 -*-

"""
单个粒子 跑一个导线
"""

import numpy as np

import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
// cct_for_cuda.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <stdio.h>
#include <math.h> // CUDA IGNORE
#define MM 0.001f
#define DIM 3
#define PI 3.1415927f
#define X 0
#define Y 1
#define Z 2
#define Proton_Charge_Quantity 1.6021766208e-19f
#define Proton_Static_MassKg 1.672621898e-27f
#define Proton_Static_MassMeV 938.2720813f
#define Light_Speed 299792458.0f
#define RUN_STEP 0.001f

__device__ __forceinline__ void add3d(float *a, float *b, float *ret)
{
    ret[X] = a[X] + b[X];
    ret[Y] = a[Y] + b[Y];
    ret[Z] = a[Z] + b[Z];
}

__device__ __forceinline__ void add3d_local(float *a_local, float *b)
{
    a_local[X] += b[X];
    a_local[Y] += b[Y];
    a_local[Z] += b[Z];
}

__device__ __forceinline__ void sub3d(float *a, float *b, float *ret)
{
    ret[X] = a[X] - b[X];
    ret[Y] = a[Y] - b[Y];
    ret[Z] = a[Z] - b[Z];
}

__device__ __forceinline__ void copy3d(float *src, float *des)
{
    des[X] = src[X];
    des[Y] = src[Y];
    des[Z] = src[Z];
}

__device__ __forceinline__ void cross3d(float *a, float *b, float *ret)
{
    ret[X] = a[Y] * b[Z] - a[Z] * b[Y];
    ret[Y] = -a[X] * b[Z] + a[Z] * b[X];
    ret[Z] = a[X] * b[Y] - a[Y] * b[X];
}

__device__ __forceinline__ void dot_a_v(float a, float *v)
{
    v[X] *= a;
    v[Y] *= a;
    v[Z] *= a;
}

__device__ __forceinline__ void dot_a_v_ret(float a, float *v, float *ret)
{
    ret[X] = v[X] * a;
    ret[Y] = v[Y] * a;
    ret[Z] = v[Z] * a;
}

__device__ __forceinline__ float dot_v_v(float *v1, float *v2)
{
    return v1[X] * v2[X] + v1[Y] * v2[Y] + v1[Z] * v2[Z];
}

__device__ __forceinline__ float len3d(float *v)
{
    return sqrtf(v[X] * v[X] + v[Y] * v[Y] + v[Z] * v[Z]);
}

__device__ __forceinline__ void neg3d(float *v)
{
    v[X] = -v[X];
    v[Y] = -v[Y];
    v[Z] = -v[Z];
}

// 注意，这里计算的不是电流元的磁场，还需要乘以 电流 和 μ0/4π (=1e-7)
// 2020年11月11日 测试通过
__device__  void dB(float *p0, float *p1, float *p, float *ret)
{
    float p01[DIM];
    float r[DIM];
    float rr;

    sub3d(p1, p0, p01); // p01 = p1 - p0

    add3d(p0, p1, r); // r = p0 + p1

    dot_a_v(0.5, r); // r = (p0 + p1)/2

    sub3d(p, r, r); // r = p - r

    rr = len3d(r); // rr = len(r)

    cross3d(p01, r, ret); // ret = p01 x r

    rr = 1.0 / rr / rr / rr; // changed

    dot_a_v(rr, ret); // rr . (p01 x r)
}

// line = float[length][3]。计算导线 line 在 p 点产生的磁场，length 表示 line 中点数目，返回值放到 ret
__device__  void magnet_at_point(float **line, int length, float current, float *p, float *ret)
{
    int i;
    float db[3];

    ret[X] = 0.0f;
    ret[Y] = 0.0f;
    ret[Z] = 0.0f;

    for (i = 0; i < length - 1; i++)
    {
        dB(line[i], line[i + 1], p, db);
        add3d_local(ret, db);
    }

    dot_a_v(current * 1e-7, ret);
}

// 粒子走一步 m 磁场，p 位置，v 速度，rm 动质量，sp 速率
__device__  __forceinline__  void particle_run_step(float *m, float *p, float *v, float run_mass, float speed)
{
    float a[3]; // 加速度
    float t;    // 运动时间
    float d[3]; // 位置变化 速度变化

    // q v b
    cross3d(v, m, a); // a = v*b

    dot_a_v(Proton_Charge_Quantity / run_mass, a); // a = q v b / mass 加速度

    t = RUN_STEP / speed; // 运动时长

    dot_a_v_ret(t, v, d); // d = t v 位置变化

    add3d_local(p, d); // p+=d

    dot_a_v_ret(t, a, d); // d = t a 速度变化

    add3d_local(v, d); // v+=d
}

// 粒子在单导线磁场中运动，用于测试
// len 运动距离，p 位置，v 速度，rm 动质量，sp 速率，line[des_len][3] 导线，des_len 导线点数，current 电流
 __device__   void particle_run_len_one_line(float len, float *p, float *v, float run_mass, float speed, 
        float **line, int des_len, float current)
{
    float distance = 0.0f;
    float m[3]; // 磁场
    while (distance < len)
    {
        magnet_at_point(line, des_len, current, p, m);
        particle_run_step(m, p, v, run_mass, speed);
        distance += RUN_STEP;
    }
    
}
    
__global__ void particle_run_len_one_line_help(float *line, int* des_len, float* current,float* len,float* p,
        float* v, float* rm, float* speed){ 
        
    int i; 
    
    float** lines = (float**)malloc(*des_len*sizeof(float*));
    
    printf("des_len=%d --cuda",*des_len);
    printf("len=%f --cuda  ",*current);
    printf("len=%f --cuda  ",*len);
    printf("px=%f --cuda  ",p[X]);
    printf("py=%f --cuda ",p[Y]);
    printf("py=%f --cuda  ",p[Z]);
    printf("vx=%f --cuda  ",v[X]);
    printf("vy=%f --cuda  ",v[Y]);
    printf("vz=%f --cuda  ",v[Z]);
    printf("rm=%f --cuda  ",*rm);
    printf("speed=%f --cuda ",*speed);
    
    for(i=0;i<*des_len;i++){
        lines[i] = (float*)malloc(DIM*sizeof(float));
        lines[i][X] = line[i*DIM+X];
        lines[i][Y] = line[i*DIM+Y];
        lines[i][Z] = line[i*DIM+Z];
    }
    
    //void particle_run_len_one_line(float len, float *p, float *v, float run_mass, float speed, 
    //    float **line, int des_len, float current)
    particle_run_len_one_line(*len, p , v, *rm, *speed, lines, *des_len, *current);
    
    for(i=0;i<*des_len;i++){
        free(lines[i]);
    }
    
    free(lines);
}

""")

f = mod.get_function("particle_run_len_one_line_help")


# float *line, int* des_len, float* current,float* len,float* p,float* v, float* rm, float* speed
def particle_run_len_one_line(line: np.ndarray, des_len: np.ndarray, current: np.ndarray,
                              len: np.ndarray, p: np.ndarray, v: np.ndarray,
                              rm: np.ndarray, speed: np.ndarray) -> None:
    """

    Parameters
    ----------
    line 导线
    des_len 分段数目
    current 电流
    len 运动距离
    p 初始位置
    v 初始速度
    rm 动质量
    speed 速率

    Returns
    -------

    """
    f(drv.In(line), drv.In(des_len), drv.In(current), drv.In(len),
      drv.InOut(p), drv.InOut(v), drv.In(rm), drv.In(speed)
      , block=(1, 1, 1),
      grid=(1, 1))
