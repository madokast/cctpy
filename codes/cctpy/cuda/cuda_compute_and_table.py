# -*- coding: utf-8 -*-

"""
测试打表和计算
"""

import numpy as np

import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
    #define X 0
    #define Y 1
    #define Z 2
    #define PI 3.1415927f
    
    __device__ float sin_sum_tb(float *sin_table,int number)
    {
        float sum = 0.0f;
        int i;
        for(i=0;i<number;i++){
            sum += sin_table[i%360];
        }
        
        return sum;
    }
    
    __device__ float sin_sum_compute(int number)
    {
        float sum = 0.0f;
        int i;
        for(i=0;i<number;i++){
            sum += sinf(((float)i)/180.0f*PI);
        }
        
        return sum;
    }

    __global__ void sin_sum_tb_helper(float *tb, float* ret,int* number){
        *ret = sin_sum_tb(tb,*number);
    }
    
    __global__ void sin_sum_compute_helper(float* ret,int* number){
        *ret = sin_sum_compute(*number);
    }
""")

f_tb = mod.get_function("sin_sum_tb_helper")
f_cm = mod.get_function("sin_sum_compute_helper")


def sin_sum_tb(table: np.ndarray, result: np.ndarray, number: np.ndarray) -> None:
    f_tb(drv.In(table), drv.Out(result), drv.In(number), block=(1, 1, 1), grid=(1, 1))


def sin_sum_compute(result: np.ndarray, number: np.ndarray) -> None:
    f_cm(drv.Out(result), drv.In(number), block=(1, 1, 1), grid=(1, 1))
