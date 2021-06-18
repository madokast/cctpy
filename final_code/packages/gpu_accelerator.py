"""
CCT 建模优化代码
GPU CUDA 加速 cctpy 束流跟踪

注意测试代码中的 ga32 和 ga64 定义为
ga32 = GPU_ACCELERATOR(float_number_type=GPU_ACCELERATOR.FLOAT32)
ga64 = GPU_ACCELERATOR(float_number_type=GPU_ACCELERATOR.FLOAT64,block_dim_x=512)

2021年6月17日 增加 CPU 模式

作者：赵润晓
日期：2021年5月4日
"""

# 是否采用 CPU 模式运行
from packages.beamline import Beamline
from packages.cct import CCT
from packages.magnets import *
from packages.particles import *
from packages.trajectory import Trajectory
from packages.line2s import *
from packages.local_coordinate_system import LocalCoordinateSystem
from packages.base_utils import BaseUtils
from packages.constants import *
from packages.point import *
import warnings  # since v0.1.1 提醒方法过时
from scipy.integrate import solve_ivp  # since v0.1.1 ODE45
import numpy
import os  # since v0.1.1 查看CPU核心数
import sys
import random  # since v0.1.1 随机数
import math
import matplotlib.pyplot as plt
from typing import Callable, Dict, Generic, Iterable, List, NoReturn, Optional, Tuple, TypeVar, Union
import time  # since v0.1.1 统计计算时长
import multiprocessing  # since v0.1.1 多线程计算
__CPU_MODE__: bool = False

try:
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
except ModuleNotFoundError as e:
    print("未安装 pycuda，GPU 加速功能将以 CPU 模式运行")
    __CPU_MODE__ = True


class GPU_ACCELERATOR:
    # CUDA 浮点数类型，可以选择 32 位和 64 位浮点数，前者计算速度快，精度较低
    # 使用 32 位浮点数计算时，典型误差为 0.05 mm 和 0.01 mrad
    FLOAT32: str = "FLOAT32"
    FLOAT64: str = "FLOAT64"
    QS_DATA_LENGTH = 16

    def __init__(self,
                 float_number_type: str = FLOAT32,
                 block_dim_x: int = 1024,
                 # 电流元数目最多 2000*120，如果 120 段 1匝，匝最多2000匝
                 max_current_element_number: int = 2000*120,
                 max_qs_datas_length: int = 10000,  # 最多 10000 个 qs
                 cpu_mode: bool = False
                 ) -> None:
        """
        启动一个 GPU 加速器，用于加速 cctpy 束线的粒子跟踪
        还有一些其他功能，效率不高，仅用作测试

        float_number_type 浮点数类型，取值为 FLOAT32 或 FLOAT64，即 32 位运行或 64 位，默认 32 位。
        64 位浮点数精度更高，但是计算的速度可能比 32 位慢 2-10 倍

        block_dim_x 块线程数目，默认 1024 个，必须是 2 的幂次。如果采用 64 位浮点数，取 1024 可能会报错，应取 512 或更低
        不同大小的 block_dim_x，可能对计算效率有影响
        在抽象上，GPU 分为若干线程块，每个块内有若干线程
        块内线程，可以使用 __shared__ 使用共享内存（访问速度快），同时具有同步机制，因此可以方便的分工合作
        块之间，没有同步机制，所以线程通讯无从谈起

        max_current_element_number 最大电流元数目，在 GPU 加速中，CCT 数据以电流元的形式传入显存。
        默认值 2000*120 （可以看作一共 2000 匝，每匝分 120 段）

        max_qs_datas_length 最大 qs 磁铁数目，默认为 10000，取这么大考虑到切片

        cpu_mode 采用 CPU 模式运行
        """
        self.float_number_type = float_number_type
        self.max_current_element_number = int(max_current_element_number)
        self.max_qs_datas_length = int(max_qs_datas_length)
        self.cpu_mode:bool = __CPU_MODE__ or cpu_mode # 只要两者中有一个为 True 则采用 cpu 模式
        if self.cpu_mode:
            print("GPU 加速功能将以 CPU 模式运行")

        # 检查 block_dim_x 合法性
        if block_dim_x > 1024 or block_dim_x < 0:
            raise ValueError(
                f"block_dim_x 应 >=1 and <=1024 内取，不能是{block_dim_x}")
        if block_dim_x & (block_dim_x-1) != 0:
            raise ValueError(f"block_dim_x 应该取 2 的幂次，不能为{block_dim_x}")
        self.block_dim_x: int = int(block_dim_x)

        # 头文件导入
        cuda_code_00_include = """
        // 只导入 stdio，用于标准输出 printf() 函数
        // CUDA 中的一些内置函数和方法，无需导入
        #include <stdio.h>

        """

        # 定义浮点类型
        cuda_code_01_float_type_define: str = None
        if float_number_type == GPU_ACCELERATOR.FLOAT32:
            cuda_code_01_float_type_define = """
            // 定义为 32 位浮点数模式
            #define FLOAT32

            """
            self.numpy_dtype = numpy.float32
        elif float_number_type == GPU_ACCELERATOR.FLOAT64:
            cuda_code_01_float_type_define = """
            // 定义为 64 位浮点数模式
            #define FLOAT64

            """
            self.numpy_dtype = numpy.float64

            if self.block_dim_x > 512:
                print(f"当前 GPU 设置为 64 位模式，块线程数（{self.block_dim_x}）可能过多，内核可能无法启动\n" +
                      "典型异常为 pycuda._driver.LaunchError: cuLaunchKernel failed: too many resources requested for launch\n" +
                      "遇到此情况，可酌情调小块线程数")

        else:
            raise ValueError(
                "float_number_type 必须是 GPU_ACCELERATOR.FLOAT32 或 GPU_ACCELERATOR.FLOAT64")

        # 宏定义
        # CUDA 代码和 C 语言几乎一模一样。只要有 C/C++ 基础，就能看懂 CUDA 代码
        cuda_code_02_define = """
        // 根据定义的浮点数模式，将 FLOAT 宏替换为 float 或 double
        #ifdef FLOAT32
        #define FLOAT float
        #else
        #define FLOAT double
        #endif

        // 维度 三维
        #define DIM (3)
        // 维度索引 0 1 2 表示 X Y Z，这样对一个数组取值，看起来清晰一些
        #define X (0)
        #define Y (1)
        #define Z (2)
        // 粒子参数索引 (px0, py1, pz2, vx3, vy4, vz5, rm6 相对质量, e7 电荷量, speed8 速率, distance9 运动距离)
        #define PARTICLE_DIM (10)
        #define PX (0)
        #define PY (1)
        #define PZ (2)
        #define VX (3)
        #define VY (4)
        #define VZ (5)
        #define RM (6)
        #define E (7)
        #define SPEED (8)
        #define DISTANCE (9)

        // 块线程数目
        #define BLOCK_DIM_X ({block_dim_x})
        #define QS_DATA_LENGTH (16)
        #define MAX_CURRENT_ELEMENT_NUMBER ({max_current_element_number})
        #define MAX_QS_DATAS_LENGTH ({max_qs_datas_length})
        """.format(
            block_dim_x=self.block_dim_x,
            max_current_element_number=self.max_current_element_number,
            max_qs_datas_length=self.max_qs_datas_length
        )

        # 向量运算内联函数
        cuda_code_03_vct_functions = """
        // 向量叉乘
        // 传入 a b ret 三个数组，将 a × b 的结果传入 ret 中
        // 仔细阅读具体实现，发现 ret 不能是 a 或者 b，这样会导致结果出错
        __device__ __forceinline__ void vct_cross(FLOAT *a, FLOAT *b, FLOAT *ret) {
            ret[X] = a[Y] * b[Z] - a[Z] * b[Y];
            ret[Y] = -a[X] * b[Z] + a[Z] * b[X];
            ret[Z] = a[X] * b[Y] - a[Y] * b[X];
        }

        // 向量原地加法
        // 传入两个数组 a_local 和 b，将 a_local + b 的结果放入 a_local 中
        __device__ __forceinline__ void vct_add_local(FLOAT *a_local, FLOAT *b) {
            a_local[X] += b[X];
            a_local[Y] += b[Y];
            a_local[Z] += b[Z];
        }

        // 向量原地加法
        // 函数意义同上，但是完成的是 6 维加法
        __device__ __forceinline__ void vct6_add_local(FLOAT *a_local, FLOAT *b) {
            a_local[X] += b[X];
            a_local[Y] += b[Y];
            a_local[Z] += b[Z];
            a_local[X+DIM] += b[X+DIM];
            a_local[Y+DIM] += b[Y+DIM];
            a_local[Z+DIM] += b[Z+DIM];
        }

        // 向量加法
        // 传入 a b ret 三个数组，将 a + b 的结果传入 ret 中
        __device__ __forceinline__ void vct_add(FLOAT *a, FLOAT *b, FLOAT *ret) {
            ret[X] = a[X] + b[X];
            ret[Y] = a[Y] + b[Y];
            ret[Z] = a[Z] + b[Z];
        }

        // 向量加法
        // 函数意义同上，但是完成的是 6 维加法
        __device__ __forceinline__ void vct6_add(FLOAT *a, FLOAT *b, FLOAT *ret) {
            ret[X] = a[X] + b[X];
            ret[Y] = a[Y] + b[Y];
            ret[Z] = a[Z] + b[Z];
            ret[X+DIM] = a[X+DIM] + b[X+DIM];
            ret[Y+DIM] = a[Y+DIM] + b[Y+DIM];
            ret[Z+DIM] = a[Z+DIM] + b[Z+DIM];
        }

        // 向量*常数，原地操作
        __device__ __forceinline__ void vct_dot_a_v(FLOAT a, FLOAT *v) {
            v[X] *= a;
            v[Y] *= a;
            v[Z] *= a;
        }

        // 向量*常数，原地操作。六维
        __device__ __forceinline__ void vct6_dot_a_v(FLOAT a, FLOAT *v) {
            v[X] *= a;
            v[Y] *= a;
            v[Z] *= a;
            v[X+DIM] *= a;
            v[Y+DIM] *= a;
            v[Z+DIM] *= a;
        }

        // 向量*常数。结果写入 ret 中
        __device__ __forceinline__ void vct_dot_a_v_ret(FLOAT a, FLOAT *v, FLOAT *ret) {
            ret[X] = v[X] * a;
            ret[Y] = v[Y] * a;
            ret[Z] = v[Z] * a;
        }

        // 向量*常数。六维。结果写入 ret 中
        __device__ __forceinline__ void vct6_dot_a_v_ret(FLOAT a, FLOAT *v, FLOAT *ret) {
            ret[X] = v[X] * a;
            ret[Y] = v[Y] * a;
            ret[Z] = v[Z] * a;
            ret[X+DIM] = v[X+DIM] * a;
            ret[Y+DIM] = v[Y+DIM] * a;
            ret[Z+DIM] = v[Z+DIM] * a;
        }

        // 向量内积，直接返回标量值
        __device__ __forceinline__ FLOAT vct_dot_v_v(FLOAT *v,FLOAT *w){
            return v[X] * w[X] + v[Y] * w[Y] + v[Z] * w[Z];
        }

        // 向量拷贝赋值，源 src，宿 des
        __device__ __forceinline__ void vct_copy(FLOAT *src, FLOAT *des) {
            des[X] = src[X];
            des[Y] = src[Y];
            des[Z] = src[Z];
        }

        // 向量拷贝赋值，六维，源 src，宿 des
        __device__ __forceinline__ void vct6_copy(FLOAT *src, FLOAT *des) {
            des[X] = src[X];
            des[Y] = src[Y];
            des[Z] = src[Z];
            des[X+DIM] = src[X+DIM];
            des[Y+DIM] = src[Y+DIM];
            des[Z+DIM] = src[Z+DIM];
        }

        // 求向量长度，直接返回计算结果
        __device__ __forceinline__ FLOAT vct_len(FLOAT *v) {

            // 根据 32 位还是 64 位有不同的实现
            #ifdef FLOAT32
            return sqrtf(v[X] * v[X] + v[Y] * v[Y] + v[Z] * v[Z]);
            #else
            return sqrt(v[X] * v[X] + v[Y] * v[Y] + v[Z] * v[Z]);
            #endif
        }

        // 将矢量 v 置为 0
        __device__ __forceinline__ void vct_zero(FLOAT *v) {
            v[X] = 0.0;
            v[Y] = 0.0;
            v[Z] = 0.0;
        }

        // 打印矢量，一般用于 debug
        __device__ __forceinline__ void vct_print(FLOAT *v) {
            #ifdef FLOAT32
            printf("%.15f, %.15f, %.15f\\n", v[X], v[Y], v[Z]);
            #else
            printf("%.15lf, %.15lf, %.15lf\\n", v[X], v[Y], v[Z]);
            #endif
        }

        // 打印六维矢量，一般用于 debug
        __device__ __forceinline__ void vct6_print(FLOAT *v) {
            #ifdef FLOAT32
            printf("%.15f, %.15f, %.15f, %.15f, %.15f, %.15f\\n", v[X], v[Y], v[Z], v[X+DIM], v[Y+DIM], v[Z+DIM]);
            #else
            printf("%.15lf, %.15lf, %.15lf, %.15lf, %.15lf, %.15lf\\n", v[X], v[Y], v[Z] ,v[X+DIM], v[Y+DIM], v[Z+DIM]);
            #endif
        }

        // 矢量减法，结果放在 ret 中
        __device__ __forceinline__ void vct_sub(FLOAT *a, FLOAT *b, FLOAT *ret) {
            ret[X] = a[X] - b[X];
            ret[Y] = a[Y] - b[Y];
            ret[Z] = a[Z] - b[Z];
        }

        """

        cuda_code_04_dB = """
        // 计算电流元在 p 点产生的磁场
        // 其中 p0 表示电流元的位置
        // kl 含义见下
        // 返回值放在 ret 中
        // 
        // 原本电流元的计算公式如下：
        // dB = (miu0/4pi) * Idl × r / (r^3)
        // 其中 r = p - p0，p0 是电流元的位置
        // 
        // 如果考虑极小一段电流（起点s0，终点s1）则产生的磁场为
        // ΔB = (miu0/4pi) * I * (s1-s2)*r / (r^3)
        // 同样的，r = p - p0，p0 = (s1+s2)/2
        //
        // 因为 (miu0/4pi) * I * (s1-s2) 整体已知，所以提前计算为 kl
        // p0 提前已知，即 (s1+s2)/2，也提前给出
        // 这样可以减少无意义的重复计算
        //
        // 补充：坐标均是全局坐标
        __device__ __forceinline__ void dB(FLOAT *kl, FLOAT *p0, FLOAT *p, FLOAT *ret){
            FLOAT r[DIM];
            FLOAT rr;

            vct_sub(p, p0, r); // r = p - p0

            rr = vct_len(r); // rr = abs(r)

            rr = rr*rr*rr; // rr = rr^3

            vct_cross(kl, r, ret); // ret = kl × r

            vct_dot_a_v(1.0/rr, ret); // ret = (kl × r)/(rr^3)
        }

        // 计算所有的电流元在 p 点产生的磁场
        // number 表示电流元数目
        // kls 每 DIM = 3 组表示一个 kl
        // p0s 每 DIM = 3 组表示一个 p0
        // shared_ret 是一个 shared 量，保存返回值
        // 调用该方法后，应进行同步处理  __syncthreads();
        __device__ void current_element_B(FLOAT *kls, FLOAT *p0s, int number, FLOAT *p, FLOAT *shared_ret){
            int tid = threadIdx.x; // 0-1023 (decide by BLOCK_DIM_X)
            FLOAT db[DIM];
            __shared__ FLOAT s_dbs[DIM*BLOCK_DIM_X];

            vct_zero(s_dbs + tid*DIM);

            // 计算每个电流元产生的磁场
            for(int i = tid*DIM; i < number*DIM; i += BLOCK_DIM_X*DIM){
                dB(kls + i, p0s + i, p, db);
                vct_add_local(s_dbs + tid*DIM, db);
            }
            
            // 规约求和（from https://www.bilibili.com/video/BV15E411x7yT）
            for(int step = BLOCK_DIM_X>>1; step >= 1; step>>=1){
                __syncthreads(); // 求和前同步
                if(tid<step) vct_add_local(s_dbs + tid * DIM, s_dbs + (tid + step) * DIM);
            }

            if(tid == 0) vct_copy(s_dbs, shared_ret);
        }

        """

        cuda_code_05_QS = """
        // 计算 QS 在 p 点产生的磁场
        // origin xi yi zi 分别是 QS 的局部坐标系
        // 这个函数只需要单线程计算
        __device__ __forceinline__ void magnet_at_qs(FLOAT *origin, FLOAT *xi, FLOAT *yi, FLOAT *zi, 
                FLOAT length, FLOAT gradient, FLOAT second_gradient, FLOAT aper_r, FLOAT *p, FLOAT* ret){
            FLOAT temp1[DIM];
            FLOAT temp2[DIM];

            vct_sub(p, origin, temp1); // temp1 = p - origin
            temp2[X] = vct_dot_v_v(xi, temp1);
            temp2[Y] = vct_dot_v_v(yi, temp1);
            temp2[Z] = vct_dot_v_v(zi, temp1); // 这时 temp2 就是全局坐标 p 点在 QS 局部坐标系中的坐标

            vct_zero(ret);

            if(temp2[Z]<0 || temp2[Z]>length){
                return; // 无磁场
            }else{
                if(
                    temp2[X] > aper_r ||
                    temp2[X] < -aper_r ||
                    temp2[Y] > aper_r ||
                    temp2[Y] < -aper_r ||
                    #ifdef FLOAT32
                    sqrtf(temp2[X]*temp2[X]+temp2[Y]*temp2[Y]) > aper_r
                    #else
                    sqrt(temp2[X]*temp2[X]+temp2[Y]*temp2[Y]) > aper_r
                    #endif
                ){
                    return; // 无磁场
                }else{
                    temp1[X] = gradient * temp2[Y] + second_gradient * (temp2[X] * temp2[Y]);
                    temp1[Y] = gradient * temp2[X] + 0.5 * second_gradient * (temp2[X] * temp2[X] - temp2[Y] * temp2[Y]);

                    vct_dot_a_v_ret(temp1[X], xi, ret);
                    vct_dot_a_v_ret(temp1[Y], yi, temp2);
                    vct_add_local(ret, temp2);
                }
            }
        }

        // 计算 QS 在 p 点产生的磁场
        // 上函数的 qs_data 版本
        __device__ __forceinline__ void magnet_at_qs_date(FLOAT *qs_data, FLOAT *p, FLOAT* ret){
            magnet_at_qs(
                    qs_data, // origin
                    qs_data + 3, //xi
                    qs_data + 6, //yi
                    qs_data + 9, //zi
                    *(qs_data + 12), // len
                    *(qs_data + 13), // g
                    *(qs_data + 14), // sg
                    *(qs_data + 15), // aper r
                    p, ret
            );
        }

        // 计算多个 qs 磁铁的磁场，并行计算
        // 需要同步
        __device__ void magnet_at_qs_dates(FLOAT *qs_datas, int qs_number, FLOAT *p, FLOAT* shared_ret){
            int tid = threadIdx.x; // 0-1023 (decide by BLOCK_DIM_X)
            FLOAT db[DIM];
            __shared__ FLOAT s_dbs[DIM*BLOCK_DIM_X];

            vct_zero(s_dbs + tid*DIM);

            // 计算每个 qs 磁铁产生的磁场
            for(int i = tid; i < qs_number; i += BLOCK_DIM_X){
                magnet_at_qs_date(
                    qs_datas + i * QS_DATA_LENGTH, p, db
                );
                // printf("%d %d\\n",qs_number,tid);
                // vct_print(db);
                vct_add_local(s_dbs + tid*DIM, db);
            }
            
            // 规约求和（from https://www.bilibili.com/video/BV15E411x7yT）
            for(int step = BLOCK_DIM_X>>1; step >= 1; step>>=1){
                __syncthreads(); // 求和前同步
                if(tid<step) vct_add_local(s_dbs + tid * DIM, s_dbs + (tid + step) * DIM);
            }

            if(tid == 0) vct_copy(s_dbs, shared_ret);
        }
        
        """

        cuda_code_06_magnet_at = """
        // 整个束线在 p 点产生得磁场（只有一个 QS 磁铁！）
        // FLOAT *kls, FLOAT* p0s, int current_element_number 和 CCT 电流元相关
        // FLOAT *qs_data 表示 QS 磁铁所有参数，分别是局部坐标系原点origin,三个轴xi yi zi，长度 梯度 二阶梯度 孔径
        // p 表示要求磁场得全局坐标点
        // shared_ret 表示磁场返回值（应该是一个 __shared__）
        // 本方法已经完成同步了，不用而外调用 __syncthreads();
        __device__ void magnet_with_single_qs(FLOAT *kls, FLOAT* p0s, int current_element_number, 
                FLOAT *qs_data, FLOAT *p, FLOAT *shared_ret){
            int tid = threadIdx.x;
            FLOAT qs_magnet[DIM];
            
            current_element_B(kls, p0s, current_element_number, p, shared_ret);
            __syncthreads(); // 块内同步

            
            if(tid == 0){
                // 计算 QS 的磁场确实不能并行
                // 也没有必要让每个线程都重复计算一次
                // 虽然两次同步有点麻烦，但至少只有一个线程束参与运行
                magnet_at_qs(
                    qs_data, // origin
                    qs_data + 3, //xi
                    qs_data + 6, //yi
                    qs_data + 9, //zi
                    *(qs_data + 12), // len
                    *(qs_data + 13), // g
                    *(qs_data + 14), // sg
                    *(qs_data + 15), // aper r
                    p, qs_magnet
                );

                vct_add_local(shared_ret, qs_magnet);
            }
            __syncthreads(); // 块内同步
        }

        __device__ void magnet_with_multi_qs(FLOAT *kls, FLOAT* p0s, int current_element_number, 
                FLOAT *qs_datas, int qs_number, FLOAT *p, FLOAT *shared_ret){
            int tid = threadIdx.x;
            __shared__ FLOAT s_qs_magnet[DIM];
            
            current_element_B(kls, p0s, current_element_number, p, shared_ret);
            __syncthreads(); // 块内同步

            // 计算多个 qs 磁铁的磁场
            magnet_at_qs_dates(qs_datas, qs_number, p, s_qs_magnet);
            if(tid == 0){
                vct_add_local(shared_ret, s_qs_magnet);
            }
            __syncthreads(); // 块内同步
        }
        """

        cuda_code_07_runge_kutta4 = """
        // runge_kutta4 代码和 cctpy 中的 runge_kutta4 一模一样
        // Y0 数组长度为 6
        // Y0 会发生变化，既是输入也是输出
        // 这个函数单线程运行
        // void (*call)(FLOAT,FLOAT*,FLOAT*) 表示 tn Yn 到 Yn+1 的转移，实际使用中还会带更多参数（C 语言没有闭包）
        // 所以这个函数仅仅是原型
        __device__ void runge_kutta4(FLOAT t0, FLOAT t_end, FLOAT *Y0, void (*call)(FLOAT,FLOAT*,FLOAT*), FLOAT dt){
            #ifdef FLOAT32
            int number = (int)(ceilf((t_end - t0) / dt));
            #else
            int number = (int)(ceil((t_end - t0) / dt));
            #endif

            // 重新定义了 dt
            dt = (t_end - t0) / ((FLOAT)(number));
            FLOAT k1[DIM*2];
            FLOAT k2[DIM*2];
            FLOAT k3[DIM*2];
            FLOAT k4[DIM*2];
            FLOAT temp[DIM*2];

            for(int ignore = 0; ignore < number; ignore++){
                (*call)(t0, Y0, k1);

                vct6_dot_a_v_ret(dt / 2., k1, temp); // temp = dt / 2 * k1
                vct6_add_local(temp, Y0); // temp =  Y0 + temp
                (*call)(t0 + dt / 2., temp, k2);


                vct6_dot_a_v_ret(dt / 2., k2, temp); // temp = dt / 2 * k2
                vct6_add_local(temp, Y0); // temp =  Y0 + temp
                (*call)(t0 + dt / 2., temp, k3);

                vct6_dot_a_v_ret(dt, k3, temp); // temp = dt * k3
                vct6_add_local(temp, Y0); // temp =  Y0 + temp
                (*call)(t0 + dt, temp, k4);

                t0 += dt;
                
                vct6_add(k1, k4, temp); // temp = k1 + k4
                vct6_dot_a_v(2.0, k2);
                vct6_dot_a_v(2.0, k3);
                vct6_add(k2, k3, k1); // k1 已经没用了，所以装 k1 = k2 + k3
                vct6_add_local(temp, k1);
                vct6_dot_a_v(dt / 6.0, temp);
                vct6_add_local(Y0, temp);
                // Y0 += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
            }
        }

        """

        cuda_code_08_run_only = """
        // runge_kutta4_for_magnet_with_single_qs 函数用到的回调
        // FLOAT t0, FLOAT* Y0, FLOAT* Y1 微分计算
        // 其中 Y = [P, V]
        // FLOAT k = particle[E] / particle[RM]; // k: float = particle.e / particle.relativistic_mass
        // FLOAT *kls, FLOAT* p0s, int current_element_number, 表示所有电流元
        // FLOAT *qs_data 表示一个 QS 磁铁
        __device__ void callback_for_runge_kutta4_for_magnet_with_single_qs(
            FLOAT t0, FLOAT* Y0, FLOAT* Y1, FLOAT k, 
            FLOAT *kls, FLOAT* p0s, int current_element_number, 
            FLOAT *qs_data
        )
        {
            int tid = threadIdx.x;
            __shared__ FLOAT m[DIM]; // 磁场
            magnet_with_single_qs(kls, p0s, current_element_number, qs_data, Y0, m); //Y0 只使用前3项，表示位置。已同步

            if(tid == 0){ // 单线程完成即可
                // ------------ 以下两步计算加速度，写入 Y1 + 3 中 ----------
                // Y0 + 3 是原速度 v
                // Y1 + 3 用于存加速度，即 v × m，还没有乘 k = e/rm
                vct_cross(Y0 + 3, m, Y1 + 3);
                vct_dot_a_v(k,  Y1 + 3); // 即 (v × m) * a，并且把积存在 Y1 + 3 中

                // ------------- 以下把原速度复制到 Y1 中 ------------
                vct_copy(Y0 + 3, Y1); // Y0 中后三项，速度。复制到 Y1 的前3项
            }

            __syncthreads(); // 块内同步
        }

        // 多 qs
        __device__ void callback_for_runge_kutta4_for_magnet_with_multi_qs(
            FLOAT t0, FLOAT* Y0, FLOAT* Y1, FLOAT k, 
            FLOAT *kls, FLOAT* p0s, int current_element_number, 
            FLOAT *qs_datas, int qs_number
        )
        {
            int tid = threadIdx.x;
            __shared__ FLOAT m[DIM]; // 磁场
            magnet_with_multi_qs(kls, p0s, current_element_number, qs_datas, qs_number, Y0, m); //Y0 只使用前3项，表示位置。已同步

            if(tid == 0){ // 单线程完成即可
                // ------------ 以下两步计算加速度，写入 Y1 + 3 中 ----------
                // Y0 + 3 是原速度 v
                // Y1 + 3 用于存加速度，即 v × m，还没有乘 k = e/rm
                vct_cross(Y0 + 3, m, Y1 + 3);
                vct_dot_a_v(k,  Y1 + 3); // 即 (v × m) * a，并且把积存在 Y1 + 3 中

                // ------------- 以下把原速度复制到 Y1 中 ------------
                vct_copy(Y0 + 3, Y1); // Y0 中后三项，速度。复制到 Y1 的前3项
            }

            __syncthreads(); // 块内同步
        }

        // 单个粒子跟踪
        // runge_kutta4 函数用于 magnet_with_single_qs 的版本，即粒子跟踪
        // Y0 即是 [P, v] 粒子位置、粒子速度
        // void (*call)(FLOAT,FLOAT*,FLOAT*,FLOAT,FLOAT*,FLOAT*,int,FLOAT*) 改为 callback_for_runge_kutta4_for_magnet_with_single_qs
        // 前 3 项 FLOAT,FLOAT*,FLOAT* 和函数原型 runge_kutta4 函数一样，即 t0 Y0 Y1
        // 第 4 项，表示 k = particle[E] / particle[RM]; // k: float = particle.e / particle.relativistic_mass
        // 第 567 项，FLOAT*,FLOAT*,int 表示所有电流源，FLOAT *kls, FLOAT* p0s, int current_element_number
        // 最后一项，表示 qs_data
        // particle 表示粒子 (px0, py1, pz2, vx3, vy4, vz5, rm6, e7, speed8, distance9) len = 10
        /*__global__*/ __device__ void track_for_magnet_with_single_qs(FLOAT *distance, FLOAT *footstep,
                FLOAT *kls, FLOAT* p0s, int *current_element_number, 
                FLOAT *qs_data, FLOAT *particle)
        {
            int tid = threadIdx.x;
            FLOAT t0 = 0.0; // 开始时间为 0
            FLOAT t_end = (*distance) / particle[SPEED]; // 用时 = 距离/速率
            
            #ifdef FLOAT32
            int number = (int)(ceilf( (*distance) / (*footstep) ));
            #else
            int number = (int)(ceil( (*distance) / (*footstep)));
            #endif

            FLOAT dt = (t_end - t0) / ((FLOAT)(number));
            FLOAT k = particle[E] / particle[RM]; // k: float = particle.e / particle.relativistic_mass

            __shared__ FLOAT Y0[DIM*2]; // Y0 即是 [P, v] 粒子位置、粒子速度，就是 particle 前两项
            __shared__ FLOAT k1[DIM*2];
            __shared__ FLOAT k2[DIM*2];
            __shared__ FLOAT k3[DIM*2];
            __shared__ FLOAT k4[DIM*2];
            __shared__ FLOAT temp[DIM*2];

            if(tid == 0){
                vct6_copy(particle, Y0); // 写 Y0
            }

            for(int ignore = 0; ignore < number; ignore++){
                __syncthreads(); // 循环前同步

                callback_for_runge_kutta4_for_magnet_with_single_qs(t0, Y0, k1, k, kls, p0s, *current_element_number, qs_data); // 已同步


                if(tid == 0){
                    vct6_dot_a_v_ret(dt / 2., k1, temp); // temp = dt / 2 * k1
                    vct6_add_local(temp, Y0); // temp =  Y0 + temp
                }
                __syncthreads();

                callback_for_runge_kutta4_for_magnet_with_single_qs(t0 + dt / 2., temp, k2, k, kls, p0s, *current_element_number, qs_data);

                if(tid == 0){
                    vct6_dot_a_v_ret(dt / 2., k2, temp); // temp = dt / 2 * k2
                    vct6_add_local(temp, Y0); // temp =  Y0 + temp
                }
                __syncthreads();

                callback_for_runge_kutta4_for_magnet_with_single_qs(t0 + dt / 2., temp, k3, k, kls, p0s, *current_element_number, qs_data);

                if(tid == 0){
                    vct6_dot_a_v_ret(dt, k3, temp); // temp = dt * k3
                    vct6_add_local(temp, Y0); // temp =  Y0 + temp
                }
                __syncthreads();

                callback_for_runge_kutta4_for_magnet_with_single_qs(t0 + dt, temp, k4, k, kls, p0s, *current_element_number, qs_data);

                t0 += dt;

                if(tid == 0){
                    vct6_add(k1, k4, temp); // temp = k1 + k4
                    vct6_dot_a_v(2.0, k2);
                    vct6_dot_a_v(2.0, k3);
                    vct6_add(k2, k3, k1); // k1 已经没用了，所以装 k1 = k2 + k3
                    vct6_add_local(temp, k1);
                    vct6_dot_a_v(dt / 6.0, temp);
                    vct6_add_local(Y0, temp);
                    // Y0 += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
                }
            }

            // 写回 particle
            if(tid == 0){
                vct6_copy(Y0 ,particle); // 写 Y0
                particle[DISTANCE] = *distance;
            }

            __syncthreads();
        }

        // 上函数的 global 版本
        __global__ void track_for_magnet_with_single_qs_g(FLOAT *distance, FLOAT *footstep,
                FLOAT *kls, FLOAT* p0s, int *current_element_number, 
                FLOAT *qs_data, FLOAT *particle)
        {
            int tid = threadIdx.x;
            FLOAT t0 = 0.0; // 开始时间为 0
            FLOAT t_end = (*distance) / particle[SPEED]; // 用时 = 距离/速率
            
            #ifdef FLOAT32
            int number = (int)(ceilf( (*distance) / (*footstep) ));
            #else
            int number = (int)(ceil( (*distance) / (*footstep)));
            #endif

            FLOAT dt = (t_end - t0) / ((FLOAT)(number));
            FLOAT k = particle[E] / particle[RM]; // k: float = particle.e / particle.relativistic_mass

            __shared__ FLOAT Y0[DIM*2]; // Y0 即是 [P, v] 粒子位置、粒子速度，就是 particle 前两项
            __shared__ FLOAT k1[DIM*2];
            __shared__ FLOAT k2[DIM*2];
            __shared__ FLOAT k3[DIM*2];
            __shared__ FLOAT k4[DIM*2];
            __shared__ FLOAT temp[DIM*2];

            if(tid == 0){
                vct6_copy(particle, Y0); // 写 Y0
            }

            for(int ignore = 0; ignore < number; ignore++){
                __syncthreads(); // 循环前同步

                callback_for_runge_kutta4_for_magnet_with_single_qs(t0, Y0, k1, k, kls, p0s, *current_element_number, qs_data); // 已同步


                if(tid == 0){
                    vct6_dot_a_v_ret(dt / 2., k1, temp); // temp = dt / 2 * k1
                    vct6_add_local(temp, Y0); // temp =  Y0 + temp
                }
                __syncthreads();

                callback_for_runge_kutta4_for_magnet_with_single_qs(t0 + dt / 2., temp, k2, k, kls, p0s, *current_element_number, qs_data);

                if(tid == 0){
                    vct6_dot_a_v_ret(dt / 2., k2, temp); // temp = dt / 2 * k2
                    vct6_add_local(temp, Y0); // temp =  Y0 + temp
                }
                __syncthreads();

                callback_for_runge_kutta4_for_magnet_with_single_qs(t0 + dt / 2., temp, k3, k, kls, p0s, *current_element_number, qs_data);

                if(tid == 0){
                    vct6_dot_a_v_ret(dt, k3, temp); // temp = dt * k3
                    vct6_add_local(temp, Y0); // temp =  Y0 + temp
                }
                __syncthreads();

                callback_for_runge_kutta4_for_magnet_with_single_qs(t0 + dt, temp, k4, k, kls, p0s, *current_element_number, qs_data);

                t0 += dt;

                if(tid == 0){
                    vct6_add(k1, k4, temp); // temp = k1 + k4
                    vct6_dot_a_v(2.0, k2);
                    vct6_dot_a_v(2.0, k3);
                    vct6_add(k2, k3, k1); // k1 已经没用了，所以装 k1 = k2 + k3
                    vct6_add_local(temp, k1);
                    vct6_dot_a_v(dt / 6.0, temp);
                    vct6_add_local(Y0, temp);
                    // Y0 += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
                }
            }

            // 写回 particle
            if(tid == 0){
                vct6_copy(Y0 ,particle); // 写 Y0
                particle[DISTANCE] = *distance;
            }

            __syncthreads();
        }

        // -------------------------  多 qs 版本------------------------------------------ //
        // 单个粒子跟踪，多 qs 版本
        /*__global__*/ __device__ void track_for_magnet_with_multi_qs(FLOAT *distance, FLOAT *footstep,
                FLOAT *kls, FLOAT* p0s, int *current_element_number, 
                FLOAT *qs_datas, int *qs_number, FLOAT *particle)
        {
            int tid = threadIdx.x;
            FLOAT t0 = 0.0; // 开始时间为 0
            FLOAT t_end = (*distance) / particle[SPEED]; // 用时 = 距离/速率
            
            #ifdef FLOAT32
            int number = (int)(ceilf( (*distance) / (*footstep) ));
            #else
            int number = (int)(ceil( (*distance) / (*footstep)));
            #endif

            FLOAT dt = (t_end - t0) / ((FLOAT)(number));
            FLOAT k = particle[E] / particle[RM]; // k: float = particle.e / particle.relativistic_mass

            __shared__ FLOAT Y0[DIM*2]; // Y0 即是 [P, v] 粒子位置、粒子速度，就是 particle 前两项
            __shared__ FLOAT k1[DIM*2];
            __shared__ FLOAT k2[DIM*2];
            __shared__ FLOAT k3[DIM*2];
            __shared__ FLOAT k4[DIM*2];
            __shared__ FLOAT temp[DIM*2];

            if(tid == 0){
                vct6_copy(particle, Y0); // 写 Y0
            }

            for(int ignore = 0; ignore < number; ignore++){
                __syncthreads(); // 循环前同步

                callback_for_runge_kutta4_for_magnet_with_multi_qs(t0, Y0, k1, k, kls, p0s, *current_element_number, qs_datas, *qs_number); // 已同步


                if(tid == 0){
                    vct6_dot_a_v_ret(dt / 2., k1, temp); // temp = dt / 2 * k1
                    vct6_add_local(temp, Y0); // temp =  Y0 + temp
                }
                __syncthreads();

                callback_for_runge_kutta4_for_magnet_with_multi_qs(t0 + dt / 2., temp, k2, k, kls, p0s, *current_element_number, qs_datas, *qs_number);

                if(tid == 0){
                    vct6_dot_a_v_ret(dt / 2., k2, temp); // temp = dt / 2 * k2
                    vct6_add_local(temp, Y0); // temp =  Y0 + temp
                }
                __syncthreads();

                callback_for_runge_kutta4_for_magnet_with_multi_qs(t0 + dt / 2., temp, k3, k, kls, p0s, *current_element_number, qs_datas, *qs_number);

                if(tid == 0){
                    vct6_dot_a_v_ret(dt, k3, temp); // temp = dt * k3
                    vct6_add_local(temp, Y0); // temp =  Y0 + temp
                }
                __syncthreads();

                callback_for_runge_kutta4_for_magnet_with_multi_qs(t0 + dt, temp, k4, k, kls, p0s, *current_element_number, qs_datas, *qs_number);

                t0 += dt;

                if(tid == 0){
                    vct6_add(k1, k4, temp); // temp = k1 + k4
                    vct6_dot_a_v(2.0, k2);
                    vct6_dot_a_v(2.0, k3);
                    vct6_add(k2, k3, k1); // k1 已经没用了，所以装 k1 = k2 + k3
                    vct6_add_local(temp, k1);
                    vct6_dot_a_v(dt / 6.0, temp);
                    vct6_add_local(Y0, temp);
                    // Y0 += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
                }
            }

            // 写回 particle
            if(tid == 0){
                vct6_copy(Y0 ,particle); // 写 Y0
                particle[DISTANCE] = *distance;
            }

            __syncthreads();
        }

        // 上函数的 global 版本，多 qs 版本
        __global__ void track_for_magnet_with_multi_qs_g(FLOAT *distance, FLOAT *footstep,
                FLOAT *kls, FLOAT* p0s, int *current_element_number, 
                FLOAT *qs_datas, int *qs_number, FLOAT *particle)
        {
            int tid = threadIdx.x;
            FLOAT t0 = 0.0; // 开始时间为 0
            FLOAT t_end = (*distance) / particle[SPEED]; // 用时 = 距离/速率
            
            #ifdef FLOAT32
            int number = (int)(ceilf( (*distance) / (*footstep) ));
            #else
            int number = (int)(ceil( (*distance) / (*footstep)));
            #endif

            FLOAT dt = (t_end - t0) / ((FLOAT)(number));
            FLOAT k = particle[E] / particle[RM]; // k: float = particle.e / particle.relativistic_mass

            __shared__ FLOAT Y0[DIM*2]; // Y0 即是 [P, v] 粒子位置、粒子速度，就是 particle 前两项
            __shared__ FLOAT k1[DIM*2];
            __shared__ FLOAT k2[DIM*2];
            __shared__ FLOAT k3[DIM*2];
            __shared__ FLOAT k4[DIM*2];
            __shared__ FLOAT temp[DIM*2];

            if(tid == 0){
                vct6_copy(particle, Y0); // 写 Y0
            }

            for(int ignore = 0; ignore < number; ignore++){
                __syncthreads(); // 循环前同步

                callback_for_runge_kutta4_for_magnet_with_multi_qs(t0, Y0, k1, k, kls, p0s, *current_element_number, qs_datas, *qs_number); // 已同步


                if(tid == 0){
                    vct6_dot_a_v_ret(dt / 2., k1, temp); // temp = dt / 2 * k1
                    vct6_add_local(temp, Y0); // temp =  Y0 + temp
                }
                __syncthreads();

                callback_for_runge_kutta4_for_magnet_with_multi_qs(t0 + dt / 2., temp, k2, k, kls, p0s, *current_element_number, qs_datas, *qs_number);

                if(tid == 0){
                    vct6_dot_a_v_ret(dt / 2., k2, temp); // temp = dt / 2 * k2
                    vct6_add_local(temp, Y0); // temp =  Y0 + temp
                }
                __syncthreads();

                callback_for_runge_kutta4_for_magnet_with_multi_qs(t0 + dt / 2., temp, k3, k, kls, p0s, *current_element_number, qs_datas, *qs_number);

                if(tid == 0){
                    vct6_dot_a_v_ret(dt, k3, temp); // temp = dt * k3
                    vct6_add_local(temp, Y0); // temp =  Y0 + temp
                }
                __syncthreads();

                callback_for_runge_kutta4_for_magnet_with_multi_qs(t0 + dt, temp, k4, k, kls, p0s, *current_element_number, qs_datas, *qs_number);

                t0 += dt;

                if(tid == 0){
                    vct6_add(k1, k4, temp); // temp = k1 + k4
                    vct6_dot_a_v(2.0, k2);
                    vct6_dot_a_v(2.0, k3);
                    vct6_add(k2, k3, k1); // k1 已经没用了，所以装 k1 = k2 + k3
                    vct6_add_local(temp, k1);
                    vct6_dot_a_v(dt / 6.0, temp);
                    vct6_add_local(Y0, temp);
                    // Y0 += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
                }
            }

            // 写回 particle
            if(tid == 0){
                vct6_copy(Y0 ,particle); // 写 Y0
                particle[DISTANCE] = *distance;
            }

            __syncthreads();
        }
        
        """

        cuda_code_09_run_multi_particle = """
        // 多粒子跟踪，串行
        __device__ void track_multi_particle_for_magnet_with_single_qs(FLOAT *distance, FLOAT *footstep,
                FLOAT *kls, FLOAT *p0s, int *current_element_number, 
                FLOAT *qs_data, FLOAT *particle, int *particle_number)
        {
            for(int i = 0; i< (*particle_number);i++){
                track_for_magnet_with_single_qs(distance, footstep, kls, p0s, 
                    current_element_number, qs_data, particle + i * PARTICLE_DIM);
            }
        }

        __global__ void track_multi_particle_for_magnet_with_single_qs_g(FLOAT *distance, FLOAT *footstep,
                FLOAT *kls, FLOAT *p0s, int *current_element_number, 
                FLOAT *qs_data, FLOAT *particle, int *particle_number)
        {
            for(int i = 0; i< (*particle_number);i++){
                track_for_magnet_with_single_qs(distance, footstep, kls, p0s, 
                    current_element_number, qs_data, particle + i * PARTICLE_DIM);
            }
        }


        // 多 qs 版本
        __device__ void track_multi_particle_for_magnet_with_multi_qs(FLOAT *distance, FLOAT *footstep,
                FLOAT *kls, FLOAT *p0s, int *current_element_number, 
                FLOAT *qs_datas, int *qs_number, FLOAT *particle, int *particle_number)
        {
            for(int i = 0; i< (*particle_number);i++){
                track_for_magnet_with_multi_qs(distance, footstep, kls, p0s, 
                    current_element_number, qs_datas, qs_number, particle + i * PARTICLE_DIM);
            }
        }

        __global__ void track_multi_particle_for_magnet_with_multi_qs_g(FLOAT *distance, FLOAT *footstep,
                FLOAT *kls, FLOAT *p0s, int *current_element_number, 
                FLOAT *qs_datas, int *qs_number, FLOAT *particle, int *particle_number)
        {
            for(int i = 0; i< (*particle_number);i++){
                track_for_magnet_with_multi_qs(distance, footstep, kls, p0s, 
                    current_element_number, qs_datas, qs_number, particle + i * PARTICLE_DIM);
            }
        }
        """

        cuda_code_10_run_multi_particle_multi_beamline = """
        __global__ void track_multi_particle_beamline_for_magnet_with_single_qs(FLOAT *distance, FLOAT *footstep,
                FLOAT *kls, FLOAT *p0s, int *current_element_number, 
                FLOAT *qs_data, FLOAT *particle, int *particle_number)
        {
            int bid = blockIdx.x;
            track_multi_particle_for_magnet_with_single_qs(
                distance, // 全局相同
                footstep, // 全局相同

                kls + MAX_CURRENT_ELEMENT_NUMBER * DIM * bid,
                p0s + MAX_CURRENT_ELEMENT_NUMBER * DIM * bid, // 当前组电流元参数
                current_element_number + bid, // 当前组电流元数目

                qs_data + QS_DATA_LENGTH * bid, // 当前组 QS 参数

                particle + (*particle_number) * PARTICLE_DIM * bid, // 当前组粒子
                particle_number // 全局相同
            );
        }

        // 多 qs 版本
        __global__ void track_multi_particle_beamline_for_magnet_with_multi_qs(FLOAT *distance, FLOAT *footstep,
                FLOAT *kls, FLOAT *p0s, int *current_element_number, 
                FLOAT *qs_datas, int *qs_number, FLOAT *particle, int *particle_number)
        {
            int bid = blockIdx.x;
            track_multi_particle_for_magnet_with_multi_qs(
                distance, // 全局相同
                footstep, // 全局相同

                kls + MAX_CURRENT_ELEMENT_NUMBER * DIM * bid,
                p0s + MAX_CURRENT_ELEMENT_NUMBER * DIM * bid, // 当前组电流元参数
                current_element_number + bid, // 当前组电流元数目

                qs_datas + MAX_QS_DATAS_LENGTH * QS_DATA_LENGTH * bid , // 当前组 QS 参数
                qs_number + bid, // 当前组 QS 数目

                particle + (*particle_number) * PARTICLE_DIM * bid, // 当前组粒子
                particle_number // 全局相同
            );
        }
        """

        self.cuda_code: str = (
            cuda_code_00_include +
            cuda_code_01_float_type_define +
            cuda_code_02_define +
            cuda_code_03_vct_functions +
            cuda_code_04_dB +
            cuda_code_05_QS +
            cuda_code_06_magnet_at +
            cuda_code_07_runge_kutta4 +
            cuda_code_08_run_only +
            cuda_code_09_run_multi_particle +
            cuda_code_10_run_multi_particle_multi_beamline
        )

    def print_cuda_code(self) -> None:
        """
        打印cuda代码，供整体分析
        """
        print(self.cuda_code)

    def vct_length(self, p3: P3):
        """
        测试用函数，计算矢量长度
        示例：
        ga32 = GPU_ACCELERATOR(float_number_type=GPU_ACCELERATOR.FLOAT32)
        ga64 = GPU_ACCELERATOR(float_number_type=GPU_ACCELERATOR.FLOAT64)
        v = P3(1,1,1)
        print(f"diff={ga32.vct_length(v) - v.length()}") # diff=-3.1087248775207854e-08
        print(f"diff={ga64.vct_length(v) - v.length()}") # diff=0.0
        """
        if self.cpu_mode:
            return p3.length()

        code = """
        __global__ void vl(FLOAT* v, FLOAT* ret){
            *ret = vct_len(v);
        }

        """

        mod = SourceModule(self.cuda_code + code)

        vl = mod.get_function("vl")

        ret = numpy.empty((1,), dtype=self.numpy_dtype)

        vl(drv.In(p3.to_numpy_ndarry3(numpy_dtype=self.numpy_dtype)),
           drv.Out(ret), grid=(1, 1, 1), block=(1, 1, 1))

        return float(ret[0])

    def vct_print(self, p3: P3):
        """
        测试用函数，打印矢量
        示例：
        ga32 = GPU_ACCELERATOR(float_number_type=GPU_ACCELERATOR.FLOAT32)
        ga64 = GPU_ACCELERATOR(float_number_type=GPU_ACCELERATOR.FLOAT64)
        v = P3(1/3, 1/6, 1/7)
        ga32.vct_print(v)
        ga64.vct_print(v)
        >>>
        0.333333343267441, 0.166666671633720, 0.142857149243355
        0.333333333333333, 0.166666666666667, 0.142857142857143
        """
        if self.cpu_mode:
            print(p3)

        code = """
        __global__ void vp(FLOAT* v){
            vct_print(v);
        }

        """

        mod = SourceModule(self.cuda_code + code)

        vp = mod.get_function("vp")

        vp(drv.In(p3.to_numpy_ndarry3(numpy_dtype=self.numpy_dtype)),
           grid=(1, 1, 1), block=(1, 1, 1))

    def current_element_B(self, kls: numpy.ndarray, p0s: numpy.ndarray, number: int, p: P3):
        """
        计算电流元集合，在 p 点产生的磁场
        对比代码如下：
        cct = CCT(
            local_coordinate_system=LocalCoordinateSystem.global_coordinate_system(),
            big_r=1*M, small_r=30*MM, bending_angle=30, tilt_angles=[30],
            winding_number=30, current=1000,
            starting_point_in_ksi_phi_coordinate=P2(0,0),
            end_point_in_ksi_phi_coordinate=P2(2*30*math.pi,30/180*math.pi)
        )
        point = P3(1,0.1,0)
        print("计算电流元集合，在 p 点产生的磁场")
        print("CPU计算结果：",cct.magnetic_field_at(point))

        # 获取 kls p0s，有 32 位和 64 位之分
        kls64,p0s64=cct.global_current_elements_and_elementary_current_positions(numpy.float64)
        kls32,p0s32=cct.global_current_elements_and_elementary_current_positions(numpy.float32)
        # 电流元数目
        current_element_number = cct.total_disperse_number
        print("GPU32计算结果：",ga32.current_element_B(kls32,p0s32,current_element_number,point))
        print("GPU64计算结果：",ga64.current_element_B(kls64,p0s64,current_element_number,point))
        """
        if self.cpu_mode:
            raise NotImplementedError

        code = """
        __global__ void ce(FLOAT *kls, FLOAT *p0s, int* number, FLOAT *p, FLOAT *ret){
            __shared__ FLOAT s_ret[DIM];
            int tid = threadIdx.x;
            current_element_B(kls,p0s,*number,p,s_ret);
            if(tid == 0) vct_copy(s_ret, ret);
        }

        """

        mod = SourceModule(self.cuda_code + code)

        ce = mod.get_function("ce")

        ret = numpy.empty((3,), dtype=self.numpy_dtype)

        ce(drv.In(kls.astype(self.numpy_dtype)),
           drv.In(p0s.astype(self.numpy_dtype)),
           drv.In(numpy.array([number], dtype=numpy.int32)),
           drv.In(p.to_numpy_ndarry3(numpy_dtype=self.numpy_dtype)),
           drv.Out(ret),
           grid=(1, 1, 1), block=(self.block_dim_x, 1, 1))

        return P3.from_numpy_ndarry(ret)

    def magnet_at_qs(self, qs_data, p3: P3):
        """
        qs 磁铁在 p 点产生的磁场
        p 点是 全局坐标点
        测试代码：
        qs = QS(
            local_coordinate_system=LocalCoordinateSystem(
                location=P3(1,0,0),
                x_direction=-P3.x_direct(),
                z_direction=P3.y_direct()
            ),
            length=0.27,
            gradient=5,
            second_gradient=20,
            aperture_radius=100*MM
        )
        point = P3(0.95,0,0)
        # 查看入口中心位置的磁场
        print("magnet_at_qs 计算 qs 磁铁，在 p 点产生的磁场")
        print("CPU计算结果：",qs.magnetic_field_at(point))
        print("GPU32计算结果：",ga32.magnet_at_qs(qs.to_numpy_array(numpy.float32),point))
        print("GPU64计算结果：",ga64.magnet_at_qs(qs.to_numpy_array(numpy.float64),point))
        # CPU计算结果： (0.0, 0.0, 0.27500000000000024)
        # GPU32计算结果： (0.0, 0.0, 0.27500006556510925)
        # GPU64计算结果： (0.0, 0.0, 0.27500000000000024)
        """
        if self.cpu_mode:
            raise NotImplementedError

        code = """
        __global__ void mq(FLOAT *qs_data, FLOAT *p, FLOAT *ret){
            magnet_at_qs(
                qs_data, // origin
                qs_data + 3, //xi
                qs_data + 6, //yi
                qs_data + 9, //zi
                *(qs_data + 12), // len
                *(qs_data + 13), // g
                *(qs_data + 14), // sg
                *(qs_data + 15), // aper r
                p, ret
            );
        }

        """

        mod = SourceModule(self.cuda_code + code)

        mq = mod.get_function("mq")

        ret = numpy.empty((3,), dtype=self.numpy_dtype)

        mq(drv.In(qs_data.astype(self.numpy_dtype)),
           drv.In(p3.to_numpy_ndarry3(numpy_dtype=self.numpy_dtype)),
           drv.Out(ret),
           grid=(1, 1, 1), block=(1, 1, 1)
           )

        return P3.from_numpy_ndarry(ret)

    def magnet_at_qs_date(self, qs_data, p3: P3):
        """
        qs 磁铁在 p 点产生的磁场
        qs_data 版本
        测试代码：
        qs = QS(
            local_coordinate_system=LocalCoordinateSystem(
                location=P3(1,0,0),
                x_direction=-P3.x_direct(),
                z_direction=P3.y_direct()
            ),
            length=0.27,
            gradient=5,
            second_gradient=20,
            aperture_radius=100*MM
        )
        point = P3(0.95,0,0)
        print("magnet_at_qs_date 计算 qs 磁铁，在 p 点产生的磁场")
        print("CPU计算结果：",qs.magnetic_field_at(point))
        print("GPU32计算结果：",ga32.magnet_at_qs_date(qs.to_numpy_array(numpy.float32),point))
        print("GPU64计算结果：",ga64.magnet_at_qs_date(qs.to_numpy_array(numpy.float64),point))

        since 2021年5月6日
        """
        if self.cpu_mode:
            raise NotImplementedError

        code = """
        __global__ void mq_data(FLOAT *qs_data, FLOAT *p, FLOAT *ret){
            magnet_at_qs_date(qs_data, p ,ret);
        }

        """

        mod = SourceModule(self.cuda_code + code)

        mq = mod.get_function("mq_data")

        ret = numpy.empty((3,), dtype=self.numpy_dtype)

        mq(drv.In(qs_data.astype(self.numpy_dtype)),
           drv.In(p3.to_numpy_ndarry3(numpy_dtype=self.numpy_dtype)),
           drv.Out(ret),
           grid=(1, 1, 1), block=(1, 1, 1)
           )

        return P3.from_numpy_ndarry(ret)

    def magnet_at_qs_dates(self, qss: List[QS], p: P3) -> P3:
        """
        多个 qs 磁铁在 p 点产生的磁场

        2021年5月6日 验证成功
        """
        code = """
        __global__ void qss(FLOAT *qs_data, int* qs_number, FLOAT *p, FLOAT *ret){
            // 这里必须用 __shared__ 去接收
            __shared__ FLOAT s_ret[DIM];
            int tid = threadIdx.x;
            magnet_at_qs_dates(qs_data, *qs_number, p, s_ret);
            if(tid == 0) vct_copy(s_ret, ret);
        }

        """
        if self.cpu_mode:
            m = P3.zeros()
            for qs in qss:
                m += qs.magnetic_field_at(p)
            return m

        mod = SourceModule(self.cuda_code + code)

        mq = mod.get_function("qss")

        ret = numpy.empty((3,), dtype=self.numpy_dtype)

        qs_datas: List[numpy.ndarray] = []
        for qs in qss:
            qs_datas.append(qs.to_numpy_array(self.numpy_dtype))

        qs_datas: numpy.ndarray = numpy.concatenate(tuple(qs_datas))

        mq(drv.In(qs_datas.astype(self.numpy_dtype)),
           drv.In(numpy.array([len(qss)], dtype=numpy.int32)),
           drv.In(p.to_numpy_ndarry3(numpy_dtype=self.numpy_dtype)),
           drv.Out(ret),
           grid=(1, 1, 1), block=(self.block_dim_x, 1, 1)
           )

        return P3.from_numpy_ndarry(ret)

    def magnet_at_beamline_with_single_qs(self, bl: Beamline, p: P3) -> P3:
        """
        CCT 和 QS 合起来测试
        测试代码：
        bl = (
            Beamline.set_start_point(P2.origin())
            .first_drift(direct=P2.x_direct(),length=1)
            .append_qs(length=0.27,gradient=5,second_gradient=20,aperture_radius=100*MM)
            .append_drift(length=1)
            .append_dipole_cct(
                big_r=1,small_r_inner=100*MM,small_r_outer=120*MM,bending_angle=45,
                tilt_angles=[30],winding_number=60,current=10000
            ).append_drift(length=1)
        )
        print(" magnet_at_beamline_with_single_qs 单一 qs 的 beamline 磁场计算")
        point1 = P3(1.2,50*MM,0)
        point2 = P3(2.3,50*MM,0)
        print("CPU计算结果1：",bl.magnetic_field_at(point1))
        print("GPU32计算结果1：",ga32.magnet_at_beamline_with_single_qs(bl,point1))
        print("GPU64计算结果1：",ga64.magnet_at_beamline_with_single_qs(bl,point1))
        print("CPU计算结果2：",bl.magnetic_field_at(point2))
        print("GPU32计算结果2：",ga32.magnet_at_beamline_with_single_qs(bl,point2))
        print("GPU64计算结果2：",ga64.magnet_at_beamline_with_single_qs(bl,point2))
        # GPU32计算结果1： (0.0006631895666942, -9.404712182004005e-05, 0.2723771035671234)
        # GPU64计算结果1： (0.000663189549448528, -9.404708930037921e-05, 0.2723771039989055)
        # CPU计算结果2： (-0.021273493843574243, 0.048440921145815385, 1.0980479752081713)
        # GPU32计算结果2： (-0.021273484453558922, 0.04844103008508682, 1.0980477333068848)
        # GPU64计算结果2： (-0.021273493843573958, 0.04844092114581488, 1.0980479752081695)
        """
        if self.cpu_mode:
            return bl.magnetic_field_at(p)

        code = """
        __global__ void ma(FLOAT *kls, FLOAT* p0s, int* current_element_number, 
                FLOAT *qs_data, FLOAT *p, FLOAT *ret){
            int tid = threadIdx.x;
            __shared__ FLOAT shared_ret[DIM];

            magnet_with_single_qs(kls, p0s, *current_element_number, qs_data, p, shared_ret);

            if(tid == 0) vct_copy(shared_ret, ret);
        }
        """

        mod = SourceModule(self.cuda_code + code)

        ma = mod.get_function('ma')

        ret = numpy.empty((3,), dtype=self.numpy_dtype)

        kls_list: List[numpy.ndarray] = []  # 存放多个 cct 线圈的 kls
        p0s_list: List[numpy.ndarray] = []  # 存放多个 cct 线圈的 p0s
        current_element_number = 0

        qs_data = None  # 只有一个 qs

        for m in bl.magnets:
            if isinstance(m, CCT):
                cct = CCT.as_cct(m)
                kls, p0s = cct.global_current_elements_and_elementary_current_positions(
                    numpy_dtype=self.numpy_dtype)
                current_element_number += cct.total_disperse_number
                kls_list.append(kls)  # 记住 kls 和 p0s 是一维数组，没三个为一组表示一个三维矢量
                p0s_list.append(p0s)
            elif isinstance(m, QS):
                qs = QS.as_qs(m)
                qs_data = qs.to_numpy_array(self.numpy_dtype)
            else:
                raise ValueError(f"磁铁 {m} 无法用 GPU 加速")

        kls_all = numpy.concatenate(tuple(kls_list))
        p0s_all = numpy.concatenate(tuple(p0s_list))

        ma(
            drv.In(kls_all),
            drv.In(p0s_all),
            drv.In(numpy.array([current_element_number], dtype=numpy.int32)),
            drv.In(qs_data),
            drv.In(p.to_numpy_ndarry3(numpy_dtype=self.numpy_dtype)),
            drv.Out(ret),
            grid=(1, 1, 1), block=(self.block_dim_x, 1, 1)
        )

        return P3.from_numpy_ndarry(ret)

    def track_one_particle_with_single_qs(self, bl: Beamline, p: RunningParticle, distance: float, footstep: float):
        """
        粒子跟踪，电流元 + 单个 QS
        测试代码：
        # 创建 beamline 只有一个 qs
        bl = HUST_SC_GANTRY().create_second_bending_part_beamline()
        # 创建粒子（理想粒子）
        particle = ParticleFactory.create_proton_along(bl,kinetic_MeV=215)
        # 复制三份
        particle_cpu = particle.copy()
        particle_gpu32 = particle.copy()
        particle_gpu64 = particle.copy()
        # 运行
        footstep=100*MM
        ParticleRunner.run_only(particle_cpu,bl,bl.get_length(),footstep=footstep)
        ga32.track_one_particle_with_single_qs(bl,particle_gpu32,bl.get_length(),footstep=footstep)
        ga64.track_one_particle_with_single_qs(bl,particle_gpu64,bl.get_length(),footstep=footstep)
        print("CPU计算结果: ",particle_cpu.detailed_info())
        print("GPU32计算结果: ",particle_gpu32.detailed_info())
        print("GPU64计算结果: ",particle_gpu64.detailed_info())
        print("GPU32计算和CPU对比: ",(particle_cpu-particle_gpu32).detailed_info())
        print("GPU64计算和CPU对比: ",(particle_cpu-particle_gpu64).detailed_info())
        # CPU计算结果:  Particle[p=(7.409509849267735, -0.028282989447753218, 5.0076184754665586e-05), v=(1809891.9615852616, -174308430.5414393, -330480.4098605619)], rm=2.0558942080656965e-27, e=1.6021766208e-19, speed=174317774.94179922, distance=7.104727865682728]
        # GPU32计算结果:  Particle[p=(7.409510612487793, -0.02828289568424225, 5.0118236686103046e-05), v=(1809917.875, -174308416.0, -330476.3125)], rm=2.0558942007434142e-27, e=1.602176597458587e-19, speed=174317776.0, distance=7.104727745056152]
        # GPU64计算结果:  Particle[p=(7.409509849267735, -0.028282989447752843, 5.0076184754525616e-05), v=(1809891.961585234, -174308430.54143927, -330480.409860578)], rm=2.0558942080656965e-27, e=1.6021766208e-19, speed=174317774.94179922, distance=7.104727865682728]
        # GPU32计算和CPU对比:  Particle[p=(-7.632200578200354e-07, -9.376351096934687e-08, -4.2051931437459694e-08), v=(-25.91341473837383, -14.541439294815063, -4.097360561892856)], rm=7.322282306994799e-36, e=2.3341413164924317e-27, speed=-1.0582007765769958, distance=1.2062657539502197e-07]
        # GPU64计算和CPU对比:  Particle[p=(0.0, -3.7470027081099033e-16, 1.3997050046787862e-16), v=(2.7706846594810486e-08, -2.9802322387695312e-08, 1.6123522073030472e-08)], rm=0.0, e=0.0, speed=0.0, distance=0.0]
        """
        if self.cpu_mode:
            ParticleRunner.run_only(
                p = p,
                m = bl,
                length = distance,
                footstep = footstep
            )
            return

        mod = SourceModule(self.cuda_code)

        track = mod.get_function('track_for_magnet_with_single_qs_g')

        particle: numpy.ndarray = p.to_numpy_array_data(
            numpy_dtype=self.numpy_dtype)

        kls_list: List[numpy.ndarray] = []  # 存放多个 cct 线圈的 kls
        p0s_list: List[numpy.ndarray] = []  # 存放多个 cct 线圈的 p0s
        current_element_number = 0

        qs_data = None
        for m in bl.magnets:
            if isinstance(m, CCT):
                cct = CCT.as_cct(m)
                kls, p0s = cct.global_current_elements_and_elementary_current_positions(
                    numpy_dtype=self.numpy_dtype)
                current_element_number += cct.total_disperse_number
                kls_list.append(kls)  # 记住 kls 和 p0s 是一维数组，没三个为一组表示一个三维矢量
                p0s_list.append(p0s)
            elif isinstance(m, QS):
                qs = QS.as_qs(m)
                qs_data = qs.to_numpy_array(numpy_dtype=self.numpy_dtype)
            else:
                raise ValueError(f"磁铁 {m} 无法用 GPU 加速")

        kls_all = numpy.concatenate(tuple(kls_list))
        p0s_all = numpy.concatenate(tuple(p0s_list))

        track(
            drv.In(numpy.array([distance], dtype=self.numpy_dtype)),
            drv.In(numpy.array([footstep], dtype=self.numpy_dtype)),
            drv.In(kls_all),
            drv.In(p0s_all),
            drv.In(numpy.array([current_element_number], dtype=numpy.int32)),
            drv.In(qs_data),
            drv.InOut(particle),
            grid=(1, 1, 1), block=(self.block_dim_x, 1, 1)
        )

        p.populate(RunningParticle.from_numpy_array_data(particle))

    def track_one_particle_with_multi_qs(self, bl: Beamline, p: RunningParticle, distance: float, footstep: float):
        """
        粒子跟踪，电流元 + 多个 QS

        测试代码：
        # 创建 beamline 3个 qs
        bl = HUST_SC_GANTRY().create_first_bending_part_beamline()
        # 创建粒子（理想粒子）
        particle = ParticleFactory.create_proton_along(bl,kinetic_MeV=215)
        # 复制三份
        particle_cpu = particle.copy()
        particle_gpu32 = particle.copy()
        particle_gpu64 = particle.copy()
        # 运行
        footstep=100*MM
        ParticleRunner.run_only(particle_cpu,bl,bl.get_length(),footstep=footstep)
        ga32.track_one_particle_with_multi_qs(bl,particle_gpu32,bl.get_length(),footstep=footstep)
        ga64.track_one_particle_with_multi_qs(bl,particle_gpu64,bl.get_length(),footstep=footstep)
        print("track_one_particle_with_multi_qs 2 ")
        print("CPU计算结果: ",particle_cpu.detailed_info())
        print("GPU32计算结果: ",particle_gpu32.detailed_info())
        print("GPU64计算结果: ",particle_gpu64.detailed_info())
        print("GPU32计算和CPU对比: ",(particle_cpu-particle_gpu32).detailed_info())
        print("GPU64计算和CPU对比: ",(particle_cpu-particle_gpu64).detailed_info())
        # track_one_particle_with_multi_qs 2
        # CPU计算结果:  Particle[p=(3.687315812380205, 1.548315945537494, -0.003352065021200123), v=(119474899.55705348, 126923892.97270872, -352485.58348381834)], rm=2.0558942080656965e-27, e=1.6021766208e-19, speed=174317774.94179922, distance=4.149802255227576]
        # GPU32计算结果:  Particle[p=(3.6873157024383545, 1.5483157634735107, -0.0033521109726279974), v=(119474888.0, 126923888.0, -352490.09375)], rm=2.0558942007434142e-27, e=1.602176597458587e-19, speed=174317776.0, distance=4.149802207946777]
        # GPU64计算结果:  Particle[p=(3.687315812380205, 1.5483159455374929, -0.0033520650212005175), v=(119474899.55705343, 126923892.97270869, -352485.58348386886)], rm=2.0558942080656965e-27, e=1.6021766208e-19, speed=174317774.94179922, distance=4.149802255227576]
        # GPU32计算和CPU对比:  Particle[p=(1.0994185029034043e-07, 1.8206398322284656e-07, 4.595142787458539e-08), v=(11.557053476572037, 4.9727087169885635, 4.51026618166361)], rm=7.322282306994799e-36, e=2.3341413164924317e-27, speed=-1.0582007765769958, distance=4.728079883165037e-08]
        # GPU64计算和CPU对比:  Particle[p=(0.0, 1.1102230246251565e-15, 3.946495907847236e-16), v=(4.470348358154297e-08, 2.9802322387695312e-08, 5.052424967288971e-08)], rm=0.0, e=0.0, speed=0.0, distance=0.0]
        """
        if self.cpu_mode:
            ParticleRunner.run_only(
                p = p,
                m = bl,
                length = distance,
                footstep = footstep
            )
            return

        mod = SourceModule(self.cuda_code)

        track = mod.get_function('track_for_magnet_with_multi_qs_g')

        particle: numpy.ndarray = p.to_numpy_array_data(
            numpy_dtype=self.numpy_dtype)

        kls_list: List[numpy.ndarray] = []  # 存放多个 cct 线圈的 kls
        p0s_list: List[numpy.ndarray] = []  # 存放多个 cct 线圈的 p0s
        current_element_number = 0  # 电流元数目
        qs_number = 0  # qs 数目

        qs_datas: List[numpy.ndarray] = []  # qs_data 数据
        for m in bl.magnets:
            if isinstance(m, CCT):
                cct = CCT.as_cct(m)
                kls, p0s = cct.global_current_elements_and_elementary_current_positions(
                    numpy_dtype=self.numpy_dtype)
                current_element_number += cct.total_disperse_number
                kls_list.append(kls)  # 记住 kls 和 p0s 是一维数组，没三个为一组表示一个三维矢量
                p0s_list.append(p0s)
            elif isinstance(m, QS):
                qs = QS.as_qs(m)
                qs_datas.append(qs.to_numpy_array(
                    numpy_dtype=self.numpy_dtype))
                qs_number += 1
            else:
                raise ValueError(f"磁铁 {m} 无法用 GPU 加速")

        kls_all = numpy.concatenate(tuple(kls_list))
        p0s_all = numpy.concatenate(tuple(p0s_list))
        qs_datas_con = numpy.concatenate(tuple(qs_datas))

        track(
            drv.In(numpy.array([distance], dtype=self.numpy_dtype)),
            drv.In(numpy.array([footstep], dtype=self.numpy_dtype)),
            drv.In(kls_all),
            drv.In(p0s_all),
            drv.In(numpy.array([current_element_number], dtype=numpy.int32)),
            drv.In(qs_datas_con),
            drv.In(numpy.array([qs_number], dtype=numpy.int32)),
            drv.InOut(particle),
            grid=(1, 1, 1), block=(self.block_dim_x, 1, 1)
        )

        p.populate(RunningParticle.from_numpy_array_data(particle))

    def track_multi_particle_for_magnet_with_single_qs(self, bl: Beamline, ps: List[RunningParticle], distance: float, footstep: float):
        """
        多粒子跟踪，电流元 + 单个 QS
        测试代码：
        # track_multi_particle_for_magnet_with_single_qs_g 多粒子跟踪，电流元 + 单个 QS
        # 创建 beamline 只有一个 qs
        bl = HUST_SC_GANTRY().create_second_bending_part_beamline()
        # 创建起点和终点的理想粒子
        ip_start = ParticleFactory.create_proton_along(bl,kinetic_MeV=215,s=0)
        ip_end = ParticleFactory.create_proton_along(bl,kinetic_MeV=215,s=bl.get_length())
        # 创建相椭圆分布粒子
        pps = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_xxp_plane(
            xMax=3.5*MM,xpMax=7*MRAD,delta=0.0,number=6
        )
        # 将相椭圆分布粒子转为实际粒子
        ps = ParticleFactory.create_from_phase_space_particles(
            ideal_particle=ip_start,
            coordinate_system=ip_start.get_natural_coordinate_system(),
            phase_space_particles=pps
        )
        # 复制三分
        ps_cpu = [p.copy() for p in ps]
        ps_gpu32 = [p.copy() for p in ps]
        ps_gpu64 = [p.copy() for p in ps]
        # 运行
        footstep=100*MM
        ParticleRunner.run_only(ps_cpu,bl,bl.get_length(),footstep)
        ga32.track_multi_particle_for_magnet_with_single_qs(bl,ps_gpu32,bl.get_length(),footstep)
        ga64.track_multi_particle_for_magnet_with_single_qs(bl,ps_gpu64,bl.get_length(),footstep)
        # 转回相空间
        pps_end_cpu = PhaseSpaceParticle.create_from_running_particles(ip_end,ip_end.get_natural_coordinate_system(),ps_cpu)
        pps_end_gpu32 = PhaseSpaceParticle.create_from_running_particles(ip_end,ip_end.get_natural_coordinate_system(),ps_gpu32)
        pps_end_gpu64 = PhaseSpaceParticle.create_from_running_particles(ip_end,ip_end.get_natural_coordinate_system(),ps_gpu64)
        # 绘图
        Plot2.plot_p2s(PhaseSpaceParticle.phase_space_particles_project_to_xxp_plane(pps_end_cpu,True),describe='rx')
        Plot2.plot_p2s(PhaseSpaceParticle.phase_space_particles_project_to_xxp_plane(pps_end_gpu32,True),describe='k|')
        Plot2.plot_p2s(PhaseSpaceParticle.phase_space_particles_project_to_xxp_plane(pps_end_gpu64,True),describe='b_')
        Plot2.legend("CPU","GPU32","GPU64",font_size=32)
        Plot2.info(x_label='x/mm',y_label="xp/mr",title="xxp-plane",font_size=32)
        Plot2.equal()
        Plot2.show()
        """
        if self.cpu_mode:
            ParticleRunner.run_only(
                p = ps,
                m = bl,
                length = distance,
                footstep = footstep
            )
            return

        mod = SourceModule(self.cuda_code)

        track = mod.get_function(
            'track_multi_particle_for_magnet_with_single_qs_g')

        kls_list: List[numpy.ndarray] = []  # 存放多个 cct 线圈的 kls
        p0s_list: List[numpy.ndarray] = []  # 存放多个 cct 线圈的 p0s
        particle_list: List[numpy.ndarray] = [
            p.to_numpy_array_data(numpy_dtype=self.numpy_dtype) for p in ps]
        current_element_number = 0

        qs_data = None
        for m in bl.magnets:
            if isinstance(m, CCT):
                cct = CCT.as_cct(m)
                kls, p0s = cct.global_current_elements_and_elementary_current_positions(
                    numpy_dtype=self.numpy_dtype)
                current_element_number += cct.total_disperse_number
                kls_list.append(kls)  # 记住 kls 和 p0s 是一维数组，没三个为一组表示一个三维矢量
                p0s_list.append(p0s)
            elif isinstance(m, QS):
                qs = QS.as_qs(m)
                qs_data = qs.to_numpy_array(numpy_dtype=self.numpy_dtype)
            else:
                raise ValueError(f"磁铁 {m} 无法用 GPU 加速")

        kls_all = numpy.concatenate(tuple(kls_list))
        p0s_all = numpy.concatenate(tuple(p0s_list))
        particles_all = numpy.concatenate(tuple(particle_list))

        track(
            drv.In(numpy.array([distance], dtype=self.numpy_dtype)),
            drv.In(numpy.array([footstep], dtype=self.numpy_dtype)),
            drv.In(kls_all),
            drv.In(p0s_all),
            drv.In(numpy.array([current_element_number], dtype=numpy.int32)),
            drv.In(qs_data),
            drv.InOut(particles_all),
            drv.In(numpy.array([len(ps)], dtype=numpy.int32)),
            grid=(1, 1, 1), block=(self.block_dim_x, 1, 1)
        )

        particles_all = particles_all.reshape((-1, 10))
        for i in range(len(ps)):
            ps[i].populate(
                RunningParticle.from_numpy_array_data(particles_all[i]))

    def track_multi_particle_for_magnet_with_multi_qs(self, bl: Beamline, ps: List[RunningParticle], distance: float, footstep: float):
        """
        多粒子跟踪，电流元 + 多个 QS

        测试代码：
        # ----- track_multi_particle_for_magnet_with_multi_qs -----
        # 创建 beamline 3个 qs
        bl = HUST_SC_GANTRY().create_first_bending_part_beamline()
        # 创建起点和终点的理想粒子
        ip_start = ParticleFactory.create_proton_along(bl,kinetic_MeV=215,s=0)
        ip_end = ParticleFactory.create_proton_along(bl,kinetic_MeV=215,s=bl.get_length())
        # 创建相椭圆分布粒子
        pps = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_xxp_plane(
            xMax=3.5*MM,xpMax=7*MRAD,delta=0.0,number=6
        )
        # 将相椭圆分布粒子转为实际粒子
        ps = ParticleFactory.create_from_phase_space_particles(
            ideal_particle=ip_start,
            coordinate_system=ip_start.get_natural_coordinate_system(),
            phase_space_particles=pps
        )
        # 复制三分
        ps_cpu = [p.copy() for p in ps]
        ps_gpu32 = [p.copy() for p in ps]
        ps_gpu64 = [p.copy() for p in ps]
        # 运行
        footstep=100*MM
        ParticleRunner.run_only(ps_cpu,bl,bl.get_length(),footstep)
        ga32.track_multi_particle_for_magnet_with_multi_qs(bl,ps_gpu32,bl.get_length(),footstep)
        ga64.track_multi_particle_for_magnet_with_multi_qs(bl,ps_gpu64,bl.get_length(),footstep)
        # 转回相空间
        pps_end_cpu = PhaseSpaceParticle.create_from_running_particles(ip_end,ip_end.get_natural_coordinate_system(),ps_cpu)
        pps_end_gpu32 = PhaseSpaceParticle.create_from_running_particles(ip_end,ip_end.get_natural_coordinate_system(),ps_gpu32)
        pps_end_gpu64 = PhaseSpaceParticle.create_from_running_particles(ip_end,ip_end.get_natural_coordinate_system(),ps_gpu64)
        # 绘图
        Plot2.plot_p2s(PhaseSpaceParticle.phase_space_particles_project_to_xxp_plane(pps_end_cpu,True),describe='rx')
        Plot2.plot_p2s(PhaseSpaceParticle.phase_space_particles_project_to_xxp_plane(pps_end_gpu32,True),describe='k|')
        Plot2.plot_p2s(PhaseSpaceParticle.phase_space_particles_project_to_xxp_plane(pps_end_gpu64,True),describe='b_')
        Plot2.legend("CPU","GPU32","GPU64",font_size=32)
        Plot2.info(x_label='x/mm',y_label="xp/mr",title="xxp-plane",font_size=32)
        Plot2.equal()
        Plot2.show()
        """
        if self.cpu_mode:
            ParticleRunner.run_only(
                p = ps,
                m = bl,
                length = distance,
                footstep = footstep
            )
            return

        mod = SourceModule(self.cuda_code)

        track = mod.get_function(
            'track_multi_particle_for_magnet_with_multi_qs_g')

        kls_list: List[numpy.ndarray] = []  # 存放多个 cct 线圈的 kls
        p0s_list: List[numpy.ndarray] = []  # 存放多个 cct 线圈的 p0s
        particle_list: List[numpy.ndarray] = [
            p.to_numpy_array_data(numpy_dtype=self.numpy_dtype) for p in ps]
        current_element_number = 0

        qs_datas: List[numpy.ndarray] = []
        qs_number = 0

        for m in bl.magnets:
            if isinstance(m, CCT):
                cct = CCT.as_cct(m)
                kls, p0s = cct.global_current_elements_and_elementary_current_positions(
                    numpy_dtype=self.numpy_dtype)
                current_element_number += cct.total_disperse_number
                kls_list.append(kls)  # 记住 kls 和 p0s 是一维数组，没三个为一组表示一个三维矢量
                p0s_list.append(p0s)
            elif isinstance(m, QS):
                qs = QS.as_qs(m)
                qs_datas.append(qs.to_numpy_array(
                    numpy_dtype=self.numpy_dtype))
                qs_number += 1
            else:
                raise ValueError(f"磁铁 {m} 无法用 GPU 加速")

        kls_all = numpy.concatenate(tuple(kls_list))
        p0s_all = numpy.concatenate(tuple(p0s_list))
        particles_all = numpy.concatenate(tuple(particle_list))
        qs_datas_con = numpy.concatenate(tuple(qs_datas))

        track(
            drv.In(numpy.array([distance], dtype=self.numpy_dtype)),
            drv.In(numpy.array([footstep], dtype=self.numpy_dtype)),
            drv.In(kls_all),
            drv.In(p0s_all),
            drv.In(numpy.array([current_element_number], dtype=numpy.int32)),
            drv.In(qs_datas_con),
            drv.In(numpy.array([qs_number], dtype=numpy.int32)),
            drv.InOut(particles_all),
            drv.In(numpy.array([len(ps)], dtype=numpy.int32)),
            grid=(1, 1, 1), block=(self.block_dim_x, 1, 1)
        )

        particles_all = particles_all.reshape((-1, 10))
        for i in range(len(ps)):
            ps[i].populate(
                RunningParticle.from_numpy_array_data(particles_all[i]))

    def track_multi_particle_beamline_for_magnet_with_single_qs(
        self, bls: List[Beamline], ps: List[RunningParticle],
            distance: float, footstep: float) -> List[List[RunningParticle]]:
        """
        多粒子多束线跟踪，电流元 + 单个 QS
        """
        if self.cpu_mode:
            ret:List[List[RunningParticle]] = []
            for bl in bls:
                cps = [p.copy() for p in ps]
                ParticleRunner.run_only(
                    p = cps,
                    m = bl,
                    length = distance,
                    footstep = footstep
                )
                ret.append(cps)
            return ret

        mod = SourceModule(self.cuda_code)

        track = mod.get_function(
            'track_multi_particle_beamline_for_magnet_with_single_qs')

        # 所有 beamline 的数据
        kls_all_all_beamline: List[numpy.ndarray] = []
        p0s_all_all_beamline: List[numpy.ndarray] = []
        qs_data_all_beamline: List[numpy.ndarray] = []
        particles_all_all_beamline: List[numpy.ndarray] = []
        current_element_numbers: List[int] = []
        for bl in bls:
            kls_list: List[numpy.ndarray] = []
            p0s_list: List[numpy.ndarray] = []
            particle_list: List[numpy.ndarray] = [
                p.to_numpy_array_data(numpy_dtype=self.numpy_dtype) for p in ps]
            current_element_number = 0

            qs_data = None
            for m in bl.magnets:
                if isinstance(m, CCT):
                    cct = CCT.as_cct(m)
                    kls, p0s = cct.global_current_elements_and_elementary_current_positions(
                        numpy_dtype=self.numpy_dtype)
                    current_element_number += cct.total_disperse_number
                    kls_list.append(kls)
                    p0s_list.append(p0s)
                elif isinstance(m, QS):
                    qs = QS.as_qs(m)
                    qs_data = qs.to_numpy_array(numpy_dtype=self.numpy_dtype)
                else:
                    raise ValueError(f"{m} 无法用 GOU 加速")

            kls_all = numpy.concatenate(tuple(kls_list))  # 多个连起来
            p0s_all = numpy.concatenate(tuple(p0s_list))

            # 这里复制一下的意义是什么呢？
            # 回答：因为一个机架的 kls p0s 长度是由 max_current_element_number*3 决定
            # 而不是由 len(kls_all) 决定
            kls_all_pad = numpy.zeros(
                (self.max_current_element_number*3,), dtype=self.numpy_dtype)
            p0s_all_pad = numpy.zeros(
                (self.max_current_element_number*3,), dtype=self.numpy_dtype)

            kls_all_pad[0:len(kls_all)] = kls_all
            p0s_all_pad[0:len(p0s_all)] = p0s_all

            particles_all = numpy.concatenate(tuple(particle_list))

            kls_all_all_beamline.append(kls_all_pad)
            p0s_all_all_beamline.append(p0s_all_pad)
            qs_data_all_beamline.append(qs_data)
            particles_all_all_beamline.append(particles_all)
            current_element_numbers.append(current_element_number)

        kls_all_all_beamline = numpy.concatenate(tuple(kls_all_all_beamline))
        p0s_all_all_beamline = numpy.concatenate(tuple(p0s_all_all_beamline))
        qs_data_all_beamline = numpy.concatenate(tuple(qs_data_all_beamline))
        particles_all_all_beamline = numpy.concatenate(
            tuple(particles_all_all_beamline))

        track(
            drv.In(numpy.array([distance], dtype=self.numpy_dtype)),  # 运动路程
            drv.In(numpy.array([footstep], dtype=self.numpy_dtype)),  # 步长

            drv.In(kls_all_all_beamline),
            drv.In(p0s_all_all_beamline),
            drv.In(numpy.array(current_element_numbers, dtype=numpy.int32)),

            drv.In(qs_data_all_beamline),
            drv.InOut(particles_all_all_beamline),
            drv.In(numpy.array([len(ps)], dtype=numpy.int32)),  # 粒子数
            grid=(len(bls), 1, 1),
            block=(self.block_dim_x, 1, 1)
        )

        particles_all_all_beamline = particles_all_all_beamline.reshape(
            (len(bls), len(ps), 10))

        ret: List[List[RunningParticle]] = []
        for bid in range(len(bls)):
            ps_ran: List[RunningParticle] = []
            for pid in range(len(ps)):
                ps_ran.append(RunningParticle.from_numpy_array_data(
                    particles_all_all_beamline[bid][pid]))
            ret.append(ps_ran)

        return ret

    def track_multi_particle_beamline_for_magnet_with_multi_qs(
        self, bls: List[Beamline], ps: List[RunningParticle],
            distance: float, footstep: float) -> List[List[RunningParticle]]:
        """
        多粒子多束线跟踪，电流元 + 多个 QS
        """
        if self.cpu_mode:
            ret:List[List[RunningParticle]] = []
            for bl in bls:
                cps = [p.copy() for p in ps]
                ParticleRunner.run_only(
                    p = cps,
                    m = bl,
                    length = distance,
                    footstep = footstep
                )
                ret.append(cps)
            return ret


        mod = SourceModule(self.cuda_code)

        track = mod.get_function(
            'track_multi_particle_beamline_for_magnet_with_multi_qs')

        # 所有 beamline 的数据
        kls_all_all_beamline: List[numpy.ndarray] = []
        p0s_all_all_beamline: List[numpy.ndarray] = []
        qs_datas_all_beamline: List[numpy.ndarray] = []
        particles_all_all_beamline: List[numpy.ndarray] = []
        current_element_numbers: List[int] = []
        qs_numbers: List[int] = []

        for bl in bls:
            kls_list: List[numpy.ndarray] = []
            p0s_list: List[numpy.ndarray] = []
            particle_list: List[numpy.ndarray] = [
                p.to_numpy_array_data(numpy_dtype=self.numpy_dtype) for p in ps]
            current_element_number = 0

            qs_number = 0
            qs_datas: List[numpy.ndarray] = []

            for m in bl.magnets:
                if isinstance(m, CCT):
                    cct = CCT.as_cct(m)
                    kls, p0s = cct.global_current_elements_and_elementary_current_positions(
                        numpy_dtype=self.numpy_dtype)
                    current_element_number += cct.total_disperse_number
                    kls_list.append(kls)
                    p0s_list.append(p0s)
                elif isinstance(m, QS):
                    qs = QS.as_qs(m)
                    qs_datas.append(qs.to_numpy_array(
                        numpy_dtype=self.numpy_dtype))
                    qs_number += 1
                else:
                    raise ValueError(f"{m} 无法用 GPU 加速")

            kls_all = numpy.concatenate(tuple(kls_list))  # 多个连起来
            p0s_all = numpy.concatenate(tuple(p0s_list))
            qs_datas_con = numpy.concatenate(tuple(qs_datas))

            # 制作 kls_all_all_beamline  p0s_all_all_beamline
            # 这里复制一下的意义是什么呢？
            # 回答：因为一个机架的 kls p0s 长度是由 max_current_element_number*3 决定
            # 而不是由 len(kls_all) 决定
            kls_all_pad = numpy.zeros(
                (self.max_current_element_number*3,), dtype=self.numpy_dtype)
            p0s_all_pad = numpy.zeros(
                (self.max_current_element_number*3,), dtype=self.numpy_dtype)

            kls_all_pad[0:len(kls_all)] = kls_all
            p0s_all_pad[0:len(p0s_all)] = p0s_all

            kls_all_all_beamline.append(kls_all_pad)
            p0s_all_all_beamline.append(p0s_all_pad)

            # 制作 qs_datas_all_beamline
            qs_datas_all_pad = numpy.zeros(
                (self.max_qs_datas_length*GPU_ACCELERATOR.QS_DATA_LENGTH,),
                dtype=self.numpy_dtype
            )
            qs_datas_all_pad[0:qs_number *
                             GPU_ACCELERATOR.QS_DATA_LENGTH] = qs_datas_con

            qs_datas_all_beamline.append(qs_datas_all_pad)

            particles_all = numpy.concatenate(tuple(particle_list))

            particles_all_all_beamline.append(particles_all)
            current_element_numbers.append(current_element_number)
            qs_numbers.append(qs_number)

        kls_all_all_beamline: numpy.ndarray = numpy.concatenate(
            tuple(kls_all_all_beamline))
        p0s_all_all_beamline: numpy.ndarray = numpy.concatenate(
            tuple(p0s_all_all_beamline))
        qs_datas_all_beamline: numpy.ndarray = numpy.concatenate(
            tuple(qs_datas_all_beamline))
        particles_all_all_beamline = numpy.concatenate(
            tuple(particles_all_all_beamline))

        track(
            drv.In(numpy.array([distance], dtype=self.numpy_dtype)),  # 运动路程
            drv.In(numpy.array([footstep], dtype=self.numpy_dtype)),  # 步长

            drv.In(kls_all_all_beamline),
            drv.In(p0s_all_all_beamline),
            drv.In(numpy.array(current_element_numbers, dtype=numpy.int32)),

            drv.In(qs_datas_all_beamline),
            drv.In(numpy.array(qs_numbers, dtype=numpy.int32)),

            drv.InOut(particles_all_all_beamline),
            drv.In(numpy.array([len(ps)], dtype=numpy.int32)),  # 粒子数
            grid=(len(bls), 1, 1),
            block=(self.block_dim_x, 1, 1)
        )

        particles_all_all_beamline = particles_all_all_beamline.reshape(
            (len(bls), len(ps), 10))

        ret: List[List[RunningParticle]] = []
        for bid in range(len(bls)):
            ps_ran: List[RunningParticle] = []
            for pid in range(len(ps)):
                ps_ran.append(RunningParticle.from_numpy_array_data(
                    particles_all_all_beamline[bid][pid]))
            ret.append(ps_ran)

        return ret

    
    def track_phase_ellipse_in_multi_beamline(
        self, 
        beamlines: List[Beamline],
        x_sigma_mm: float,
        xp_sigma_mrad: float,
        y_sigma_mm: float,
        yp_sigma_mrad,
        delta: float,
        particle_number: int,
        kinetic_MeV: float,
        s: float = 0.0,
        length: Optional[float] = None,
        footstep: float = 10 * MM,
        report: bool = True
    )->List[Tuple[List[P2], List[P2]]]:
        """
        是 Beamline.track_phase_ellipse 函数的升级
        Beamline.track_phase_ellipse 只能在 CPU 中运行，且一次只能计算一个机架
        这里计算多组机架，返回每组机架对应的相空间 x-xp / y-yp 粒子

        注意：这些机架的设计轨道应该一致！即机架长度、起点终点应该相同
        """
        if length is None:
            length = beamlines[0].trajectory.get_length() - s
        
        # 起点和终点位置的理想粒子
        ip_start = ParticleFactory.create_proton_along(
            beamlines[0].trajectory, s, kinetic_MeV)
        ip_end = ParticleFactory.create_proton_along(
            beamlines[0].trajectory, s + length, kinetic_MeV
        )

        # 起点处相空间粒子
        pp_x = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_xxp_plane(
            xMax=x_sigma_mm * MM,
            xpMax=xp_sigma_mrad * MRAD,
            delta=delta,
            number=particle_number,
        )

        pp_y = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_yyp_plane(
            yMax=y_sigma_mm * MM,
            ypMax=yp_sigma_mrad * MRAD,
            delta=delta,
            number=particle_number,
        )

        # 起点处实际粒子
        rp_x = ParticleFactory.create_from_phase_space_particles(
            ideal_particle=ip_start,
            coordinate_system=ip_start.get_natural_coordinate_system(),
            phase_space_particles=pp_x,
        )

        rp_y = ParticleFactory.create_from_phase_space_particles(
            ideal_particle=ip_start,
            coordinate_system=ip_start.get_natural_coordinate_system(),
            phase_space_particles=pp_y,
        )

        # 合起来
        all_rp = rp_x + rp_y

        ps_end_list_list = self.track_multi_particle_beamline_for_magnet_with_multi_qs(
            bls=beamlines,
            ps=all_rp,
            distance=length,
            footstep=footstep
        )

        # 返回值
        ret:List[Tuple[List[P2], List[P2]]] = []

        for ps_end_list_each_beamline in ps_end_list_list:
            # 不知道为什么，有些粒子的速率 speed 和速度 velocity 差别巨大
            for p in ps_end_list_each_beamline:
                p.speed = p.velocity.length()

            # 转为相空间
            pps_end_list_each_beamline: List[PhaseSpaceParticle] = PhaseSpaceParticle.create_from_running_particles(
                ip_end, ip_end.get_natural_coordinate_system(), ps_end_list_each_beamline
            )

            # 前一半是 x-xp
            pps_end_list_each_beamline_for_xxp = pps_end_list_each_beamline[0:particle_number]

            # 后一半是 y-yp
            pps_end_list_each_beamline_for_yyp = pps_end_list_each_beamline[particle_number:]

            # 前一半提取出来
            xs = PhaseSpaceParticle.phase_space_particles_project_to_xxp_plane(
                phase_space_particles=pps_end_list_each_beamline_for_xxp,
                convert_to_mm=True
            )

            # 后一半提取出来
            ys = PhaseSpaceParticle.phase_space_particles_project_to_yyp_plane(
                phase_space_particles=pps_end_list_each_beamline_for_yyp,
                convert_to_mm=True
            )

            ret.append(
                (xs,ys)
            )

        return ret






