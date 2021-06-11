# -*- coding: utf-8 -*-

"""
GPU CUDA 加速 cctpy 束流跟踪

2020年12月8日 12点15分 核心束流跟踪功能已经完成，对比成功
track cpu   p=p=[7.347173281024637, -5.038232430353374, -0.008126589272623864],v=[157601.42662973067, -174317561.422342, -223027.84656550566],v0=174317774.94179922
track gpu32 p=p=[7.347180366516113, -5.038208484649658, -0.008126441389322281],v=[157731.0625, -174317456.0, -223028.46875],v0=174317776.0
track gpu64 p=p=[7.347173281024622, -5.03823243035337,  -0.008126589272624135],v=[157601.42662950495, -174317561.42234194, -223027.8465655048],v0=174317774.94179922
利用 32 位计算，则误差约为 0.05 mm 和 0.01 mr
利用 64 位计算，误差约为 1e-10 mm
"""

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy
import time
import sys

from cctpy import *



class GPU_ACCELERATOR:
    FLOAT32: str = "FLOAT32"
    FLOAT64: str = "FLOAT64"

    def __init__(self, float_number_type: str = FLOAT32, block_dim_x: int = 1024, max_current_element_number: int = 2000*120) -> None:
        """
        float_number_type 浮点数类型，取值为 FLOAT32 或 FLOAT64，即 32 位运行或 64 位，默认 32 位。
        64 位浮点数精度更高，但是计算的速度可能比 32 位慢 2-10 倍

        block_dim_x 块线程数目，默认 1024 个，必须是 2 的幂次。如果采用 64 位浮点数，取 1024 可能会报错，应取 512 或更低
        不同大小的 block_dim_x，可能对计算效率有影响
        在抽象上，GPU 分为若干线程块，每个块内有若干线程
        块内线程，可以使用 __shared__ 使用共享内存（访问速度快），同时具有同步机制，因此可以方便的分工合作
        块之间，没有同步机制，所以线程通讯无从谈起

        max_current_element_number 最大电流元数目，在 GPU 加速中，CCT 数据以电流元的形式传入显存。
        默认值 2000*120 （可以看作一共 2000 匝，每匝分 120 段）
        """
        self.float_number_type = float_number_type
        self.max_current_element_number = max_current_element_number

        if block_dim_x > 1024 or block_dim_x < 0:
            raise ValueError(
                f"block_dim_x 应 >=1 and <=1024 内取，不能是{block_dim_x}")
        if block_dim_x & (block_dim_x-1) != 0:
            raise ValueError(f"block_dim_x 应该取 2 的幂次，不能为{block_dim_x}")
        self.block_dim_x: int = int(block_dim_x)

        cuda_code_00_include = """
        #include <stdio.h>

        """

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

        # 头信息
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
        """.format(block_dim_x=self.block_dim_x, max_current_element_number=self.max_current_element_number)

        # 向量运算内联函数
        cuda_code_03_vct_functions = """
        // 向量叉乘
        __device__ __forceinline__ void vct_cross(FLOAT *a, FLOAT *b, FLOAT *ret) {
            ret[X] = a[Y] * b[Z] - a[Z] * b[Y];
            ret[Y] = -a[X] * b[Z] + a[Z] * b[X];
            ret[Z] = a[X] * b[Y] - a[Y] * b[X];
        }

        // 向量原地加法
        __device__ __forceinline__ void vct_add_local(FLOAT *a_local, FLOAT *b) {
            a_local[X] += b[X];
            a_local[Y] += b[Y];
            a_local[Z] += b[Z];
        }

        // 向量原地加法
        __device__ __forceinline__ void vct6_add_local(FLOAT *a_local, FLOAT *b) {
            a_local[X] += b[X];
            a_local[Y] += b[Y];
            a_local[Z] += b[Z];
            a_local[X+DIM] += b[X+DIM];
            a_local[Y+DIM] += b[Y+DIM];
            a_local[Z+DIM] += b[Z+DIM];
        }

        // 向量加法
        __device__ __forceinline__ void vct_add(FLOAT *a, FLOAT *b, FLOAT *ret) {
            ret[X] = a[X] + b[X];
            ret[Y] = a[Y] + b[Y];
            ret[Z] = a[Z] + b[Z];
        }

        // 向量加法
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

        // 向量*常数，原地操作
        __device__ __forceinline__ void vct6_dot_a_v(FLOAT a, FLOAT *v) {
            v[X] *= a;
            v[Y] *= a;
            v[Z] *= a;
            v[X+DIM] *= a;
            v[Y+DIM] *= a;
            v[Z+DIM] *= a;
        }

        // 向量*常数
        __device__ __forceinline__ void vct_dot_a_v_ret(FLOAT a, FLOAT *v, FLOAT *ret) {
            ret[X] = v[X] * a;
            ret[Y] = v[Y] * a;
            ret[Z] = v[Z] * a;
        }

        // 向量*常数
        __device__ __forceinline__ void vct6_dot_a_v_ret(FLOAT a, FLOAT *v, FLOAT *ret) {
            ret[X] = v[X] * a;
            ret[Y] = v[Y] * a;
            ret[Z] = v[Z] * a;
            ret[X+DIM] = v[X+DIM] * a;
            ret[Y+DIM] = v[Y+DIM] * a;
            ret[Z+DIM] = v[Z+DIM] * a;
        }

        __device__ __forceinline__ FLOAT vct_dot_v_v(FLOAT *v,FLOAT *w){
            return v[X] * w[X] + v[Y] * w[Y] + v[Z] * w[Z];
        }

        // 向量拷贝赋值
        __device__ __forceinline__ void vct_copy(FLOAT *src, FLOAT *des) {
            des[X] = src[X];
            des[Y] = src[Y];
            des[Z] = src[Z];
        }

        // 向量拷贝赋值
        __device__ __forceinline__ void vct6_copy(FLOAT *src, FLOAT *des) {
            des[X] = src[X];
            des[Y] = src[Y];
            des[Z] = src[Z];
            des[X+DIM] = src[X+DIM];
            des[Y+DIM] = src[Y+DIM];
            des[Z+DIM] = src[Z+DIM];
        }

        // 求向量长度
        __device__ __forceinline__ FLOAT vct_len(FLOAT *v) {

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

        // 打印矢量，一般用于 debug
        __device__ __forceinline__ void vct6_print(FLOAT *v) {
            #ifdef FLOAT32
            printf("%.15f, %.15f, %.15f, %.15f, %.15f, %.15f\\n", v[X], v[Y], v[Z], v[X+DIM], v[Y+DIM], v[Z+DIM]);
            #else
            printf("%.15lf, %.15lf, %.15lf, %.15lf, %.15lf, %.15lf\\n", v[X], v[Y], v[Z] ,v[X+DIM], v[Y+DIM], v[Z+DIM]);
            #endif
        }

        // 矢量减法
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
        // shared_ret 应该是一个 shared 量，保存返回值
        // 调用该方法后，应该同步处理  __syncthreads();
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
        
        """

        cuda_code_06_magnet_at = """
        // 整个束线在 p 点产生得磁场（只有一个 QS 磁铁！）
        // FLOAT *kls, FLOAT* p0s, int current_element_number 和 CCT 电流元相关
        // FLOAT *qs_data 表示 QS 磁铁所有参数，分别是局部坐标系（原点origin,三个轴xi yi zi，长度 梯度 二阶梯度 孔径）
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
        """

        cuda_code_07_runge_kutta4 = """
        // runge_kutta4 代码和 cctpy 中的 runge_kutta4 一模一样
        // Y0 数组长度为 6
        // Y0 会发生变化，既是输入也是输出
        // 为了分析包络等，会出一个记录全部 YO 的函数
        // 这个函数单线程运行

        // void (*call)(FLOAT,FLOAT*,FLOAT*) 表示 tn Yn 到 Yn+1 的转义，实际使用中还会带更多参数（C 语言没有闭包）
        // 所以这个函数仅仅是原型
        __device__ void runge_kutta4(FLOAT t0, FLOAT t_end, FLOAT *Y0, void (*call)(FLOAT,FLOAT*,FLOAT*), FLOAT dt){
            #ifdef FLOAT32
            int number = (int)(ceilf((t_end - t0) / dt));
            #else
            int number = (int)(ceil((t_end - t0) / dt));
            #endif

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
        
        """

        cuda_code_09_run_multi_particle = """
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

        """

        cuda_code_10_run_multi_particle_multi_beamline = """
        __global__ void track_multi_particle_beamlime_for_magnet_with_single_qs(FLOAT *distance, FLOAT *footstep,
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
        """

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
        """

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

    def magnet_at(self, bl: Beamline, p: P3) -> P3:
        """
        CCT 和 QS 合起来测试
        """
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

        kls_list: List[numpy.ndarray] = []
        p0s_list: List[numpy.ndarray] = []
        current_element_number = 0

        qs_data = None
        for m in bl.magnets:
            if isinstance(m, CCT):
                cct = m
                kls, p0s = cct.global_current_elements_and_elementary_current_positions(
                    numpy_dtype=self.numpy_dtype)
                current_element_number += cct.total_disperse_number
                kls_list.append(kls)
                p0s_list.append(p0s)
            elif isinstance(m, QS):
                qs = m
                qs_data = numpy.array(
                    qs.local_coordinate_system.location.to_list(
                    ) + qs.local_coordinate_system.XI.to_list()
                    + qs.local_coordinate_system.YI.to_list() + qs.local_coordinate_system.ZI.to_list()
                    + [qs.length, qs.gradient, qs.second_gradient, qs.aperture_radius], dtype=self.numpy_dtype)
            else:
                raise ValueError(f"{m} 无法用 GOU 加速")

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
        """
        mod = SourceModule(self.cuda_code)

        track = mod.get_function('track_for_magnet_with_single_qs_g')

        particle = p.to_numpy_array_data(numpy_dtype=self.numpy_dtype)

        kls_list: List[numpy.ndarray] = []
        p0s_list: List[numpy.ndarray] = []
        current_element_number = 0

        qs_data = None
        for m in bl.magnets:
            if isinstance(m, CCT):
                cct = m
                kls, p0s = cct.global_current_elements_and_elementary_current_positions(
                    numpy_dtype=self.numpy_dtype)
                current_element_number += cct.total_disperse_number
                kls_list.append(kls)
                p0s_list.append(p0s)
            elif isinstance(m, QS):
                qs = m
                qs_data = numpy.array(
                    qs.local_coordinate_system.location.to_list(
                    ) + qs.local_coordinate_system.XI.to_list()
                    + qs.local_coordinate_system.YI.to_list() + qs.local_coordinate_system.ZI.to_list()
                    + [qs.length, qs.gradient, qs.second_gradient, qs.aperture_radius], dtype=self.numpy_dtype)
            else:
                raise ValueError(f"{m} 无法用 GOU 加速")

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

    def track_multi_particle_for_magnet_with_single_qs(self, bl: Beamline, ps: List[RunningParticle], distance: float, footstep: float):
        """
        多粒子跟踪，电流元 + 单个 QS
        """
        mod = SourceModule(self.cuda_code)

        track = mod.get_function(
            'track_multi_particle_for_magnet_with_single_qs_g')

        kls_list: List[numpy.ndarray] = []
        p0s_list: List[numpy.ndarray] = []
        particle_list: List[numpy.ndarray] = [
            p.to_numpy_array_data(numpy_dtype=self.numpy_dtype) for p in ps]
        current_element_number = 0

        qs_data = None
        for m in bl.magnets:
            if isinstance(m, CCT):
                cct = m
                kls, p0s = cct.global_current_elements_and_elementary_current_positions(
                    numpy_dtype=self.numpy_dtype)
                current_element_number += cct.total_disperse_number
                kls_list.append(kls)
                p0s_list.append(p0s)
            elif isinstance(m, QS):
                qs = m
                qs_data = numpy.array(
                    qs.local_coordinate_system.location.to_list(
                    ) + qs.local_coordinate_system.XI.to_list()
                    + qs.local_coordinate_system.YI.to_list() + qs.local_coordinate_system.ZI.to_list()
                    + [qs.length, qs.gradient, qs.second_gradient, qs.aperture_radius], dtype=self.numpy_dtype)
            else:
                raise ValueError(f"{m} 无法用 GOU 加速")

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

    def track_multi_particle_beamlime_for_magnet_with_single_qs(
            self, bls: List[Beamline], ps: List[RunningParticle], distance: float, footstep: float)->List[List[RunningParticle]]:
        """
        多粒子多束线跟踪，电流元 + 单个 QS
        """
        mod = SourceModule(self.cuda_code)

        track = mod.get_function(
            'track_multi_particle_beamlime_for_magnet_with_single_qs')

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
                    cct = m
                    kls, p0s = cct.global_current_elements_and_elementary_current_positions(
                        numpy_dtype=self.numpy_dtype)
                    current_element_number += cct.total_disperse_number
                    kls_list.append(kls)
                    p0s_list.append(p0s)
                elif isinstance(m, QS):
                    qs = m
                    qs_data = numpy.array(
                        qs.local_coordinate_system.location.to_list(
                        ) + qs.local_coordinate_system.XI.to_list()
                        + qs.local_coordinate_system.YI.to_list() + qs.local_coordinate_system.ZI.to_list()
                        + [qs.length, qs.gradient, qs.second_gradient, qs.aperture_radius], dtype=self.numpy_dtype)
                else:
                    raise ValueError(f"{m} 无法用 GOU 加速")

            kls_all = numpy.concatenate(tuple(kls_list))
            p0s_all = numpy.concatenate(tuple(p0s_list))

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


if __name__ == "__main__":
    # 测试
    import unittest

    gantry = HUST_SC_GANTRY()
    bl = gantry.create_beamline()
    ga64 = GPU_ACCELERATOR(float_number_type=GPU_ACCELERATOR.FLOAT64)
    ga32 = GPU_ACCELERATOR(float_number_type=GPU_ACCELERATOR.FLOAT32)

    ga64_b128 = GPU_ACCELERATOR(
        float_number_type=GPU_ACCELERATOR.FLOAT64, block_dim_x=128)
    ga32_b128 = GPU_ACCELERATOR(
        float_number_type=GPU_ACCELERATOR.FLOAT32, block_dim_x=128)

    ga64_b256 = GPU_ACCELERATOR(
        float_number_type=GPU_ACCELERATOR.FLOAT64, block_dim_x=256)
    ga32_b256 = GPU_ACCELERATOR(
        float_number_type=GPU_ACCELERATOR.FLOAT32, block_dim_x=256)

    ga64_b512 = GPU_ACCELERATOR(
        float_number_type=GPU_ACCELERATOR.FLOAT64, block_dim_x=512)
    ga32_b512 = GPU_ACCELERATOR(
        float_number_type=GPU_ACCELERATOR.FLOAT32, block_dim_x=512)

    class Test(unittest.TestCase):
        def test_vct_length(self):
            v = P3(1, 1, 1)
            length_cpu = v.length()
            length_gpu_32 = ga32.vct_length(v)
            length_gpu_64 = ga64.vct_length(v)

            print(f"test_vct_length 32 ={length_gpu_32 - length_cpu}")
            print(f"test_vct_length 64 ={length_gpu_64 - length_cpu}")

            self.assertTrue((length_gpu_32 - length_cpu) < 1e-6)
            self.assertTrue((length_gpu_64 - length_cpu) < 1e-14)

        def test_print(self):
            v = P3(1/3, 1/6, 1/7)
            ga32.vct_print(v)
            ga64.vct_print(v)
            self.assertTrue(True)

        def test_cct(self):
            cct: CCT = bl.magnets[15]
            p_cct = bl.trajectory.point_at(
                gantry.first_bending_part_length()+gantry.DL2+0.5).to_p3() + P3(1E-3, 1E-4, 1E-5)

            magnet_cpu = cct.magnetic_field_at(p_cct)

            kls, p0s = cct.global_current_elements_and_elementary_current_positions(
                numpy_dtype=numpy.float64)

            magnet_gpu_32 = ga32.current_element_B(
                kls.flatten(),
                p0s.flatten(),
                cct.total_disperse_number,
                p_cct,
            )

            magnet_gpu_64 = ga64.current_element_B(
                kls.flatten(),
                p0s.flatten(),
                cct.total_disperse_number,
                p_cct,
            )

            print(f"test_cct, diff_32={magnet_cpu-magnet_gpu_32}")
            print(f"test_cct, diff_64={magnet_cpu-magnet_gpu_64}")
            self.assertTrue((magnet_cpu-magnet_gpu_32).length() < 1e-6)
            self.assertTrue((magnet_cpu-magnet_gpu_64).length() < 1e-14)

        def test_qs(self):
            qs: QS = bl.magnets[23]
            p_qs = (bl.trajectory.point_at(gantry.first_bending_part_length()+gantry.DL2 +
                                           1.19+gantry.GAP1+gantry.qs3_length/2).to_p3() +
                    P3(10*MM, 10*MM, 10*MM))
            magnet_cpu = qs.magnetic_field_at(p_qs)
            qs_data = numpy.array(
                qs.local_coordinate_system.location.to_list(
                ) + qs.local_coordinate_system.XI.to_list()
                + qs.local_coordinate_system.YI.to_list() + qs.local_coordinate_system.ZI.to_list()
                + [qs.length, qs.gradient, qs.second_gradient, qs.aperture_radius], dtype=numpy.float64)
            magnet_gpu_32 = ga32.magnet_at_qs(
                qs_data=qs_data,
                p3=p_qs
            )
            magnet_gpu_64 = ga64.magnet_at_qs(
                qs_data=qs_data,
                p3=p_qs
            )
            print(f"test_qs, diff_32={magnet_cpu-magnet_gpu_32}")
            print(f"test_qs, diff_64={magnet_cpu-magnet_gpu_64}")
            self.assertTrue((magnet_cpu-magnet_gpu_32).length() < 1e-6)
            self.assertTrue((magnet_cpu-magnet_gpu_64).length() < 1e-14)

        def test_magnet_at0(self):
            p_cct = bl.trajectory.point_at(
                gantry.first_bending_part_length()+gantry.DL2+0.5).to_p3() + P3(1E-3, 1E-4, 1E-5)

            magnet_cpu = bl.magnetic_field_at(p_cct)

            magnet_gpu_64 = ga64.magnet_at(bl, p_cct)
            magnet_gpu_32 = ga32.magnet_at(bl, p_cct)

            print(f"test_magnet_at0 diff32 = {magnet_cpu - magnet_gpu_32}")
            print(f"test_magnet_at0 diff64 = {magnet_cpu - magnet_gpu_64}")

            print(f"-- test_magnet_at0 all beamline--")
            print(f"magnet_cpu = {magnet_cpu}")
            print(f"magnet_gpu_32 = {magnet_gpu_32}")
            print(f"magnet_gpu_64 = {magnet_gpu_64}")
            # test_magnet_at0 diff32 = [-9.995611723045972e-08, -2.9023106392321585e-07, -2.0517209438075668e-06]
            # test_magnet_at0 diff64 = [-1.5404344466674047e-15, -2.1805474093028465e-15, 0.0]
            self.assertTrue((magnet_cpu-magnet_gpu_32).length() < 1e-5)
            self.assertTrue((magnet_cpu-magnet_gpu_64).length() < 1e-14)

        def test_magnet_at1(self):
            p_cct = bl.trajectory.point_at(
                gantry.first_bending_part_length()+gantry.DL2+0.5).to_p3() + P3(1E-3, 1E-4, 1E-5)

            magnet_cpu = bl.magnetic_field_at(p_cct)

            magnet_gpu_64 = ga64_b128.magnet_at(bl, p_cct)
            magnet_gpu_32 = ga32_b128.magnet_at(bl, p_cct)

            print(f"test_magnet_at1 diff32 = {magnet_cpu - magnet_gpu_32}")
            print(f"test_magnet_at1 diff64 = {magnet_cpu - magnet_gpu_64}")

            print(f"-- test_magnet_at0 all beamline--")
            print(f"magnet_cpu = {magnet_cpu}")
            print(f"magnet_gpu_32 = {magnet_gpu_32}")
            print(f"magnet_gpu_64 = {magnet_gpu_64}")
            # test_cct, diff_32=[2.5088516841798025e-07, -2.2562693963168456e-07, -4.375363960029688e-08]
            # test_cct, diff_64=[2.4424906541753444e-15, 9.43689570931383e-16, 8.881784197001252e-16]
            self.assertTrue((magnet_cpu-magnet_gpu_32).length() < 1e-5)
            self.assertTrue((magnet_cpu-magnet_gpu_64).length() < 1e-14)

        def test_magnet_at2(self):
            p_cct = bl.trajectory.point_at(
                gantry.first_bending_part_length()+gantry.DL2+0.5).to_p3() + P3(1E-3, 1E-4, 1E-5)

            magnet_cpu = bl.magnetic_field_at(p_cct)

            magnet_gpu_64 = ga64_b256.magnet_at(bl, p_cct)
            magnet_gpu_32 = ga32_b256.magnet_at(bl, p_cct)

            print(f"test_magnet_at2 diff32 = {magnet_cpu - magnet_gpu_32}")
            print(f"test_magnet_at2 diff64 = {magnet_cpu - magnet_gpu_64}")

            print(f"-- test_magnet_at0 all beamline--")
            print(f"magnet_cpu = {magnet_cpu}")
            print(f"magnet_gpu_32 = {magnet_gpu_32}")
            print(f"magnet_gpu_64 = {magnet_gpu_64}")
            # test_cct, diff_32=[2.5088516841798025e-07, -2.2562693963168456e-07, -4.375363960029688e-08]
            # test_cct, diff_64=[2.4424906541753444e-15, 9.43689570931383e-16, 8.881784197001252e-16]
            self.assertTrue((magnet_cpu-magnet_gpu_32).length() < 1e-5)
            self.assertTrue((magnet_cpu-magnet_gpu_64).length() < 1e-14)

        def test_magnet_at3(self):
            p_qs = (bl.trajectory.point_at(gantry.first_bending_part_length()+gantry.DL2 +
                                           1.19+gantry.GAP1+gantry.qs3_length/2).to_p3() +
                    P3(10*MM, 10*MM, 10*MM))

            magnet_cpu = bl.magnetic_field_at(p_qs)

            magnet_gpu_64 = ga64.magnet_at(bl, p_qs)
            magnet_gpu_32 = ga32.magnet_at(bl, p_qs)

            print(f"test_magnet_at3 diff32 = {magnet_cpu - magnet_gpu_32}")
            print(f"test_magnet_at3 diff64 = {magnet_cpu - magnet_gpu_64}")

            print(f"-- test_magnet_at0 all beamline--")
            print(f"magnet_cpu = {magnet_cpu}")
            print(f"magnet_gpu_32 = {magnet_gpu_32}")
            print(f"magnet_gpu_64 = {magnet_gpu_64}")
            # test_magnet_at0 diff32 = [-2.2375529054596832e-08, -6.045702764800875e-08, -4.853957882300364e-07]
            # test_magnet_at0 diff64 = [4.0245584642661925e-16, -1.5959455978986625e-16, -3.608224830031759e-16]
            self.assertTrue((magnet_cpu-magnet_gpu_32).length() < 1e-5)
            self.assertTrue((magnet_cpu-magnet_gpu_64).length() < 1e-14)

        def test_magnet_at4(self):
            p_qs = (bl.trajectory.point_at(gantry.first_bending_part_length()+gantry.DL2 +
                                           1.19+gantry.GAP1+gantry.qs3_length/2).to_p3() +
                    P3(10*MM, 10*MM, 10*MM))

            magnet_cpu = bl.magnetic_field_at(p_qs)

            magnet_gpu_64 = ga64_b128.magnet_at(bl, p_qs)
            magnet_gpu_32 = ga32_b128.magnet_at(bl, p_qs)

            print(f"test_magnet_at4 diff32 = {magnet_cpu - magnet_gpu_32}")
            print(f"test_magnet_at4 diff64 = {magnet_cpu - magnet_gpu_64}")

            print(f"-- test_magnet_at0 all beamline--")
            print(f"magnet_cpu = {magnet_cpu}")
            print(f"magnet_gpu_32 = {magnet_gpu_32}")
            print(f"magnet_gpu_64 = {magnet_gpu_64}")
            # test_magnet_at0 diff32 = [-2.2375529054596832e-08, -5.673173734954684e-08, -4.704946270361887e-07]
            # test_magnet_at0 diff64 = [4.0245584642661925e-16, -1.6653345369377348e-16, -3.608224830031759e-16]
            self.assertTrue((magnet_cpu-magnet_gpu_32).length() < 1e-5)
            self.assertTrue((magnet_cpu-magnet_gpu_64).length() < 1e-14)

        def test_magnet_at5(self):
            p_qs = (bl.trajectory.point_at(gantry.first_bending_part_length()+gantry.DL2 +
                                           1.19+gantry.GAP1+gantry.qs3_length/2).to_p3() +
                    P3(10*MM, 10*MM, 10*MM))

            magnet_cpu = bl.magnetic_field_at(p_qs)

            magnet_gpu_64 = ga64_b256.magnet_at(bl, p_qs)
            magnet_gpu_32 = ga32_b256.magnet_at(bl, p_qs)

            print(f"test_magnet_at5 diff32 = {magnet_cpu - magnet_gpu_32}")
            print(f"test_magnet_at5 diff64 = {magnet_cpu - magnet_gpu_64}")

            print(f"-- test_magnet_at0 all beamline--")
            print(f"magnet_cpu = {magnet_cpu}")
            print(f"magnet_gpu_32 = {magnet_gpu_32}")
            print(f"magnet_gpu_64 = {magnet_gpu_64}")
            # test_magnet_at0 diff32 = [-2.2375529054596832e-08, -6.045702764800875e-08, -4.853957882300364e-07]
            # test_magnet_at0 diff64 = [3.885780586188048e-16, -1.5959455978986625e-16, -3.608224830031759e-16]
            self.assertTrue((magnet_cpu-magnet_gpu_32).length() < 1e-5)
            self.assertTrue((magnet_cpu-magnet_gpu_64).length() < 1e-14)

        def test_track(self):
            p = ParticleFactory.create_proton_along(
                bl.trajectory, gantry.first_bending_part_length() + gantry.DL2, 215
            )
            print(f"init p={p}")
            ParticleRunner.run_only(p, bl, 2, 10*MM)
            print(f"track cpu p={p}")

            p = ParticleFactory.create_proton_along(
                bl.trajectory, gantry.first_bending_part_length() + gantry.DL2, 215
            )
            ga32.track_one_particle_with_single_qs(bl, p, 2, 10*MM)
            print(f"track gpu32 p={p}")

            p = ParticleFactory.create_proton_along(
                bl.trajectory, gantry.first_bending_part_length() + gantry.DL2, 215
            )
            ga64_b512.track_one_particle_with_single_qs(bl, p, 2, 10*MM)
            print(f"track gpu64 p={p}")

        def test_track_particles(self):
            """
CPU finish，time = 8.32676386833191 s
p=[6.9952707893127615, 2.8450465238850184, 0.0007167060845681786],v=[137902813.48249766, -106627503.84776628, 276209.6460511479],v0=174317774.94179922
p=[6.991651980302911, 2.84687609876132, 0.0007879913873482307],v=[137570793.41293398, -107061906.14231186, 295395.40762040997],v0=174317774.94179922
p=[7.000264413031172, 2.8482101363154277, 0.000664009633475038],v=[137484972.58009422, -107167541.8711567, 259612.51092981483],v0=174317774.94179922
p=[7.006462737769314, 2.8471267081000597, 0.0006521308980659168],v=[137727592.68641925, -106855557.66272765, 261035.38507345604],v0=174317774.94179922
p=[7.00360052384275, 2.844813163521684, 0.000672045260457023],v=[138068550.86893737, -106419283.32886522, 267163.18213671655],v0=174317774.94179922
GPU32 finish，time = 2.0207936763763428 s
diff=p=[-1.8471008127463051e-06, 7.189076014491036e-07, -8.941999065281616e-08],v=[349.48249766230583, 368.15223371982574, -27.260198852105532],v0=-1.0582007765769958
diff=p=[-9.85242743389847e-07, -7.609035970190803e-07, -1.367907528022948e-07],v=[121.41293397545815, 125.85768814384937, -44.40487959003076],v0=-1.0582007765769958
diff=p=[1.1989198442918791e-06, -1.6289738788977104e-06, -1.3034790655717422e-07],v=[-131.4199057817459, -253.87115670740604, -41.06719518516911],v0=-1.0582007765769958
diff=p=[-1.7435844457125427e-06, -4.910729138885017e-07, -7.633820543479896e-08],v=[376.68641924858093, 378.33727234601974, -24.536801543959882],v0=-1.0582007765769958
diff=p=[-1.0272131580890687e-06, 1.485588944749594e-06, -1.6725037835631316e-07],v=[118.86893737316132, 308.6711347848177, -52.81786328344606],v0=-1.0582007765769958
GPU64 finish，time = 3.7295095920562744 s
diff=p=[0.0, 0.0, -3.0249240612345574e-17],v=[-2.9802322387695312e-08, 0.0, -7.275957614183426e-09],v0=0.0
diff=p=[0.0, 0.0, 9.443400922348744e-17],v=[0.0, 1.4901161193847656e-08, 3.317836672067642e-08],v0=0.0
diff=p=[0.0, -4.440892098500626e-16, 1.7661653389788867e-16],v=[1.1920928955078125e-07, -1.4901161193847656e-08, 5.954643711447716e-08],v0=0.0
diff=p=[0.0, 0.0, 3.5344990823027445e-16],v=[2.9802322387695312e-08, -2.9802322387695312e-08, 1.1886004358530045e-07],v0=0.0
diff=p=[0.0, 4.440892098500626e-16, 8.716985466783456e-17],v=[1.7881393432617188e-07, 2.384185791015625e-07, 2.9802322387695312e-08],v0=0.0
            """
            BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main()
            ip = ParticleFactory.create_proton_along(
                bl.trajectory, gantry.first_bending_part_length() + gantry.DL2, 215
            )
            number = 5
            length = 2
            pp = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_xxp_plane(
                3.5*MM, 7.5*MM, 0.0, number
            )

            ps = ParticleFactory.create_from_phase_space_particles(
                ip, ip.get_natural_coordinate_system(), pp
            )
            ps_copy_for_cpu = [p.copy() for p in ps]
            s = time.time()
            ParticleRunner.run_only(ps_copy_for_cpu, bl, length, 10*MM, number)
            print(f"CPU finish，time = {time.time()-s} s")
            for p in ps_copy_for_cpu:
                print(p)

            ps_copy_for_gpu32 = [p.copy() for p in ps]
            s = time.time()
            ga32.track_multi_particle_for_magnet_with_single_qs(
                bl, ps_copy_for_gpu32, length, 10*MM)
            print(f"GPU32 finish，time = {time.time()-s} s")
            for i in range(len(ps_copy_for_gpu32)):
                print(f"diff={ps_copy_for_cpu[i]-ps_copy_for_gpu32[i]}")

            ps_copy_for_gpu64 = [p.copy() for p in ps]
            s = time.time()
            ga64_b512.track_multi_particle_for_magnet_with_single_qs(
                bl, ps_copy_for_gpu64, length, 10*MM)
            print(f"GPU64 finish，time = {time.time()-s} s")
            for i in range(len(ps_copy_for_gpu64)):
                print(f"diff={ps_copy_for_cpu[i]-ps_copy_for_gpu64[i]}")

        def test_track_particles_multi_beamline(self):
            """
CPU time = 46.470869064331055
p=[7.347045605483152, -5.038190852364211, -0.008126687951838598],v=[153429.15883334717, -174317566.6083884, -222984.23247565638],v0=174317774.94179922
p=[7.334089667832231, -5.030501484758217, -0.008305356296718356],v=[-171544.4749936812, -175781388.6674571, -229869.1911368924],v0=175781619.95982552
p=[7.359979027555547, -5.0455122541805535, -0.007776454452929961],v=[457482.144593381, -172821389.29986903, -212754.63869286922],v0=172822122.75297824
p=[7.347965549877004, -5.036742197173884, -0.00680266146120651],v=[121395.76132774584, -174317594.97821772, -220906.61484601715],v0=174317774.94179922
p=[7.348650531739526, -5.0330294642527775, -0.00660374692590846],v=[227297.55545838206, -175781340.8954671, -217122.2981601591],v0=175781619.95982552
p=[7.34475355176842, -5.03989404652068, -0.007153724839178421],v=[-58106.42448355021, -172821965.44718364, -228721.36663409878],v0=172822122.75297824
p=[7.347558101890118, -5.0374728307509296, -0.007485100785398123],v=[138628.18010870926, -174317578.3359581, -222383.28587266372],v0=174317774.94179922
p=[7.341713322047886, -5.031883095258832, -0.007453546571116006],v=[39058.28246667987, -175781473.9719213, -223465.34914476663],v0=175781619.95982552
p=[7.3519134653687255, -5.042608451774057, -0.007566444532410116],v=[185849.5993363538, -172821879.26466498, -223066.98384745672],v0=172822122.75297824
GPU64
GPU64 time = 5.543462753295898
p=[7.3470456054831565, -5.038190852364211, -0.008126687951838529],v=[153429.1588334337, -174317566.6083884, -222984.232475657],v0=174317774.94179922
p=[7.334089667832242, -5.030501484758215, -0.008305356296717829],v=[-171544.4749934145, -175781388.6674571, -229869.19113686983],v0=175781619.95982552
p=[7.359979027555571, -5.045512254180555, -0.007776454452929699],v=[457482.14459405426, -172821389.29986906, -212754.6386928537],v0=172822122.75297824
p=[7.347965549877003, -5.03674219717388, -0.006802661461206674],v=[121395.76132773953, -174317594.97821763, -220906.61484602414],v0=174317774.94179922
p=[7.348650531739527, -5.033029464252767, -0.006603746925910345],v=[227297.55545837895, -175781340.89546695, -217122.2981601885],v0=175781619.95982552
p=[7.344753551768423, -5.039894046520685, -0.007153724839179523],v=[-58106.424483512936, -172821965.44718373, -228721.36663410708],v0=172822122.75297824
p=[7.347558101890107, -5.0374728307509296, -0.0074851007853986564],v=[138628.18010849392, -174317578.33595803, -222383.2858726776],v0=174317774.94179922
p=[7.341713322047882, -5.031883095258819, -0.007453546571116809],v=[39058.28246653917, -175781473.97192112, -223465.34914477202],v0=175781619.95982552
p=[7.3519134653687255, -5.042608451774063, -0.007566444532410439],v=[185849.59933636745, -172821879.26466507, -223066.98384745035],v0=172822122.75297824
diff=p=[4.440892098500626e-15, 0.0, 6.938893903907228e-17],v=[8.65256879478693e-08, 0.0, -6.111804395914078e-10],v0=0.0
diff=p=[1.0658141036401503e-14, 1.7763568394002505e-15, 5.273559366969494e-16],v=[2.666783984750509e-07, 0.0, 2.255546860396862e-08],v0=0.0
diff=p=[2.398081733190338e-14, -1.7763568394002505e-15, 2.6194324487249787e-16],v=[6.732880137860775e-07, -2.9802322387695312e-08, 1.5512341633439064e-08],v0=0.0
diff=p=[-8.881784197001252e-16, 3.552713678800501e-15, -1.6393136847980827e-16],v=[-6.315531209111214e-09, 8.940696716308594e-08, -6.984919309616089e-09],v0=0.0
diff=p=[8.881784197001252e-16, 1.0658141036401503e-14, -1.884777056648801e-15],v=[-3.1141098588705063e-09, 1.4901161193847656e-07, -2.9423972591757774e-08],v0=0.0diff=p=[2.6645352591003757e-15, -5.329070518200751e-15, -1.1015494072452725e-15],v=[3.727473085746169e-08, -8.940696716308594e-08, -8.294591680169106e-09],v0=0.0
diff=p=[-1.0658141036401503e-14, 0.0, -5.334274688628682e-16],v=[-2.1533924154937267e-07, 5.960464477539063e-08, -1.3882527127861977e-08],v0=0.0
diff=p=[-3.552713678800501e-15, 1.2434497875801753e-14, -8.031769693772617e-16],v=[-1.407024683430791e-07, 1.7881393432617188e-07, -5.384208634495735e-09],v0=0.0
diff=p=[0.0, -6.217248937900877e-15, -3.226585665316861e-16],v=[1.3649696484208107e-08, -8.940696716308594e-08, 6.373738870024681e-09],v0=0.0
            """
            BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main()
            bl1 = HUST_SC_GANTRY().create_beamline()
            bl2 = HUST_SC_GANTRY(qs3_gradient=7).create_beamline()
            bl3 = HUST_SC_GANTRY(qs3_gradient=0).create_beamline()

            p1 = ParticleFactory.create_proton_along(
                bl.trajectory, gantry.first_bending_part_length() + gantry.DL2, 215
            )

            p2 = ParticleFactory.create_proton_along(
                bl.trajectory, gantry.first_bending_part_length() + gantry.DL2, 220
            )

            p3 = ParticleFactory.create_proton_along(
                bl.trajectory, gantry.first_bending_part_length() + gantry.DL2, 210
            )

            ps_cpu1 = [p1.copy(), p2.copy(), p3.copy()]
            ps_cpu2 = [p1.copy(), p2.copy(), p3.copy()]
            ps_cpu3 = [p1.copy(), p2.copy(), p3.copy()]
            ps_gpu32 = [p1.copy(), p2.copy(), p3.copy()]
            ps_gpu64 = [p1.copy(), p2.copy(), p3.copy()]

            print("CPU")
            s = time.time()
            ParticleRunner.run_only(ps_cpu1, bl1, 10, 20*MM, 6)
            ParticleRunner.run_only(ps_cpu2, bl2, 10, 20*MM, 6)
            ParticleRunner.run_only(ps_cpu3, bl3, 10, 20*MM, 6)
            print(f"CPU time = {time.time()-s}")
            for p in ps_cpu1+ps_cpu2 + ps_cpu3:
                print(p)

            print("GPU64")
            s = time.time()
            ps_end = ga64_b512.track_multi_particle_beamlime_for_magnet_with_single_qs(
                [bl1, bl2, bl3], ps_gpu64, 10, 20*MM
            )
            print(f"GPU64 time = {time.time()-s}")

            for ps in ps_end:
                for p in ps:
                    print(p)

            for gid in range(3):
                for pid in range(3):
                    print(f"diff={ps_end[gid][pid]-(ps_cpu1+ps_cpu2 + ps_cpu3)[gid*3+pid]}")

    # Test().test_track_particles_multi_beamline()
    unittest.main(verbosity=1)
