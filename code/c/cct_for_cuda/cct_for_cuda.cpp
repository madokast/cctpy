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

/*__device__ __forceinline__*/ void add3d(float* a, float* b, float* ret)
{
    ret[X] = a[X] + b[X];
    ret[Y] = a[Y] + b[Y];
    ret[Z] = a[Z] + b[Z];
}

/*__device__ __forceinline__*/ void add3d_local(float* a_local, float* b)
{
    a_local[X] += b[X];
    a_local[Y] += b[Y];
    a_local[Z] += b[Z];
}

/*__device__ __forceinline__*/ void copy3d(float* src, float* des)
{
    des[X] = src[X];
    des[Y] = src[Y];
    des[Z] = src[Z];
}

/*__device__ __forceinline__*/ void cross3d(float* a, float* b, float* ret)
{
    ret[X] = a[Y] * b[Z] - a[Z] * b[Y];
    ret[Y] = -a[X] * b[Z] + a[Z] * b[X];
    ret[Z] = a[X] * b[Y] - a[Y] * b[X];
}

/*__device__ __forceinline__*/ void dot_a_v(float* a, float* v)
{
    v[X] *= *a;
    v[Y] *= *a;
    v[Z] *= *a;
}

/*__device__ __forceinline__*/ void dot_v_v(float* v1, float* v2, float* ret)
{
    *ret = v1[X] * v2[X] + v1[Y] * v2[Y] + v1[Z] * v2[Z];
}

/*__device__ __forceinline__*/ void len3d(float* v, float* len)
{
    *len = sqrtf(v[X] * v[X] + v[Y] * v[Y] + v[Z] * v[Z]);
}



int main()
{
	printf("hello, world");
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
