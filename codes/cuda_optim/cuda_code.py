cuda_coda = """

/**
 * 机架匝数已经定死
 * 机架位置已经定死
 *
 */

#include <stdio.h>
#include <math.h>
#include "cuda.h"

#define MM (0.001f)
#define DIM (3)
#define PI (3.1415927f)
#define X (0)
#define Y (1)
#define Z (2)
#define Proton_Charge_Quantity (1.6021766208e-19f)
#define Proton_Static_MassKg (1.672621898e-27f)
#define Proton_Static_MassMeV (938.2720813f)
#define Light_Speed (299792458.0f)

// 默认 CCT 分段为 3 度，这是满足计算精度下最粗的分段
#define STEP_KSI (3)

// 粒子运动步长，默认 1mm
#define STEP_RUN (0.001f)

// 倾斜角几个，默认 4 个，即二级场、四极场、六极场、八级场，如果修个这个参数，需要修改方法 ksi_phi_fun 因为为了性能写死了
#define TILE_ANGLE_LENGTH (4)

#define SIN_45 (0.7071067811865476f)
#define COS_45 (0.7071067811865476f)

// 机架移动
#define CCT345_1_MOVE_X (5.871941355773217f)
#define CCT345_1_MOVE_Y (2.3588051447781218f)

// CCT345_2 对称点 对称切线，V 已归一化
#define CCT345_2_SYMMETRY_PX (6.528314834871201f)
#define CCT345_2_SYMMETRY_PY (3.1151989380314977f)
#define CCT345_2_SYMMETRY_VX (0.38268343236508984f)
#define CCT345_2_SYMMETRY_VY (0.9238795325112867f)

// 每匝 CCT 需要的参数 11 个 起点 ksi0，匝弧度 phi0，k[0][1][2][3]，极角 a，ch_eta0，sh_eta0，电流current，起点 phi_start
#define NUMBER_OF_VARIABLES_PER_CCT (11)
#define BICCT_WIND_NUM (128)
#define AGCCT_WIND_NUM_1 (21)
#define AGCCT_WIND_NUM_2 (50)
#define AGCCT_WIND_NUM_3 (50)
#define CCT_TOTAL_WIND_NUM ((BICCT_WIND_NUM+AGCCT_WIND_NUM_1+AGCCT_WIND_NUM_2+AGCCT_WIND_NUM_3)*2)
// 最大CCT匝数和。当前 (128*2+(21+50+50)*2)*2) < 1024。一组线程 1024 个计算一个机架
#define THREAD_NUMBER (1024)
#define DATA_LENGTH_PER_GANTRY (THREAD_NUMBER*NUMBER_OF_VARIABLES_PER_CCT)

// 粒子参数，共 9 个，位置 3个 px py pz 速度 3 个 vx vy vz 动质量 rm 速度 speed 运行距离 distance
#define NUMBER_OF_VARIABLES_PER_PARTICLE (9)
// 粒子返回参数 6 个，位置 和 速度
#define NUMBER_OF_VARIABLES_PER_PARTICLE_RETURN (6)

// QS 磁铁。注意只能放置在 Z=0 的平面上。位置、长度定死，只有四级梯度和六级梯度可以变化
// D 和 RIGHT 已经归一化。2020年11月17日 更新完毕
#define QS3_LENGTH (0.2382791f)
#define QS3_X (6.4182442431135955f)
#define QS3_Y (3.16079166995593f)
#define QS3_DX (0.9238795325112867f)
#define QS3_DY (-0.3826834323650898f)
#define QS3_RIGHT_X (-0.3826834323650897f)
#define QS3_RIGHT_Y (-0.9238795325112867f)

// 简单向量常量操作
__device__ __forceinline__ void vct_cross(float *a, float *b, float *ret);

__device__ __forceinline__ void vct_add_local(float *a_local, float *b);

__device__ __forceinline__ void vct_add(float *a, float *b, float *ret);

__device__ __forceinline__ void vct_sub(float *a, float *b, float *ret);

__device__ __forceinline__ void vct_dot_a_v(float a, float *v);

__device__ __forceinline__ void vct_dot_a_v_ret(float a, float *v, float *ret);

__device__ __forceinline__ void vct_copy(float *src, float *des);

__device__ __forceinline__ float vct_len(float *v);

__device__ __forceinline__ void vct_zero(float *v);

__device__ __forceinline__ void vct_print(float *v);

__device__ __forceinline__ float deg2rad(int deg); // 角度转弧度。本代码中，角度一定是整数。这个方法，以后可能要打表
__device__ __forceinline__ float sin_deg(int deg); // 三角函数，参数整数的角度。这个方法，以后可能要打表。--re. 2020年11月14日 打表意义不大
__device__ __forceinline__ float cos_deg(int deg); // 同上

// 磁场计算 注意，这里计算的不是电流元的磁场，还需要乘以 电流 和 μ0/4π (=1e-7)
__device__ void dB(float *p0, float *p1, float *p, float *ret);

// ksi phi 函数。phi0 即一匝线圈后，大半径转过的弧度。k_tilt_angles 是倾斜角系数 == cot(倾斜角[i])/(i+1)sinh(eta)
__device__ __forceinline__ float ksi_phi_fun(int ksi_deg, float phi_start, float phi0, float *k_tilt_angles);

// 计算 CCT 上 ksi_deg 处的点，存放在 p_ret 中。k_tilt_angles 的含义见 ksi_phi_fun，a 是极角。ch_eta0 = ch(eta0)，sh_eta0 = sh(eta0)
__device__ __forceinline__ void
point_cct(int ksi_deg, float phi_start, float phi0, float *k_tilt_angles, float a, float ch_eta0, float sh_eta0,
          float *p_ret);

// 计算一匝 CCT 线圈在 p 点产生的磁场，注意磁场还要再乘电流 和 μ0/4π (=1e-7)
// ksi_deg0 是计算的起点。phi0、k_tilt_angles、a、ch_eta0、sh_eta0 见 point_cct 函数，p 点为需要计算磁场的点，m_ret 是返回的磁场
__device__ void
dB_cct_wind(int ksi_deg0, float phi_start, float phi0, float *k_tilt_angles, float a, float ch_eta0, float sh_eta0,
            float *p,
            float *m_ret);

// 粒子走一步 m 磁场，p 位置，v 速度，rm 动质量，sp 速率。默认步长 STEP_RUN == 1mm
__device__  __forceinline__  void particle_run_step(float *m, float *p, float *v, float run_mass, float speed);

// 在 Java 的 CCT 建模中，我们移动的是 CCT，将 CCT 平移 / 旋转 到指定的位置，但是这么做开销很大
// 与其移动带有上万个点的 CCT 模型，不如移动只有 1 个点的粒子。 p 为绝对坐标点，pr 为相对于 cct345_1（后偏转段第一段 CCT）的点
// 因此此函数的使用方法为，首先已知绝对坐标下的粒子 p，利用此函数求相对点 pr，然后进行磁场计算，得到的磁场也仅仅是相对磁场，
// 再利用 cct345_1_absolute_m 把相对磁场转为绝对磁场
// 此函数中带有大量的魔数，如果修改了机架模型的长度 / 位置，必须做出调整
__device__ __forceinline__ void cct345_1_relative_point(float *p, float *pr);

// 函数意义见 cct345_1_relative_point
__device__ __forceinline__ void cct345_1_absolute_m(float *mr, float *m);


/***************** DEFINE **********************/

__device__ __forceinline__ void vct_cross(float *a, float *b, float *ret) {
    ret[X] = a[Y] * b[Z] - a[Z] * b[Y];
    ret[Y] = -a[X] * b[Z] + a[Z] * b[X];
    ret[Z] = a[X] * b[Y] - a[Y] * b[X];
}

__device__ __forceinline__ void vct_add_local(float *a_local, float *b) {
    a_local[X] += b[X];
    a_local[Y] += b[Y];
    a_local[Z] += b[Z];
}

__device__ __forceinline__ void vct_add(float *a, float *b, float *ret) {
    ret[X] = a[X] + b[X];
    ret[Y] = a[Y] + b[Y];
    ret[Z] = a[Z] + b[Z];
}

__device__ __forceinline__ void vct_dot_a_v(float a, float *v) {
    v[X] *= a;
    v[Y] *= a;
    v[Z] *= a;
}

__device__ __forceinline__ void vct_dot_a_v_ret(float a, float *v, float *ret) {
    ret[X] = v[X] * a;
    ret[Y] = v[Y] * a;
    ret[Z] = v[Z] * a;
}

__device__ __forceinline__ void vct_copy(float *src, float *des) {
    des[X] = src[X];
    des[Y] = src[Y];
    des[Z] = src[Z];
}

__device__ __forceinline__ float vct_len(float *v) {
    return sqrtf(v[X] * v[X] + v[Y] * v[Y] + v[Z] * v[Z]);
}

__device__ __forceinline__ void vct_zero(float *v) {
    v[X] = 0.0f;
    v[Y] = 0.0f;
    v[Z] = 0.0f;
}

__device__ __forceinline__ void vct_print(float *v) {
    printf(" --  %f, %f, %f --  ", v[X], v[Y], v[Z]);
}

__device__ __forceinline__ void vct_sub(float *a, float *b, float *ret) {
    ret[X] = a[X] - b[X];
    ret[Y] = a[Y] - b[Y];
    ret[Z] = a[Z] - b[Z];
}

__device__ __forceinline__ float deg2rad(int deg) {
    return ((float) deg) * PI / 180.0f;
}

__device__ __forceinline__ float sin_deg(int deg) {
    return __sinf(deg2rad(deg));
}

__device__ __forceinline__ float cos_deg(int deg) {
    return __cosf(deg2rad(deg));
}

__device__ void dB(float *p0, float *p1, float *p, float *ret) {
    float p01[DIM];
    float r[DIM];
    float rr;

    vct_sub(p1, p0, p01); // p01 = p1 - p0

    vct_add(p0, p1, r); // r = p0 + p1

    vct_dot_a_v(0.5f, r); // r = (p0 + p1)/2

    vct_sub(p, r, r); // r = p - r

    rr = vct_len(r); // rr = len(r)

    vct_cross(p01, r, ret); // ret = p01 x r

    rr = 1.0f / rr / rr / rr; // changed

    vct_dot_a_v(rr, ret); // rr . (p01 x r)
}

__device__ __forceinline__ float ksi_phi_fun(int ksi_deg, float phi_start, float phi0, float *k_tilt_angles) {
    // k 数组长度是 TILE_ANGLE_LENGTH，默认 4
    float ksi_rad = deg2rad(ksi_deg);

    return phi0 / (2.0f * PI) * ksi_rad +
           k_tilt_angles[0] * sin_deg(ksi_deg) +
           k_tilt_angles[1] * sin_deg(2 * ksi_deg) +
           k_tilt_angles[2] * sin_deg(3 * ksi_deg) +
           k_tilt_angles[3] * sin_deg(4 * ksi_deg) +
           phi_start;
}

__device__ __forceinline__ void
point_cct(int ksi_deg, float phi_start, float phi0, float *k_tilt_angles, float a, float ch_eta0, float sh_eta0,
          float *p_ret) {
    float phi = ksi_phi_fun(ksi_deg, phi_start, phi0, k_tilt_angles);
    float temp = a / (ch_eta0 - cos_deg(ksi_deg));

    p_ret[X] = temp * sh_eta0 * cosf(phi); // 太惨了，这个地方不能打表
    p_ret[Y] = temp * sh_eta0 * sinf(phi); // 太惨了，这个地方不能打表
    p_ret[Z] = temp * sin_deg(ksi_deg);
}

__device__ void
dB_cct_wind(int ksi_deg0, float phi_start, float phi0, float *k_tilt_angles, float a, float ch_eta0,
            float sh_eta0, float *p, float *m_ret) {
    int end_ksi_deg = ksi_deg0 + 360;
    float pre_point[3];
    float cur_point[3];
    float delta_B[3];

    point_cct(ksi_deg0, phi_start, phi0, k_tilt_angles, a, ch_eta0, sh_eta0, pre_point); // 起点

    vct_zero(m_ret); // m = 0,0,0

    while (ksi_deg0 < end_ksi_deg) {
        ksi_deg0 += STEP_KSI;

        point_cct(ksi_deg0, phi_start, phi0, k_tilt_angles, a, ch_eta0, sh_eta0, cur_point); // 下一个点

        dB(pre_point, cur_point, p, delta_B); // 计算磁场

        vct_add_local(m_ret, delta_B);

        vct_copy(cur_point, pre_point); // pre = cur
    }
}

// p 绝对点，pr 相对点
__device__ __forceinline__ void cct345_1_relative_point(float *p, float *pr) {
    float px = p[X];
    float py = p[Y];
    float pz = p[Z];

    // 这两个魔数来自下面向量的相反数
    //Vector3 moving = afterDl2.moveSelf(
    //                directDl2.rotateSelf(BaseUtils.Converter.angleToRadian(-90))
    //                        .changeLengthSelf(secondBend.trajectoryBigRPart2))
    //                .toVector3();
    px -= CCT345_1_MOVE_X;
    py -= CCT345_1_MOVE_Y;

    // 下面是旋转
    // float r_phi = deg2rad(-135);
    // float c = -0.70710678f; // cos(-135) = - cos45
    // float s = -0.70710678f; // sin(-135) = - sin45
    // p[X] = c * x0 - s * y0;
    // p[Y] = s * x0 + c * y0;

    pr[X] = (px - py) * -SIN_45;
    pr[Y] = (px + py) * -COS_45;

    // xz 对称
    pr[Y] *= -1.f;

    // 填上 Z
    pr[Z] = pz;
}

// 函数意义见 cct345_1_relative_point
__device__ __forceinline__ void cct345_1_absolute_m(float *mr, float *m) {
    float mrx = mr[X];
    float mry = mr[Y];
    float mrz = mr[Z];

    // z
    m[Z] = mrz;

    // 对称回去
    mry *= -1.0f;

    // 旋转
    m[X] = (mrx + mry) * -SIN_45;
    m[Y] = (mrx - mry) * SIN_45;

    // 莫名其妙需要全反，我不知道为什么
    vct_dot_a_v(-1.0f, m);

//m[Y] *= -1.f;
//
//        float r_phi = deg2rad(135);
//        float c = cosf(r_phi); // -sin45
//        float s = sinf(r_phi); // sin45
//
//        float x0 = m[X];
//        float y0 = m[Y];
//
//        m[X] = c * x0 - s * y0;
//        m[Y] = s * x0 + c * y0;
//
//        vct_dot_a_v(-1.0f, m);
}

__device__ __forceinline__ void cct345_2_absolute_m(float *mr, float *m) {
    float mrx = mr[X]; // p1x
    float mry = mr[Y]; // p1y
    float mrz = mr[Z];

    // 对称回去
    float project, proj_x, proj_y, p1_normal_x, p1_normal_y, t[3];
    project = mrx * CCT345_2_SYMMETRY_VX + mry * CCT345_2_SYMMETRY_VY;
    proj_x = project * CCT345_2_SYMMETRY_VX;
    proj_y = project * CCT345_2_SYMMETRY_VY;
    p1_normal_x = mrx - proj_x;
    p1_normal_y = mry - proj_y;

    t[X] = mrx - 2 * p1_normal_x;
    t[Y] = mry - 2 * p1_normal_y;

    // 莫名其妙需要反 z
    t[Z] = mrz;

    cct345_1_absolute_m(t, m);
}

// p 绝对点，pr 相对点
__device__ __forceinline__ void cct345_2_relative_point(float *p, float *pr) {
    float px = p[X];
    float py = p[Y];
    float pz = p[Z];

    float spx, spy, project, diagx, diagy, sppx, sppy, t[3];
    // 做对称，关于xy平面中的一条线对称，对称线上的点为 (sy_x,sy_y)，对称线的切向为 (sy_xd,sy_yd) 2020年11月16日 测试通过
    spx = px - CCT345_2_SYMMETRY_PX;
    spy = py - CCT345_2_SYMMETRY_PY;
    project = spx * CCT345_2_SYMMETRY_VX + spy * CCT345_2_SYMMETRY_VY;
    diagx = 2.0f * CCT345_2_SYMMETRY_VX * project;
    diagy = 2.0f * CCT345_2_SYMMETRY_VY * project;
    sppx = diagx - spx;
    sppy = diagy - spy;
    px = CCT345_2_SYMMETRY_PX + sppx;
    py = CCT345_2_SYMMETRY_PY + sppy;

    // 之后和 cct345_1_relative_point 一致
    t[X] = px;
    t[Y] = py;
    t[Z] = pz;

    cct345_1_relative_point(t, pr);
}

// 粒子走一步 m 磁场，p 位置，v 速度，rm 动质量，sp 速率
__device__  __forceinline__  void particle_run_step(float *m, float *p, float *v, float run_mass, float speed) {
    float a[3]; // 加速度
    float t;    // 运动时间
    float d[3]; // 位置变化 速度变化

    // q v b
    vct_cross(v, m, a); // a = v*b

    vct_dot_a_v(Proton_Charge_Quantity / run_mass, a); // a = q v b / mass 加速度

    t = STEP_RUN / speed; // 运动时长

    vct_dot_a_v_ret(t, v, d); // d = t v 位置变化

    vct_add_local(p, d); // p+=d

    vct_dot_a_v_ret(t, a, d); // d = t a 速度变化

    vct_add_local(v, d); // v+=d
}


// 计算CCT345_1磁场，data是机架数据， p 所求点， m_ret 返回值必须是线程共享变量，内部已同步。
__device__ void magnet_at_cct345_1(float *data, float *p,/*__shared__*/float *m_ret) {
    unsigned int tid = threadIdx.x;
    float m_per_wind[3];

    if (tid == 0) {
        vct_zero(m_ret);
    }

    __syncthreads();

    if (tid < CCT_TOTAL_WIND_NUM) {
        dB_cct_wind(
                (int)(*((float *) (data + tid * NUMBER_OF_VARIABLES_PER_CCT + 0))), // ksi_deg0
                *(data + tid * NUMBER_OF_VARIABLES_PER_CCT + 10), // start_phi
                *(data + tid * NUMBER_OF_VARIABLES_PER_CCT + 1), // phi0
                data + tid * NUMBER_OF_VARIABLES_PER_CCT + 2, // k
                *(data + tid * NUMBER_OF_VARIABLES_PER_CCT + 6), // a
                *(data + tid * NUMBER_OF_VARIABLES_PER_CCT + 7), // ch_eta0
                *(data + tid * NUMBER_OF_VARIABLES_PER_CCT + 8), // sh_eta0
                p, m_per_wind
        );

        vct_dot_a_v((*(data + tid * NUMBER_OF_VARIABLES_PER_CCT + 9)) * 1e-7f, m_per_wind);

        atomicAdd(&m_ret[X], m_per_wind[X]);
        atomicAdd(&m_ret[Y], m_per_wind[Y]);
        atomicAdd(&m_ret[Z], m_per_wind[Z]);
    }

    __syncthreads();
}

// 计算CCT345_1和2的磁场。data是机架数据， p 代表两个点，p[1] 是 CCT345_1 看到的粒子位置，p[2] 是 CCT345_2 看到的粒子位置
// m_ret 返回值必须是线程共享变量，内部已同步。
__device__ void magnet_at_cct345_1_and_2(float *data, float p[][3],/*__shared__*/float m_ret[][3]) {
    /**
     * 2020年11月16日 测试通过
     * 感想：为了能够同时调动 1024 个线程，不知绞尽多少脑汁
     */
    unsigned int tid = threadIdx.x; // 0-1023
    unsigned int tid_512 = tid % 512; // 0-511->0-511 // 512-1023->0-511
    unsigned int tid_part_1_or_2 = tid / 512; // 0-511->0 // 512-1023->1
    float m_per_wind[2][3];

    if (tid == 0) {
        vct_zero(m_ret[0]);
        vct_zero(m_ret[1]);
    }

    __syncthreads();

    if (tid_512 < CCT_TOTAL_WIND_NUM) {
        dB_cct_wind(
                (int)(*((float *) (data + tid_512 * NUMBER_OF_VARIABLES_PER_CCT + 0))), // ksi_deg0
                *(data + tid_512 * NUMBER_OF_VARIABLES_PER_CCT + 10), // start_phi
                *(data + tid_512 * NUMBER_OF_VARIABLES_PER_CCT + 1), // phi0
                data + tid_512 * NUMBER_OF_VARIABLES_PER_CCT + 2, // k
                *(data + tid_512 * NUMBER_OF_VARIABLES_PER_CCT + 6), // a
                *(data + tid_512 * NUMBER_OF_VARIABLES_PER_CCT + 7), // ch_eta0
                *(data + tid_512 * NUMBER_OF_VARIABLES_PER_CCT + 8), // sh_eta0
                p[tid_part_1_or_2], m_per_wind[tid_part_1_or_2]
        );

        vct_dot_a_v((*(data + tid_512 * NUMBER_OF_VARIABLES_PER_CCT + 9)) * 1e-7f, m_per_wind[tid_part_1_or_2]);

        atomicAdd(&m_ret[tid_part_1_or_2][X], m_per_wind[tid_part_1_or_2][X]);
        atomicAdd(&m_ret[tid_part_1_or_2][Y], m_per_wind[tid_part_1_or_2][Y]);
        atomicAdd(&m_ret[tid_part_1_or_2][Z], m_per_wind[tid_part_1_or_2][Z]);
    }

    __syncthreads();
}

__device__ __forceinline__ void magnet_at_qs3(float gradient, float second_gradient, float *p, float *m_ret) {
    //// QS 磁铁。注意只能放置在 Z=0 的平面上。位置、长度定死，只有四级梯度和六级梯度可以变化
    //// D 和 RIGHT 已经归一化
    //#define QS3_LENGTH (0.2382791f)
    //#define QS3_X (6.226576290344914f)
    //#define QS3_Y (3.0814002044468563f)
    //#define QS3_DX (0.9238795325112867f)
    //#define QS3_DY (-0.3826834323650898f)
    //#define QS3_RIGHT_X (-0.3826834323650897f)
    //#define QS3_RIGHT_Y (-0.9238795325112867f)

    float px = p[X];
    float py = p[Y];

    float p0p[2], project, p_online[2], p_online_2_p[2], rx, ry, bx, by;

    p0p[X] = px - QS3_X;
    p0p[Y] = py - QS3_Y;

    project = p0p[X] * QS3_DX + p0p[Y] * QS3_DY;

    if (project <= 0 || project >= QS3_LENGTH) {
        vct_zero(m_ret);
    } else {
        p_online[X] = QS3_X + project * QS3_DX;
        p_online[Y] = QS3_Y + project * QS3_DY;

        p_online_2_p[X] = px - p_online[X];
        p_online_2_p[Y] = py - p_online[Y];

        rx = p_online_2_p[X] * QS3_RIGHT_X + p_online_2_p[Y] * QS3_RIGHT_Y;
        ry = p[Z];

        bx = -gradient * ry + second_gradient * (rx * ry);
        by = -gradient * rx + 0.5f * second_gradient * (rx * rx - ry * ry);

        m_ret[X] = bx * QS3_RIGHT_X;
        m_ret[Y] = bx * QS3_RIGHT_Y;
        m_ret[Z] = by;
    }

}

__global__ void
particle_run_parallel(float *data, float *qs_data, int* particle_number, float *particle_data, float *ret) {
    int particle_id;
    unsigned int gantry_id = blockIdx.x;
    unsigned int tid = threadIdx.x;

    float qs_q = qs_data[0];
    float qs_s = qs_data[1];

    float* this_gantry_data = data + (gantry_id * NUMBER_OF_VARIABLES_PER_CCT * THREAD_NUMBER);

    // 粒子位置、速度、动质量、速率、运行长度、总长度
    float p[3], v[3], rm, speed, distance, length;

    float pr_cct[2][3]; // 相对位置点（相对于 CCT345_1 和 CCT345_2）
    float m[2][3]; // 绝对磁场

    __shared__ float mr[2][3]; // 相对磁场

    for (particle_id = 0; particle_id < *particle_number; particle_id++) {
        distance = 0.0f;

        p[X] = particle_data[particle_id * 9 + 0];
        p[Y] = particle_data[particle_id * 9 + 1];
        p[Z] = particle_data[particle_id * 9 + 2];

        v[X] = particle_data[particle_id * 9 + 3];
        v[Y] = particle_data[particle_id * 9 + 4];
        v[Z] = particle_data[particle_id * 9 + 5];

        rm = particle_data[particle_id * 9 + 6];
        speed = particle_data[particle_id * 9 + 7];
        length = particle_data[particle_id * 9 + 8];

        while (distance < length) {
            // 相对点
            cct345_1_relative_point(p, pr_cct[0]);
            cct345_2_relative_point(p, pr_cct[1]);
            // 磁场
            magnet_at_cct345_1_and_2(this_gantry_data, pr_cct, mr);
            // 绝对磁场
            cct345_1_absolute_m(mr[0], m[0]);
            cct345_2_absolute_m(mr[1], m[1]);

            // 总绝对磁场，再加上 qs3 的磁场
            vct_add_local(m[0], m[1]);
            magnet_at_qs3(qs_q, qs_s, p, m[1]);
            vct_add_local(m[0], m[1]);

            // 粒子运动
            particle_run_step(m[0], p, v, rm, speed);

            distance += STEP_RUN;
        }

        if (tid == 0) {
            // 写入 ret
            vct_copy(p, ret + (gantry_id * (*particle_number) * NUMBER_OF_VARIABLES_PER_PARTICLE_RETURN +
                               particle_id * NUMBER_OF_VARIABLES_PER_PARTICLE_RETURN + 0));

            vct_copy(v, ret + (gantry_id * (*particle_number) * NUMBER_OF_VARIABLES_PER_PARTICLE_RETURN +
                               particle_id * NUMBER_OF_VARIABLES_PER_PARTICLE_RETURN + 3));

            // printf("CUDA::gid={%d},pid={%d},p={%f,%f,%f},v={%f,%f,%f}\\n",gantry_id,particle_id,p[X],p[Y],p[Z],v[X],v[Y],v[Z]);
        }
    }
    
    if (tid == 0) {
        printf("CUDA::gid={%d}\\n",gantry_id+1);
    }
}

/**
 * 关于 data 参数，一维数组，全部为 float32
 * 长度 = 机架数目 * 1024 * 11
 * 机架数目传入核函数，由 blockIdx.x 区分，不同块运行不同的机架
 * 同一个块中，运行一个机架，并依次运行多个粒子
 *
 * 除了 data 表示机架中 CCT 参数外，还有 qs_q 和 qs_s 参数，分别传入 机架数目 长度的 float32 数组
 * 同样由 blockIdx.x 区分
 *
 * 粒子参数，每个机架都运行相同数目的粒子，粒子参数也相同，
 * 粒子的参数有 px py pz vx vy vz rm speed distance 9 个，注意使用时应该先复制
 *
 */
 
 """

# -*- coding: utf-8 -*-
from typing import List

import numpy as np

import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule(cuda_coda)

f = mod.get_function("particle_run_parallel")


def cuda_particle_run_parallel(gantry_number: int, cct_data: np.ndarray, qs_data: np.ndarray, particle_number: np.ndarray,
                          particle_data: np.ndarray, result: np.ndarray) -> None:
    # float *data, float *qs_data, int particle_number, float *particle_data, float *ret
    f(drv.In(cct_data), drv.In(qs_data), drv.In(particle_number), drv.In(particle_data), drv.Out(result),
      block=(1024, 1, 1), grid=(gantry_number, 1))