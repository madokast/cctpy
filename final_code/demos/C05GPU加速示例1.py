"""
CCT 建模优化代码
GPU 加速示例(1)

作者：赵润晓
日期：2021年5月4日
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from hust_sc_gantry import HUST_SC_GANTRY
from cctpy import *

ga32 = GPU_ACCELERATOR(float_number_type=GPU_ACCELERATOR.FLOAT32)
ga64 = GPU_ACCELERATOR(float_number_type=GPU_ACCELERATOR.FLOAT64,block_dim_x=512)

# ----------------------vct_length 求三维矢量P3的长度----------------------
v = P3(1,2,3)
print("CPU计算:",v.length())
print("GPU32计算:",ga32.vct_length(v))
print("GPU64计算:",ga64.vct_length(v))

print("CPU和GPU32的差异",v.length()-ga32.vct_length(v))
print("CPU和GPU64的差异",v.length()-ga64.vct_length(v))


# ----------------------vct_print 打印三维矢量----------------------
v=P3(1/3,1/6,1/7)
print("打印三维矢量\n",v)
ga32.vct_print(v)
ga64.vct_print(v)


# ----------------------current_element_B 计算电流元集合，在 p 点产生的磁场----------------------
# 定义一个 CCT
# 简单构造一个 30 度 30 匝的 CCT，就位于全局坐标系
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


# ----------------------magnet_at_qs 计算 qs 磁铁，在 p 点产生的磁场----------------------
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


# ----------------------- magnet_at_beamline_with_single_qs 单一 qs 的 beamline 磁场计算 -----
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


# -------- track_one_particle_with_single_qs 粒子跟踪，电流元 + 单个 QS ----- 
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