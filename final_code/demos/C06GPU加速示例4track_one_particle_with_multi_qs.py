"""
CCT 建模优化代码
GPU 加速示例(3)

作者：赵润晓
日期：2021年5月6日
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from hust_sc_gantry import HUST_SC_GANTRY
from cctpy import *

ga32 = GPU_ACCELERATOR(float_number_type=GPU_ACCELERATOR.FLOAT32)
ga64 = GPU_ACCELERATOR(float_number_type=GPU_ACCELERATOR.FLOAT64,block_dim_x=512)


# ----- track_one_particle_with_multi_qs -----
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
ga32.track_one_particle_with_multi_qs(bl,particle_gpu32,bl.get_length(),footstep=footstep)
ga64.track_one_particle_with_multi_qs(bl,particle_gpu64,bl.get_length(),footstep=footstep)
print("track_one_particle_with_multi_qs")
print("CPU计算结果: ",particle_cpu.detailed_info())
print("GPU32计算结果: ",particle_gpu32.detailed_info())
print("GPU64计算结果: ",particle_gpu64.detailed_info())
print("GPU32计算和CPU对比: ",(particle_cpu-particle_gpu32).detailed_info())
print("GPU64计算和CPU对比: ",(particle_cpu-particle_gpu64).detailed_info())
# track_one_particle_with_multi_qs
# CPU计算结果:  Particle[p=(7.409509849267735, -0.028282989447753218, 5.0076184754665586e-05), v=(1809891.9615852616, -174308430.5414393, -330480.4098605619)], rm=2.0558942080656965e-27, e=1.6021766208e-19, speed=174317774.94179922, distance=7.104727865682728]
# GPU32计算结果:  Particle[p=(7.409510612487793, -0.02828289568424225, 5.0118236686103046e-05), v=(1809917.875, -174308416.0, -330476.3125)],
# rm=2.0558942007434142e-27, e=1.602176597458587e-19, speed=174317776.0, distance=7.104727745056152]
# GPU64计算结果:  Particle[p=(7.409509849267735, -0.028282989447752843, 5.0076184754525616e-05), v=(1809891.961585234, -174308430.54143927, -330480.409860578)], rm=2.0558942080656965e-27, e=1.6021766208e-19, speed=174317774.94179922, distance=7.104727865682728]
# GPU32计算和CPU对比:  Particle[p=(-7.632200578200354e-07, -9.376351096934687e-08, -4.2051931437459694e-08), v=(-25.91341473837383, -14.541439294815063, -4.097360561892856)], rm=7.322282306994799e-36, e=2.3341413164924317e-27, speed=-1.0582007765769958, distance=1.2062657539502197e-07]
# GPU64计算和CPU对比:  Particle[p=(0.0, -3.7470027081099033e-16, 1.3997050046787862e-16), v=(2.7706846594810486e-08, -2.9802322387695312e-08, 1.6123522073030472e-08)], rm=0.0, e=0.0, speed=0.0, distance=0.0]

# track_one_particle_with_single_qs 对比
# CPU计算结果:  Particle[p=(7.409509849267735, -0.028282989447753218, 5.0076184754665586e-05), v=(1809891.9615852616, -174308430.5414393, -330480.4098605619)], rm=2.0558942080656965e-27, e=1.6021766208e-19, speed=174317774.94179922, distance=7.104727865682728]
# GPU32计算结果:  Particle[p=(7.409510612487793, -0.02828289568424225, 5.0118236686103046e-05), v=(1809917.875, -174308416.0, -330476.3125)], rm=2.0558942007434142e-27, e=1.602176597458587e-19, speed=174317776.0, distance=7.104727745056152]
# GPU64计算结果:  Particle[p=(7.409509849267735, -0.028282989447752843, 5.0076184754525616e-05), v=(1809891.961585234, -174308430.54143927, -330480.409860578)], rm=2.0558942080656965e-27, e=1.6021766208e-19, speed=174317774.94179922, distance=7.104727865682728]
# GPU32计算和CPU对比:  Particle[p=(-7.632200578200354e-07, -9.376351096934687e-08, -4.2051931437459694e-08), v=(-25.91341473837383, -14.541439294815063, -4.097360561892856)], rm=7.322282306994799e-36, e=2.3341413164924317e-27, speed=-1.0582007765769958, distance=1.2062657539502197e-07]
# GPU64计算和CPU对比:  Particle[p=(0.0, -3.7470027081099033e-16, 1.3997050046787862e-16), v=(2.7706846594810486e-08, -2.9802322387695312e-08, 1.6123522073030472e-08)], rm=0.0, e=0.0, speed=0.0, distance=0.0]


# ------------ 真正的多qs -------
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