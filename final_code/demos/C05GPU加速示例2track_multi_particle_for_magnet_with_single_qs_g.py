"""
CCT 建模优化代码 
GPU 加速示例(2) track_multi_particle_for_magnet_with_single_qs_g

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