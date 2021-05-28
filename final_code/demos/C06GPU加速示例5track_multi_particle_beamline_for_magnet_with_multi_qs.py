"""
CCT 建模优化代码
GPU 加速示例 track_multi_particle_beamline_for_magnet_with_multi_qs 2

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


# ---- track_multi_particle_beamline_for_magnet_with_multi_qs  -----
# 多个 qs 测试
bl1 = HUST_SC_GANTRY().create_first_bending_part_beamline()
bl2 = HUST_SC_GANTRY(agcct12_current=-3000).create_first_bending_part_beamline()
p1 = ParticleFactory.create_proton_along(bl1,kinetic_MeV=215)
p2 = ParticleFactory.create_proton_along(bl1,kinetic_MeV=220)
p11_cpu,p21_cpu = p1.copy(),p2.copy()
p12_cpu,p22_cpu = p1.copy(),p2.copy()
p1_gpu32,p2_gpu32 = p1.copy(),p2.copy()
print("track_multi_particle_beamline_for_magnet_with_multi_qs")
footstep=100*MM

ParticleRunner.run_only([p11_cpu,p21_cpu],bl1,bl1.get_length(),footstep)
ParticleRunner.run_only([p12_cpu,p22_cpu],bl2,bl2.get_length(),footstep)
print(p11_cpu.detailed_info())
print(p21_cpu.detailed_info())
print(p12_cpu.detailed_info())
print(p22_cpu.detailed_info())

pll = ga64.track_multi_particle_beamline_for_magnet_with_multi_qs(
    [bl1,bl2],[p1_gpu32,p2_gpu32],bl1.get_length(),footstep
)
print(pll[0][0].detailed_info())
print(pll[0][1].detailed_info())
print(pll[1][0].detailed_info())
print(pll[1][1].detailed_info())

print((p11_cpu-pll[0][0]).detailed_info())
print((p21_cpu-pll[0][1]).detailed_info())
print((p12_cpu-pll[1][0]).detailed_info())
print((p22_cpu-pll[1][1]).detailed_info())

# Particle[p=(3.687315812380205, 1.548315945537494, -0.003352065021200123), v=(119474899.55705348, 126923892.97270872, -352485.58348381834)], rm=2.0558942080656965e-27, e=1.6021766208e-19, speed=174317774.94179922, distance=4.149802255227576]
# Particle[p=(3.6902588367117777, 1.5457564023956827, -0.003130092145109502), v=(121103380.25921707, 127398432.4832374, -329143.847303274)], rm=2.0648075176021083e-27, e=1.6021766208e-19, speed=175781619.95982552, distance=4.149802255227576]
# Particle[p=(3.687478027359044, 1.548166411708122, -0.0039737849740400085), v=(119517094.38436584, 126885165.8237462, -432788.91453453223)], rm=2.0558942080656965e-27, e=1.6021766208e-19, speed=174317774.94179922, distance=4.149802255227576]
# Particle[p=(3.6903875694662838, 1.545650405905456, -0.0037170902813668744), v=(121140927.39202842, 127363741.12784411, -405102.59567924944)], rm=2.0648075176021083e-27, e=1.6021766208e-19, speed=175781619.95982552, distance=4.149802255227576]

# Particle[p=(3.687315812380205, 1.5483159455374929, -0.0033520650212005175), v=(119474899.55705343, 126923892.97270869, -352485.58348386886)], rm=2.0558942080656965e-27, e=1.6021766208e-19, speed=174317774.94179922, distance=4.149802255227576]
# Particle[p=(3.6902588367117777, 1.5457564023956827, -0.0031300921451098366), v=(121103380.25921713, 127398432.4832374, -329143.8473033173)], rm=2.0648075176021083e-27, e=1.6021766208e-19, speed=175781619.95982552, distance=4.149802255227576]
# Particle[p=(3.6874780273590444, 1.5481664117081209, -0.003973784974042229), v=(119517094.3843659, 126885165.82374611, -432788.91453478823)], rm=2.0558942080656965e-27, e=1.6021766208e-19, speed=174317774.94179922, distance=4.149802255227576]
# Particle[p=(3.6903875694662838, 1.545650405905456, -0.0037170902813662204), v=(121140927.39202844, 127363741.12784408, -405102.59567917266)], rm=2.0648075176021083e-27, e=1.6021766208e-19, speed=175781619.95982552, distance=4.149802255227576]

# Particle[p=(0.0, 1.1102230246251565e-15, 3.946495907847236e-16), v=(4.470348358154297e-08, 2.9802322387695312e-08, 5.052424967288971e-08)], rm=0.0, e=0.0, speed=0.0, distance=0.0]
# Particle[p=(0.0, 0.0, 3.3480163086352377e-16), v=(-5.960464477539063e-08, 0.0, 4.330649971961975e-08)], rm=0.0, e=0.0, speed=0.0, distance=0.0]
# Particle[p=(-4.440892098500626e-16, 1.1102230246251565e-15, 2.220446049250313e-15), v=(-5.960464477539063e-08, 8.940696716308594e-08, 2.5599729269742966e-07)], rm=0.0, e=0.0, speed=0.0, distance=0.0]
# Particle[p=(0.0, 0.0, -6.539907504432563e-16), v=(-1.4901161193847656e-08, 2.9802322387695312e-08, -7.677590474486351e-08)], rm=0.0, e=0.0, speed=0.0, distance=0.0]