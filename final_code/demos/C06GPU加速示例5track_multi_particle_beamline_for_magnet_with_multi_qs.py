"""
CCT 建模优化代码
GPU 加速示例 track_multi_particle_beamline_for_magnet_with_multi_qs

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
# 单个 qs 测试
bl1 = HUST_SC_GANTRY().create_second_bending_part_beamline()
bl2 = HUST_SC_GANTRY(agcct345_current=-6500).create_second_bending_part_beamline()
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

# Particle[p=(7.409509849267735, -0.028282989447753218, 5.0076184754665586e-05), v=(1809891.9615852616, -174308430.5414393, -330480.4098605619)], rm=2.0558942080656965e-27, e=1.6021766208e-19, speed=174317774.94179922, distance=7.104727865682728]
# Particle[p=(7.407672095037766, -0.021707043421101538, 6.826403597590941e-05), v=(1712449.510406203, -175773222.9259196, -304717.60964732897)], rm=2.0648075176021083e-27, e=1.6021766208e-19, speed=175781619.95982552, distance=7.104727865682728]
# Particle[p=(7.41624265063173, -0.02576953619487038, -0.0016296279614197015), v=(1979688.5833167755, -174306311.08763564, -452450.5604144091)], rm=2.0558942080656965e-27, e=1.6021766208e-19, speed=174317774.94179922, distance=7.104727865682728]
# Particle[p=(7.414874692874203, -0.020345246397707628, -0.0014275222159985121), v=(2006916.31877243, -175769899.78176892, -412961.8574664621)], rm=2.0648075176021083e-27, e=1.6021766208e-19, speed=175781619.95982552, distance=7.104727865682728]

# Particle[p=(7.409509849267735, -0.028282989447752843, 5.0076184754525616e-05), v=(1809891.961585234, -174308430.54143927, -330480.409860578)], rm=2.0558942080656965e-27, e=1.6021766208e-19, speed=174317774.94179922, distance=7.104727865682728]
# Particle[p=(7.407672095037776, -0.021707043421102315, 6.826403597606448e-05), v=(1712449.510406593, -175773222.92591968, -304717.6096473106)], rm=2.0648075176021083e-27, e=1.6021766208e-19, speed=175781619.95982552, distance=7.104727865682728]
# Particle[p=(7.416242650631723, -0.02576953619487027, -0.0016296279614195605), v=(1979688.5833165073, -174306311.08763564, -452450.56041458895)], rm=2.0558942080656965e-27, e=1.6021766208e-19, speed=174317774.94179922, distance=7.104727865682728]
# Particle[p=(7.414874692874213, -0.020345246397708072, -0.0014275222160000985), v=(2006916.3187729006, -175769899.78176892, -412961.85746645875)], rm=2.0648075176021083e-27, e=1.6021766208e-19, speed=175781619.95982552, distance=7.104727865682728]

# Particle[p=(0.0, -3.7470027081099033e-16, 1.3997050046787862e-16), v=(2.7706846594810486e-08, -2.9802322387695312e-08, 1.6123522073030472e-08)], rm=0.0, e=0.0, speed=0.0, distance=0.0]
# Particle[p=(-9.769962616701378e-15, 7.771561172376096e-16, -1.5506801571973927e-16), v=(-3.8999132812023163e-07, 8.940696716308594e-08, -1.83936208486557e-08)], rm=0.0, e=0.0, speed=0.0, distance=0.0]
# Particle[p=(7.105427357601002e-15, -1.1102230246251565e-16, -1.4094628242311558e-16), v=(2.682209014892578e-07, 0.0, 1.798616722226143e-07)], rm=0.0, e=0.0, speed=0.0, distance=0.0]
# Particle[p=(-1.0658141036401503e-14, 4.440892098500626e-16, 1.58640461878079e-15), v=(-4.705507308244705e-07, 0.0, -3.3760443329811096e-09)], rm=0.0, e=0.0, speed=0.0, distance=0.0]