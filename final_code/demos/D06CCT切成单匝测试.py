"""
CCT 建模优化代码
切成单匝 CCT

作者：赵润晓
日期：2021年6月17日
"""



from math import pi
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *


# 先建立一个最简单的 CCT
cct = CCT(
    local_coordinate_system=LocalCoordinateSystem.global_coordinate_system(),
    big_r=1.0,
    small_r=100*MM,
    bending_angle=45,
    tilt_angles=[30],
    winding_number=45,
    current=10000,
    starting_point_in_ksi_phi_coordinate=P2.origin(),
    end_point_in_ksi_phi_coordinate=P2(x=45*2*pi, y=BaseUtils.angle_to_radian(-45))
)

# Plot2.plot_cct_path3d_in_2d(cct)

# ccts = CCT.cut_to_single_winding_cct(cct)
# for i in range(len(ccts)):
#     Plot2.plot_cct_path3d_in_2d(ccts[i],describe="r-" if i%2==0 else "k-")

# Plot2.equal()
# Plot2.show()


# -------------------------------------------------------------------------------------

bl = HUST_SC_GANTRY().create_first_bending_part_beamline()

bz = bl.magnetic_field_bz_along(step=10*MM)
# Plot2.plot_p2s(bz)

# to ccts
ms = CombinedMagnet()
for m in bl.get_magnets():
    if isinstance(m,CCT):
        print(f"cct切开{m}")
        ms.add_all(CCT.cut_to_single_winding_cct(m))
    else:
        print(f"非CCT={m}")
        ms.add(m)

bz_ms = ms.magnetic_field_bz_along(
    line2=bl,step=10*MM
)

# Plot2.plot_p2s(bz_ms,describe='y--')

# Plot2.info()
# Plot2.show()

# --------------------------------------------------------------------

# 粒子跟踪测试
print("粒子跟踪测试")

ga = GPU_ACCELERATOR(cpu_mode=True)

r = ga.track_multi_particle_beamline_for_magnet_with_multi_qs(
    bls = [bl],
    ps = [ParticleFactory.create_proton_along(bl,kinetic_MeV=215)],
    distance=bl.get_length(),
    footstep=10*MM
)

print(r)

bl_ms = Beamline(trajectory=None)
bl_ms.magnets = ms.get_magnets()

r = ga.track_multi_particle_beamline_for_magnet_with_multi_qs(
    bls = [bl_ms],
    ps = [ParticleFactory.create_proton_along(bl,kinetic_MeV=215)],
    distance=bl.get_length(),
    footstep=10*MM
)

print(r)
