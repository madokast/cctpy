"""
CCT 建模优化代码
单粒子跟踪和轨迹绘制

作者：赵润晓
日期：2021年5月3日
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys

sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

beamline = (
    Beamline.set_start_point(P2.zeros())
    .first_drift(direct=P2.x_direct(),length=1)
    .append_qs(
        length=0.27,
        gradient=-10,
        second_gradient=0,
        aperture_radius=60*MM
    ).append_drift(length=1)
    .append_qs(
        length=0.27,
        gradient=10,
        second_gradient=0,
        aperture_radius=60*MM
    ).append_drift(length=1)
)

ideal_particle = ParticleFactory.create_proton_along(
    trajectory=beamline,
    s=0.0,
    kinetic_MeV=250
)

phase_space_particle = PhaseSpaceParticle(
    x=3.5*MM,xp=7.5*MRAD,
    y=3.5*MM,yp=7.5*MRAD,
    z=0.0,delta=0.0
)

particle = ParticleFactory.create_from_phase_space_particle(
    ideal_particle=ideal_particle,
    coordinate_system=ideal_particle.get_natural_coordinate_system(),
    phase_space_particle=phase_space_particle
)

print(ideal_particle.detailed_info())
print(particle.detailed_info())

# t = ParticleRunner.run_get_trajectory(
#     p=particle,
#     m=beamline,
#     length=beamline.get_length(),
#     footstep=1*MM
# )

# Plot3.plot_p3s(t,describe='k-')
# Plot3.plot_beamline(beamline)
# Plot3.set_box(P3(y=200*MM,z=200*MM),P3(y=-200*MM,z=-200*MM))
# Plot3.show()

# 粒子跟踪，获取全部信息
# 所谓全部信息，指的是每个步长计算后的粒子信息
# 所以 all_info 是一个 RunningParticle 数组
all_info = ParticleRunner.run_get_all_info(
    p=particle,
    m=beamline,
    length=beamline.get_length(),
    footstep=1*MM
)

# 存访二维轨迹信息
track_xplane = []
track_yplane = []

# 对于每个步长位置的粒子
for p in all_info:
    # 对对应的理想粒子（有了理想粒子，才能求相空间坐标）
    local_ideal_particle = ParticleFactory.create_proton_along(
        trajectory=beamline,
        s = p.distance,
        kinetic_MeV=250
    )

    # 利用理想粒子，转为对应的相空间坐标
    local_phase_space_particle = PhaseSpaceParticle.create_from_running_particle(
        ideal_particle=local_ideal_particle,
        coordinate_system=local_ideal_particle.get_natural_coordinate_system(),
        running_particle=p
    )

    # 记录二维轨迹信息，distance-x/y，其中 distance 即粒子当前运动距离
    track_xplane.append(P2(p.distance,local_phase_space_particle.x/MM))
    track_yplane.append(P2(p.distance,local_phase_space_particle.y/MRAD))

Plot2.plot_p2s(track_xplane)
Plot2.info(
    x_label='s/m',y_label='x/mm',
    title='xplane',font_size=32
)
Plot2.show()
