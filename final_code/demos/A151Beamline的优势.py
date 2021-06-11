"""
CCT 建模优化代码
A15 束线 beamline 示例


作者：赵润晓
日期：2021年6月10日
"""

from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

# --------------------  采用直接构建磁铁的方法创建三Q铁束线 --------------------------

# 要构建的束线分布图
t = (
    Trajectory.set_start_point(start_point=P2.origin())
    .first_line(direct=P2.x_direct(), length=1.0)
    .add_strait_line(0.3).as_aperture_objrct_on_last(40*MM)
    .add_strait_line(0.5)
    .add_strait_line(0.3).as_aperture_objrct_on_last(40*MM)
    .add_strait_line(0.5)
    .add_strait_line(0.3).as_aperture_objrct_on_last(40*MM)
    .add_strait_line(1.0)
)
# Plot2.plot_trajectory(t)
# Plot2.ylim(-0.08,0.08)
# Plot2.info("s/m","x/m")
# Plot2.show()


q1 = Q(
    local_coordinate_system=LocalCoordinateSystem(location=P3(0, 0, 1)),
    length=0.3,
    gradient=0.3/(40*MM),
    aperture_radius=40*MM
)

q2 = Q(
    local_coordinate_system=LocalCoordinateSystem(location=P3(0, 0, 1.8)),
    length=0.3,
    gradient=-0.2/(40*MM),
    aperture_radius=40*MM
)

q3 = Q(
    local_coordinate_system=LocalCoordinateSystem(location=P3(0, 0, 2.6)),
    length=0.3,
    gradient=0.2/(40*MM),
    aperture_radius=40*MM
)

# Plot3.plot_q(q1)
# Plot3.plot_q(q2)
# Plot3.plot_q(q3)
# Plot3.show()

# 进行跟踪的粒子的相空间坐标
pp = PhaseSpaceParticle(
    x=3.5*MM, xp=7.5*MRAD
)
# 理想粒子
ip = ParticleFactory.create_proton(
    position=P3.origin(),
    direct=P3.z_direct(),
    kinetic_MeV=250
)
# 进行跟踪的粒子的相空间转为实际三维空间粒子
rp = ParticleFactory.create_from_phase_space_particle(
    ideal_particle=ip,
    coordinate_system=LocalCoordinateSystem.global_coordinate_system(),
    phase_space_particle=pp
)
# 进行粒子跟踪
track = ParticleRunner.run_get_trajectory(
    p=rp,
    m=Magnet.combine(q1, q2, q3),  # 三个磁铁合并
    length=3.9,
    footstep=10*MM
)
# 获取三个方向轨迹数组
x, y, z = P3.extract(track)
# 绘图
# Plot2.plot_xy_array(z,x)
# Plot2.info("s/m","x/m")
# Plot2.show()


# -----------------------------  采用Trajectory创建三Q铁束线 ---------------------------

# 首先创建设计轨道
trajectory = (
    Trajectory.set_start_point(start_point=P2(123, 456))
    .first_line(direct=P2(-1, 321), length=1.0)
    .add_strait_line(0.3)
    .add_strait_line(0.5)
    .add_strait_line(0.3)
    .add_strait_line(0.5)
    .add_strait_line(0.3)
    .add_strait_line(1.0)
)

# 在设计轨道上放置 q 铁
q1 = Q.create_q_along(
    trajectory=trajectory,
    s=1.0,
    length=0.3,
    gradient=0.3/(40*MM),
    aperture_radius=40*MM,
)

q2 = Q.create_q_along(
    trajectory=trajectory,
    s=1.8,
    length=0.3,
    gradient=-0.2/(40*MM),
    aperture_radius=40*MM,
)

q3 = Q.create_q_along(
    trajectory=trajectory,
    s=2.6,
    length=0.3,
    gradient=0.2/(40*MM),
    aperture_radius=40*MM,
)

# Plot3.plot_q(q1)
# Plot3.plot_q(q2)
# Plot3.plot_q(q3)
# Plot3.plot_line2(trajectory)
# Plot3.show()

# 进行跟踪的粒子的相空间坐标
pp = PhaseSpaceParticle(
    x=3.5*MM, xp=7.5*MRAD
)
# 理想粒子
ip = ParticleFactory.create_proton_along(
    trajectory=trajectory,
    s=0.0,
    kinetic_MeV=250
)
# 进行跟踪的粒子的相空间转为实际三维空间粒子
rp = ParticleFactory.create_from_phase_space_particle(
    ideal_particle=ip,
    coordinate_system=ip.get_natural_coordinate_system(),
    phase_space_particle=pp
)
# 进行粒子跟踪，获得每一步粒子的全部信息
all_info = ParticleRunner.run_get_all_info(
    p=rp,
    m=Magnet.combine(q1, q2, q3),
    length=trajectory.get_length(),
    footstep=10*MM
)
s = []
x = []
for current_p in all_info:
    distance = current_p.distance
    current_ip = ParticleFactory.create_proton_along(
        trajectory=trajectory,
        s=distance,
        kinetic_MeV=250
    )
    current_pp = PhaseSpaceParticle.create_from_running_particle(
        ideal_particle=current_ip,
        coordinate_system=current_ip.get_natural_coordinate_system(),
        running_particle=current_p
    )

    s.append(distance)
    x.append(current_pp.x)

# Plot2.plot_xy(s,x)
# Plot2.info("s/m","x/m")
# Plot2.show()


# --------------------------------  采用Beamline创建三Q铁束线  ------------------
bl = (
    Beamline.set_start_point()  # 可以传入 start_point，默认值 P2.origin()
    # first_drift() 还有参数 direct 确定第一个漂移段的方向，默认值 P2.x_direct()
    .first_drift(length=1.0)
    .append_q(
        length=0.3,
        gradient=0.3/(40*MM),
        aperture_radius=40*MM
    )
    .append_drift(0.5)
    .append_q(
        length=0.3,
        gradient=-0.2/(40*MM),
        aperture_radius=40*MM
    )
    .append_drift(0.5)
    .append_q(
        length=0.3,
        gradient=0.2/(40*MM),
        aperture_radius=40*MM
    )
    .append_drift(1.0)
)
# Plot3.plot_beamline(bl)
# Plot3.show()
# Plot2.plot_beamline(bl)
# Plot2.show()

# Plot2.plot_beamline_straight(bl)
# Plot2.show()

phase_space_particle_distance = bl.track_phase_space_particle(
    x_mm=3.5, xp_mrad=7.5,
    y_mm=0.0, yp_mrad=0.0,
    delta=0.0, kinetic_MeV=250
)
sx = ValueWithDistance.convert_to_p2(
    data=phase_space_particle_distance,
    convertor=lambda p: p.x
)

# Plot2.plot(sx)
# Plot2.show()


data = bl.track_phase_space_particle(
    x_mm=0.0, xp_mrad=0.0,
    y_mm=2.0, yp_mrad=0.0,
    delta=0.0, kinetic_MeV=250
)
sy = ValueWithDistance.convert_to_p2(
    data=data,
    convertor=lambda p: p.y
)
Plot2.plot(sy)
Plot2.plot_beamline_straight(bl)
Plot2.show()