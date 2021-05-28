"""
CCT 建模优化代码
A19 直线二极磁铁 LocalUniformMagnet


作者：赵润晓
日期：2021年5月2日
"""

from os import error, path
import sys

sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))

from cctpy import *

lcs = LocalCoordinateSystem(
    location=P3.x_direct(1),
    x_direction=P3.y_direct(),
    z_direction=P3.x_direct()
)

lum = LocalUniformMagnet(
    local_coordinate_system=lcs,
    length=0.5,
    aperture_radius=0.05,
    magnetic_field=2
)

tr = Trajectory.set_start_point().first_line(length=2)

p = ParticleFactory.create_proton_along(
    trajectory=tr,
    kinetic_MeV=215
)

t = ParticleRunner.run_get_trajectory(p,lum,length=tr.get_length())

Plot2.plot([p.to_p2() for p in t])
Plot2.plot(lum)
Plot2.plot(tr)
Plot2.show()

