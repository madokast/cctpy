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
    location=P3(1,2,3),
    x_direction=P3.y_direct(),
    z_direction=P3.x_direct()
)

lum = LocalUniformMagnet(
    local_coordinate_system=lcs,
    length=0.5,
    aperture_radius=0.05,
    magnetic_field=50
)

print(lum.magnetic_field_at(P3(1,2,3)+P3.x_direct(0.001)))
print(lum.magnetic_field_at(P3(1,2,3)-P3.x_direct(0.001)))
print(lum.magnetic_field_at(P3(1,2,3)+P3.x_direct(0.001)+P3.y_direct(0.001)))
print(lum.magnetic_field_at(P3(1,2,3)+P3.x_direct(0.001)+P3.y_direct(0.5)))
print(lum.magnetic_field_at(P3(1,2,3)+P3.x_direct(0.001)-P3.y_direct(0.5)))