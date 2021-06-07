"""
CCT 建模优化代码
D03MagnetSlicer 示例

作者：赵润晓
日期：2021年6月3日
"""

# 设计轨道切片之两个四级铁

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *
from opera_utils import *
from cosy_utils import *

bl = (
    Beamline.set_start_point(P2.origin())
    .first_drift(P2.x_direct(),length=1.0*M)
    .append_qs(
        length=0.3,
        gradient=-1.0/(20*MM),
        second_gradient=0.0,
        aperture_radius=20*MM
    )
    .append_drift(0.5)
    .append_qs(
        length=0.3,
        gradient=1.0/(20*MM),
        second_gradient=0.0,
        aperture_radius=20*MM
    )
    .append_drift(1.0)
)

if False: # 粒子跟踪
    ip = ParticleFactory.create_proton_along(bl,kinetic_MeV=250)

    pp = PhaseSpaceParticle(x = 2*MM)

    rp = ParticleFactory.create_from_phase_space_particle(
        ideal_particle=ip,
        coordinate_system=ip.get_natural_coordinate_system(),
        phase_space_particle = pp
    )

    ai = ParticleRunner.run_get_all_info(rp,bl,bl.get_length())

    xplane = []
    yplane = []

    for cp in ai:
        d = cp.distance
        cip = ParticleFactory.create_proton_along(bl,kinetic_MeV=250,s=d)
        cpp = PhaseSpaceParticle.create_from_running_particle(
            ideal_particle=cip,
            coordinate_system=cip.get_natural_coordinate_system(),
            running_particle=cp
        )
        xplane.append(P2(d,cpp.x))
        yplane.append(P2(d,cpp.y))

    Plot2.plot_p2s(xplane)
    Plot2.plot_beamline_straight(bl)
    Plot2.show()


if True: # 切片
    slices =  MagnetSlicer.slice_trajectory(
        magnet=bl,
        trajectory=bl,
        Bp = Protons.get_magnetic_stiffness(250),
        aperture=20*MM,
        good_field_area_width=15*MM
    )

    for s in slices:
        print(s)

"""
M5 1.0 -0.0009990009990009994 0.0 0 0 0 0.02 ;
M5 0.3 -0.9966777408637876 0.0 0 0 0 0.02 ;
M5 0.5 0.001996007984031937 0.0 0 0 0 0.02 ;
M5 0.3 0.9966777408637876 0.0 0 0 0 0.02 ;
M5 0.999 0.0 0.0 0 0 0 0.02 ;
M5 0.001 0.0 0.0 0 0 0 0.02 ;
"""


"""
COSY SCRIPT

INCLUDE 'COSY' ;
PROCEDURE RUN ;

VARIABLE X0 1 ; VARIABLE XP0 1 ; VARIABLE Y0 1 ; VARIABLE YP0 1 ; 

X0 := 3.5E-3 ;
XP0 := 7.5E-3 ;
Y0 := 3.5E-3 ;
YP0 := 7.5E-3 ;

OV 5 3 0 ; {order 1, phase space MSm 3, # of parameters 0}
RPP 250 ; {particle type = proton, kinetic energy = 250MeV}
SB X0 XP0 0    Y0 YP0 0    0 0 0 0 0;
CR ;

SR  2.00000e-03  0.000000e+00  0.000000e+00  0.000000e+00  0  0.0  0  0  1 ;

UM ;  BP ; 

DL 1.0 ;
M5 0.3 -1 0.0 0 0 0 2e-2;
DL 0.5 ;
M5 0.3 1 0.0 0 0 0 2e-2;
DL 1.0 ;

EP ; PG -1 -2 ; 

ENDPROCEDURE ;
RUN ; END ;
"""