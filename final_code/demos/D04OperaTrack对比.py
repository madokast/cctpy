"""
CCT 建模优化代码
三维曲线段

作者：赵润晓
日期：2021年4月27日
"""


from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *
from opera_utils import *

# 定义 CCT
cct1 = CCT(
    local_coordinate_system=LocalCoordinateSystem.global_coordinate_system(),
    big_r=0.95, small_r=104.5*MM,
    bending_angle=22.5, tilt_angles=[-30],
    winding_number=42, current=-6192,
    starting_point_in_ksi_phi_coordinate=P2.zeros(),
    end_point_in_ksi_phi_coordinate=P2(
        -2 * math.pi * 42,
        BaseUtils.angle_to_radian(22.5)
    ),
    disperse_number_per_winding=120
)

cct2 = CCT(
    local_coordinate_system=LocalCoordinateSystem.global_coordinate_system(),
    big_r=0.95, small_r=120.5*MM,
    bending_angle=22.5, tilt_angles=[30],
    winding_number=42, current=-6192,
    starting_point_in_ksi_phi_coordinate=P2.zeros(),
    end_point_in_ksi_phi_coordinate=P2(
        2 * math.pi * 42,
        BaseUtils.angle_to_radian(22.5)),
    disperse_number_per_winding=120
)

magnet = CombinedMagnet(cct1, cct2)

if True: # cctpy 粒子跟踪
    

    # 粒子跟踪
    p = ParticleFactory.create_proton(
        position=P3(x=0.95,y=-1),
        direct=P3.y_direct(),
        kinetic_MeV=215
    )

    track = ParticleRunner.run_get_trajectory(
        p=p, 
        m=magnet, 
        length=0.95*BaseUtils.angle_to_radian(22.5)+2
    )


    # 绘图测试
    # Plot3.plot_cct(cct1)
    # Plot3.plot_cct(cct2)
    Plot3.plot_p3s(track,'r-')
    # Plot3.show()

    # Plot2.plot_p3s(track,describe='r-')


if False: # opera 粒子跟踪
    # 生成 cond 文件
    cct1_bricks = Brick8s.create_by_cct(
        cct=cct1,
        channel_depth=11*MM,
        channel_width=3.2*MM,
        disperse_number_per_winding=120,
        label="cct1"
    )

    cct2_bricks = Brick8s.create_by_cct(
        cct=cct2,
        channel_depth=11*MM,
        channel_width=3.2*MM,
        disperse_number_per_winding=120,
        label="cct2"
    )

    bricks = [cct1_bricks,cct2_bricks]

    operafile = open("cct_verify.cond", "w")
    operafile.write(OperaConductorScript.to_opera_cond_script(bricks))
    operafile.close()

if True: # opear 粒子跟踪结果
    opera_track = numpy.loadtxt(fname="./data/track_verify_opera.txt")
    x = opera_track[:,0]
    y = opera_track[:,1]
    z = opera_track[:,2]
    Plot3.plot_xyz_array(x,y,z,describe='k--')
    Plot3.show()

    # Plot2.plot_xy_array(x,y,describe='k--')
    # Plot2.info("x/m",'y/m','')
    # Plot2.show()


