"""
CCT 建模优化代码
多粒子跟踪和相椭圆绘制

作者：赵润晓
日期：2021年5月9日
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys

sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

if __name__=="__main__":
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

    pps = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_xxp_plane(
        xMax=3.5*MM,xpMax=7.5*MRAD,delta=0.0,number=16
    )

    rps = ParticleFactory.create_from_phase_space_particles(
        ideal_particle=ideal_particle,
        coordinate_system=ideal_particle.get_natural_coordinate_system(),
        phase_space_particles=pps
    )
    
    BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main()
    ParticleRunner.run_only(
        p=rps,m=beamline,length=beamline.get_length(),
        footstep=1*MM,concurrency_level=16
    )

    ideal_particle_end = ParticleFactory.create_proton_along(
        trajectory=beamline,
        s=beamline.get_length(),
        kinetic_MeV=250
    )

    pps_end = PhaseSpaceParticle.create_from_running_particles(
        ideal_particle=ideal_particle_end,
        coordinate_system=ideal_particle_end.get_natural_coordinate_system(),
        running_particles=rps
    )

    xxplane= PhaseSpaceParticle.phase_space_particles_project_to_xxp_plane(
        phase_space_particles=pps_end,convert_to_mm=True
    )

    Plot2.plot_p2s(xxplane,describe='r-',circle=True)
    Plot2.info(
        x_label='x/mm',y_label="xp/mr",
        title="xxplane",font_size=32
    )
    Plot2.equal()
    Plot2.show()



# # 外部代码
# x = 1
# if __name__=="__main__":
#     # 内部代码
#     y = 2