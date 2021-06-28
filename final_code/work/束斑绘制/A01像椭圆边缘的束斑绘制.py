"""
通过相椭圆边缘绘制束斑
2021年6月27日
"""

from os import error, path
import sys
from typing import Set
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
sys.path.append(path.dirname(path.dirname(
    path.abspath(path.dirname(__file__)))))

from cctpy import *

if __name__ == '__main__':

    particle_number = 4000

    agcct3_winding_number = 25
    agcct4_winding_number = 40
    agcct5_winding_number = 34

    gantry = HUST_SC_GANTRY(
        qs3_gradient=5.546,
        qs3_second_gradient=-57.646,
        dicct345_tilt_angles=[30, 87.426, 92.151, 91.668],
        agcct345_tilt_angles=[94.503, 30, 72.425,	82.442],
        dicct345_current=9445.242,
        agcct345_current=-5642.488,
        agcct3_winding_number=agcct3_winding_number,
        agcct4_winding_number=agcct4_winding_number,
        agcct5_winding_number=agcct5_winding_number,
        agcct3_bending_angle=-67.5*(agcct3_winding_number)/(
            agcct3_winding_number+agcct4_winding_number+agcct5_winding_number),
        agcct4_bending_angle=-67.5*(agcct4_winding_number)/(
            agcct3_winding_number+agcct4_winding_number+agcct5_winding_number),
        agcct5_bending_angle=-67.5*(agcct5_winding_number)/(
            agcct3_winding_number+agcct4_winding_number+agcct5_winding_number),

        DL1=0.9007765,
        GAP1=0.4301517,
        GAP2=0.370816,
        qs1_length=0.2340128,
        qs1_aperture_radius=60 * MM,
        qs1_gradient=0.0,
        qs1_second_gradient=0.0,
        qs2_length=0.200139,
        qs2_aperture_radius=60 * MM,
        qs2_gradient=0.0,
        qs2_second_gradient=0.0,

        DL2=2.35011,
        GAP3=0.43188,
        qs3_length=0.24379,

        agcct345_inner_small_r=83 * MM,
        agcct345_outer_small_r=98 * MM,  # 83+15
        dicct345_inner_small_r=114 * MM,  # 83+30+1
        dicct345_outer_small_r=130 * MM,  # 83+45 +2
    )

    bl = gantry.create_second_bending_part_beamline()

    ip_start = ParticleFactory.create_proton_along(
        bl.get_trajectory(), 0, 215
    )

    ip_end = ParticleFactory.create_proton_along(
        bl.get_trajectory(), bl.get_trajectory().get_length(), 215
    )

    ps_x = ParticleFactory.distributed_particles(
        x = 3.5*MM/2, xp = 7.5*MRAD/2, y = 0, yp = 0, delta = 0,
        number = particle_number, distribution_area = ParticleFactory.DISTRIBUTION_AREA_FULL, 
        x_distributed=True, xp_distributed=True,
        distribution_type = ParticleFactory.DISTRIBUTION_TYPE_UNIFORM
    )

    ps_y = ParticleFactory.distributed_particles(
        x = 0, xp = 0, y = 3.5*MM/2, yp = 7.5*MRAD/2, delta = 0,
        number = particle_number, distribution_area = ParticleFactory.DISTRIBUTION_AREA_FULL, 
        y_distributed=True, yp_distributed=True,
        distribution_type = ParticleFactory.DISTRIBUTION_TYPE_UNIFORM
    )


    ps:List[PhaseSpaceParticle] = []
    for i in range(particle_number):
        ps.append(PhaseSpaceParticle(
            x = ps_x[i].x,
            xp = ps_x[i].xp,
            y = ps_y[i].y,
            yp = ps_y[i].yp,
            z = 0.0,
            delta = 0.0
        ))

    to_be_removed:Set[PhaseSpaceParticle] = set()

    for i in range(len(ps)):
        pi = ps[i]
        for j in range(i,len(ps)):
            pj = ps[j]
            if pi.dominate(pj):
                to_be_removed.add(pj)

    for p in to_be_removed:
        ps.remove(p)

    print(f"去除非支配粒子后，len(ps) = {len(ps)}")



    BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main()

    for delta in BaseUtils.linspace(-0.07, 0.1, 18):

        for p in ps:
            p.delta = delta
        
        rps = ParticleFactory.create_from_phase_space_particles(
            ideal_particle = ip_start, 
            coordinate_system= ip_start.get_natural_coordinate_system(),
            phase_space_particles = ps
        )

        ParticleRunner.run_only(rps,bl,bl.get_length(),concurrency_level=16)

        ps_end = PhaseSpaceParticle.create_from_running_particles(
            ideal_particle=ip_end,
            coordinate_system=ip_end.get_natural_coordinate_system(),
            running_particles=rps
        )

        xy = [P2(pp.x, pp.y)/MM for pp in ps_end]

        Plot2.plot_p2s(xy,describe='ro')
        Plot2.show()

        

        






