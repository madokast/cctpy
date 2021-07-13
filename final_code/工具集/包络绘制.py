"""
包络绘制

作者：赵润晓
日期：2021年7月13日
"""
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)

from cctpy import *

# 动量分散
deltas = [-0.05, 0, 0.05]
# 绘图颜色
colors = ['r-', 'b-', 'g-']
# 粒子数目
particle_numer = 2
# x 方向包络还是 y 方向
x_plane = True

if __name__ == "__main__":
    # beamline
    bl = HUST_SC_GANTRY().create_second_bending_part_beamline()

    ip_start = ParticleFactory.create_proton_along(bl, 0.0, 215)

    # 每个动量分散
    for i in range(len(deltas)):
        # 生产粒子（相空间）
        ps = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_plane(
            plane_id=PhaseSpaceParticle.XXP_PLANE if x_plane else PhaseSpaceParticle.YYP_PLANE,
            xMax=3.5*MM,
            xpMax=7.5*MRAD,
            delta=deltas[i],
            number=particle_numer
        )
        # 转为三维空间
        rps = ParticleFactory.create_from_phase_space_particles(
            ip_start, ip_start.get_natural_coordinate_system(), ps)
        # 每个粒子
        for rp in rps:
            print(rp)
            all_info = ParticleRunner.run_get_all_info(
                rp, bl, bl.get_length(), footstep=10*MM)
            track: List[P2] = []
            # 单个粒子的轨迹
            for p in all_info:
                run_ip = ParticleFactory.create_proton_along(
                    bl.trajectory, p.distance, 215)
                run_pp = PhaseSpaceParticle.create_from_running_particle(
                    run_ip, run_ip.get_natural_coordinate_system(), p)
                if x_plane:
                    track.append(P2(p.distance, run_pp.x/MM))
                else:
                    track.append(P2(p.distance, run_pp.y/MM))
                Plot2.plot_p2s(track, describe=colors[i])


    Plot2.ylim(-0.06/MM, +0.10/MM)
    Plot2.info("s/m", "x/mm" if x_plane else "y/mm", "", font_size=30)
    Plot2.legend([str(dp) for dp in deltas])
    Plot2.show()