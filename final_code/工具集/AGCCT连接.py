"""
CCT 建模优化代码
工具集

作者：赵润晓
日期：2021年6月7日
"""

import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)

from cctpy import *

if __name__ == "__main__":
    R = 0.95
    bl = (
        Beamline.set_start_point(start_point=P2(
            R, BaseUtils.angle_to_radian(-20)*R))
        .first_drift(P2.y_direct(), BaseUtils.angle_to_radian(20)*R)
        .append_agcct(
            big_r=R,
            small_rs=[128*MM + 9.5*MM, 113*MM + 9.5 *
                      MM, 98*MM + 9.5*MM, 83*MM + 9.5*MM],
            bending_angles=[17.05, 27.27, 23.18],  # [15.14, 29.02, 23.34]
            tilt_angles=[[30, 87.076, 91.829, 85.857],
                         [101.317, 30, 75.725, 92.044]],
            winding_numbers=[[128], [25, 40, 34]],
            currents=[9536.310, -6259.974],
            disperse_number_per_winding=36
        ).append_drift(BaseUtils.angle_to_radian(20)*R)
    )

    ms = bl.magnets
    agcct3 = ms[5]
    agcct4 = ms[7]

    print(agcct3)
    print(agcct4)

    ac = AGCCT_CONNECTOR(agcct3, agcct4)
    # print(ac.current)
    # print(ac.length)
    if True:  # PLOT
        Plot3.plot_ndarry3ds(agcct3.dispersed_path3)
        Plot3.plot_ndarry3ds(agcct4.dispersed_path3)
        Plot3.plot_ndarry3ds(ac.dispersed_path3, describe='b-')

        Plot3.plot_p3(P3.from_numpy_ndarry(
            agcct3.dispersed_path3[-1]), describe='k.')
        Plot3.plot_p3(P3.from_numpy_ndarry(
            agcct3.dispersed_path3[-2]), describe='k.')
        Plot3.plot_p3(P3.from_numpy_ndarry(
            agcct3.dispersed_path3[-3]), describe='k.')

        Plot3.plot_p3(P3.from_numpy_ndarry(
            agcct4.dispersed_path3[0]), describe='y.')
        Plot3.plot_p3(P3.from_numpy_ndarry(
            agcct4.dispersed_path3[1]), describe='y.')
        Plot3.plot_p3(P3.from_numpy_ndarry(
            agcct4.dispersed_path3[2]), describe='y.')

        Plot3.plot_p3(P3.from_numpy_ndarry(
            ac.dispersed_path3[0]), describe='g.')
        Plot3.plot_p3(P3.from_numpy_ndarry(
            ac.dispersed_path3[1]), describe='g.')
        Plot3.plot_p3(P3.from_numpy_ndarry(
            ac.dispersed_path3[2]), describe='g.')

        Plot3.set_center(P3.origin(), cube_size=1.2)
        Plot3.show()
    # print(ac.dispersed_path3)

    if False: # 磁场测试
        gap = Magnets(agcct3, agcct4)
        linked = Magnets(agcct3, ac, agcct4)

        # bz_gap = gap.magnetic_field_bz_along(line2=bl.trajectory)
        # bz_linked = linked.magnetic_field_bz_along(line2=bl.trajectory)

        g_gap = gap.graident_field_along(line2=bl.trajectory)
        g_linked = linked.graident_field_along(line2=bl.trajectory)

        Plot2.plot(g_gap)
        Plot2.plot(g_linked,describe='b-')
        Plot2.plot([P2(g_gap[i].x,g_gap[i].y-g_linked[i].y) for i in range(len(g_gap))])

        Plot2.show()
    

