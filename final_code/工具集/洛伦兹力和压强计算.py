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

R = 0.95

bl = (
    Beamline.set_start_point()
    .first_drift()
    .append_agcct(
        big_r=0.95,  # 偏转半径
        small_rs=[130*MM, 114*MM, 98*MM, 83*MM],  # 从大到小的绕线半径
        # 四极交变 CCT 的偏转角度，无需给出二极 CCT 的偏转角度，因为就是这三个数的和
        bending_angles=[17.05, 27.27, 23.18],
        tilt_angles=[
            [30, 88.773, 98.139, 91.748],  # 二极 CCT 倾斜角，即 ak 的值，任意长度数组
            [101.792, 30, 62.677, 89.705]  # 四极 CCT 倾斜角，即 ak 的值，任意长度数组
        ],
        winding_numbers=[
            [128],  # 二极 CCT 匝数
            [25, 40, 34]  # 四极 CCT 匝数
        ],
        currents=[9409.261, -7107.359]  # 二极 CCT 和四极 CCT 电流
    )
)

# 提取 CCT
dicct_out = CCT.as_cct(bl.magnets[0])
dicct_in = CCT.as_cct(bl.magnets[1])
agcct3_in = CCT.as_cct(bl.magnets[2])
agcct3_out = CCT.as_cct(bl.magnets[3])
agcct4_in = CCT.as_cct(bl.magnets[4])
agcct4_out = CCT.as_cct(bl.magnets[5])
agcct5_in = CCT.as_cct(bl.magnets[6])
agcct5_out = CCT.as_cct(bl.magnets[7])

# 转为 wire
wdicct_out = Wire.create_by_cct(dicct_out)
wdicct_in = Wire.create_by_cct(dicct_in)
wagcct3_in = Wire.create_by_cct(agcct3_in)
wagcct3_out = Wire.create_by_cct(agcct3_out)
wagcct4_in = Wire.create_by_cct(agcct4_in)
wagcct4_out = Wire.create_by_cct(agcct4_out)
wagcct5_in = Wire.create_by_cct(agcct5_in)
wagcct5_out = Wire.create_by_cct(agcct5_out)

# 当前进行分析的 CCT
if True:
    if True:
        delta_angle = 10  # 当 CCT 负 ksi 方向绕线时，写负数
        s_start = 0+delta_angle/2  # 起止 ksi
        s_end = 360*128-delta_angle/2
        s_number = 36*128  # 数目
        current_cct = dicct_in  # 当前 CCT 和 wire
        固定坐标系 = False
        洛伦兹力 = False  # else 压强
        file_name = f'./二极CCT内层{"固定" if 固定坐标系 else "滑动"}坐标系-{"洛伦兹力" if 洛伦兹力 else "压强"}.txt'
    if False:
        delta_angle = -10  # 当 CCT 负 ksi 方向绕线时，写负数
        s_start = 0+delta_angle/2  # 起止 ksi
        s_end = -360*128-delta_angle/2
        s_number = 36*128  # 数目
        current_cct = dicct_out  # 当前 CCT 和 wire
        固定坐标系 = False
        洛伦兹力 = False  # else 压强
        file_name = f'./二极CCT外层{"固定" if 固定坐标系 else "滑动"}坐标系-{"洛伦兹力" if 洛伦兹力 else "压强"}.txt'
    if False:
        delta_angle = 10  # 当 CCT 负 ksi 方向绕线时，写负数
        s_start = 0+delta_angle/2  # 起止 ksi，从 0 开始，是因为原本值为 2pi 的整数倍，所以可以直接从 0 开始
        s_end = 360*25-delta_angle/2
        s_number = 36*25  # 数目
        current_cct = agcct3_in  # 当前 CCT 和 wire
        固定坐标系 = False
        洛伦兹力 = False  # else 压强
        file_name = f'./四极CCT第1段内层{"固定" if 固定坐标系 else "滑动"}坐标系-{"洛伦兹力" if 洛伦兹力 else "压强"}.txt'
    if False:
        delta_angle = -10  # 当 CCT 负 ksi 方向绕线时，写负数
        s_start = 0+delta_angle/2  # 起止 ksi
        s_end = -360*25-delta_angle/2
        s_number = 36*25  # 数目
        current_cct = agcct3_out  # 当前 CCT 和 wire
        固定坐标系 = False
        洛伦兹力 = False  # else 压强
        file_name = f'./四极CCT第1段外层{"固定" if 固定坐标系 else "滑动"}坐标系-{"洛伦兹力" if 洛伦兹力 else "压强"}.txt'
    if False:
        delta_angle = -10  # 当 CCT 负 ksi 方向绕线时，写负数
        s_start = 0+delta_angle/2 + 25*360  # 起止 ksi
        s_end = -360*40-delta_angle/2+25*360
        s_number = 36*40  # 数目
        current_cct = agcct4_in  # 当前 CCT 和 wire
        固定坐标系 = False
        洛伦兹力 = False  # else 压强
        file_name = f'./四极CCT第2段内层{"固定" if 固定坐标系 else "滑动"}坐标系-{"洛伦兹力" if 洛伦兹力 else "压强"}.txt'
    if False:
        delta_angle = 10  # 当 CCT 负 ksi 方向绕线时，写负数
        s_start = 0+delta_angle/2 - 25*360  # 起止 ksi
        s_end = 360*40-delta_angle/2-25*360
        s_number = 36*40  # 数目
        current_cct = agcct4_out  # 当前 CCT 和 wire
        固定坐标系 = False
        洛伦兹力 = False  # else 压强
        file_name = f'./四极CCT第2段外层{"固定" if 固定坐标系 else "滑动"}坐标系-{"洛伦兹力" if 洛伦兹力 else "压强"}.txt'
    if False:
        delta_angle = 10  # 当 CCT 负 ksi 方向绕线时，写负数
        s_start = 0+delta_angle/2 + 25*360 - 40*360  # 起止 ksi
        s_end = 360*34-delta_angle/2+25*360 - 40*360
        s_number = 36*34  # 数目
        current_cct = agcct5_in  # 当前 CCT 和 wire
        固定坐标系 = False
        洛伦兹力 = False  # else 压强
        file_name = f'./四极CCT第3段内层{"固定" if 固定坐标系 else "滑动"}坐标系-{"洛伦兹力" if 洛伦兹力 else "压强"}.txt'
    if False:
        delta_angle = -10  # 当 CCT 负 ksi 方向绕线时，写负数
        s_start = 0+delta_angle/2 - 25*360 + 40*360  # 起止 ksi
        s_end = -360*34-delta_angle/2-25*360 + 40*360
        s_number = 36*34  # 数目
        current_cct = agcct5_out  # 当前 CCT 和 wire
        固定坐标系 = False
        洛伦兹力 = True  # else 压强
        file_name = f'./四极CCT第3段外层{"固定" if 固定坐标系 else "滑动"}坐标系-{"洛伦兹力" if 洛伦兹力 else "压强"}.txt'

current_wire = Wire.create_by_cct(current_cct)
other_magnet = CombinedMagnet(*bl.magnets)
other_magnet.remove(current_cct)


def task(s):
    if 固定坐标系:
        lcp = LocalCoordinateSystem.global_coordinate_system()
    else:
        lcp = LocalCoordinateSystem(
            location=current_wire.function_line3.point_at_p3_function(BaseUtils.angle_to_radian(s)),
            x_direction=current_wire.function_line3.direct_at_p3_function(
                BaseUtils.angle_to_radian(s)),
            z_direction=current_cct.bipolar_toroidal_coordinate_system.main_normal_direction_at(
                current_cct.p2_function(BaseUtils.angle_to_radian(s))
            )
        )

    if 洛伦兹力:
        fon = current_wire.lorentz_force_on_wire(
            s=BaseUtils.angle_to_radian(s),
            delta_length=current_cct.small_r *
            BaseUtils.angle_to_radian(delta_angle),
            local_coordinate_point=lcp,
            other_magnet=other_magnet
        )
    else:
        fon = current_wire.pressure_on_wire_MPa(
            s=BaseUtils.angle_to_radian(s),
            delta_length=current_cct.small_r *
            BaseUtils.angle_to_radian(delta_angle),
            local_coordinate_point=lcp,
            other_magnet=other_magnet,
            channel_width=3.2*MM,
            channel_depth=11*MM
        )
    print(fon)
    return fon


if __name__ == "__main__":
    BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main()
    ss = BaseUtils.linspace(s_start, s_end, s_number)

    fons = BaseUtils.submit_process_task(
        task=task, param_list=[[s] for s in ss])

    data = []
    for i in range(len(fons)):
        p, f = fons[i]
        data.append([i+1, p.x, p.y, p.z, f.x, f.y, f.z])

    data = numpy.array(data)

    numpy.savetxt(file_name, data)

    data = numpy.loadtxt(file_name)

    if True:  # 画图
        Plot2.plot_ndarry2ds(data[:, (0, 4)], describe='r-')
        Plot2.plot_ndarry2ds(data[:, (0, 5)], describe='b-')
        Plot2.plot_ndarry2ds(data[:, (0, 6)], describe='y-')

        if 固定坐标系:
            Plot2.legend('x', 'y', 'z', font_size=18,
                         font_family="Microsoft YaHei")
        else:
            Plot2.legend('绕线方向', 'rib方向', '径向', font_size=18,
                         font_family="Microsoft YaHei")

        if 洛伦兹力:
            Plot2.info('index', 'lorentz_force/N', file_name,
                       font_size=18, font_family="Microsoft YaHei")
        else:
            Plot2.info('index', 'pressure/MPa', '',
                       font_size=18, font_family="Microsoft YaHei")
        Plot2.show()
