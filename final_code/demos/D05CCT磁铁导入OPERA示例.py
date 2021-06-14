"""
CCT 建模优化代码
CCT 建模并完成 opera 磁场计算

作者：赵润晓
日期：2021年4月27日
"""



from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *


beamline = (
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

# Plot3.plot_beamline(beamline, describes=[
#                     'r-', 'r-', 'r-']+['b-', 'b-', 'g-', 'g-', 'b-', 'b-'])
# Plot3.show()

magnets = beamline.get_magnets()
di_ccts = magnets[0:2]
quad_ccts = magnets[2:]


# 用于装 bricks 的数组，opera 8 点导体
bricks_list = [] 

# 将 diccts 转为 bricks
for cct in di_ccts:
    brick8s = Brick8s.create_by_cct(
        cct=cct,
        channel_width=3.2*MM, # 槽宽度
        channel_depth=11*MM, # 槽深度
        label="dicct", # 导体标签
        disperse_number_per_winding=10  # 每匝分 10 段
    )
    bricks_list.append(brick8s)

# 将 quad_ccts 转为 bricks
for cct in quad_ccts:
    brick8s = Brick8s.create_by_cct(
        cct=cct,
        channel_width=3.2*MM,
        channel_depth=11*MM,
        label="quadcct",
        disperse_number_per_winding=10
    )
    bricks_list.append(brick8s)


OperaConductorScript.to_opera_cond_file(bricks_list,"cct.cond")