"""
CCT 建模优化代码
OPERA 磁场表格文件读取

作者：赵润晓
日期：2021年6月3日
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *
from opera_utils import *

m = OperaFieldTableMagnet(
    file_name="./data/test_opera_field_table.table",
    first_corner_x=-0.2, first_corner_y=-0.05, first_corner_z=-0.2,
    step_between_points_x=0.02, step_between_points_y=0.01, step_between_points_z=0.01,
    number_of_points_x=21, number_of_points_y=11, number_of_points_z=41,
    unit_of_length=M, unit_of_field=1
)

# print(m.table_position_data_x)

print(m.magnetic_field_at(P3(0.0123,0.0852,0.0654)))
# print(m.magnetic_field_at(P3(0,0.001,0)))

# ms = [P2(p.y,m.magnetic_field_at(p).y) for p in BaseUtils.linspace(P3(-0.1,-0.02,0),P3(0.1,0.01,0.05),1000)]
# Plot2.plot(ms)
# Plot2.show()