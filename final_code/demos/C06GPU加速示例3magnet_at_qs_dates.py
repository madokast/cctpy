"""
CCT 建模优化代码
GPU 加速示例(3)

作者：赵润晓
日期：2021年5月6日
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from hust_sc_gantry import HUST_SC_GANTRY
from cctpy import *

ga32 = GPU_ACCELERATOR(float_number_type=GPU_ACCELERATOR.FLOAT32)
ga64 = GPU_ACCELERATOR(float_number_type=GPU_ACCELERATOR.FLOAT64,block_dim_x=512)


# ----- magnet_at_qs 测试
qs = QS(
    local_coordinate_system=LocalCoordinateSystem(
        location=P3(1,0,0),
        x_direction=-P3.x_direct(),
        z_direction=P3.y_direct()
    ),
    length=0.27,
    gradient=5,
    second_gradient=20,
    aperture_radius=100*MM
)
point = P3(0.95,0.05,0)
# 查看入口中心位置的磁场
print("magnet_at_qs 计算 qs 磁铁，在 p 点产生的磁场")
print("CPU计算结果：",qs.magnetic_field_at(point))
print("GPU32计算结果：",ga32.magnet_at_qs(qs.to_numpy_array(numpy.float32),point))
print("GPU64计算结果：",ga64.magnet_at_qs(qs.to_numpy_array(numpy.float64),point))
# CPU计算结果： (0.0, 0.0, 0.27500000000000024)
# GPU32计算结果： (0.0, 0.0, 0.27500006556510925)
# GPU64计算结果： (0.0, 0.0, 0.27500000000000024)

# ----- magnet_at_qs_date 测试
print("magnet_at_qs_date 计算 qs 磁铁，在 p 点产生的磁场")
print("CPU计算结果：",qs.magnetic_field_at(point))
print("GPU32计算结果：",ga32.magnet_at_qs_date(qs.to_numpy_array(numpy.float32),point))
print("GPU64计算结果：",ga64.magnet_at_qs_date(qs.to_numpy_array(numpy.float64),point))
# magnet_at_qs_date 计算 qs 磁铁，在 p 点产生的磁场
# CPU计算结果： (0.0, 0.0, 0.27500000000000024)
# GPU32计算结果： (0.0, 0.0, 0.27500006556510925)
# GPU64计算结果： (0.0, 0.0, 0.27500000000000024)



# ----------- magnet_at_qs_dates 测试 1 --------
qss = [qs]*10
print("magnet_at_qs_dates ")
print("CPU计算结果：",Magnet.combine(qss).magnetic_field_at(point))
print("GPU32计算结果：",ga32.magnet_at_qs_dates(qss,point))
print("GPU64计算结果：",ga64.magnet_at_qs_dates(qss,point))

# ----------- magnet_at_qs_dates 测试 2 --------
qss = [qs]
print("magnet_at_qs_dates2 ")
print("CPU计算结果：",Magnet.combine(qss).magnetic_field_at(point))
print("GPU32计算结果：",ga32.magnet_at_qs_dates(qss,point))
print("GPU64计算结果：",ga64.magnet_at_qs_dates(qss,point))

# ----------- magnet_at_qs_dates 测试 3 --------
qs1 = QS(
    local_coordinate_system=LocalCoordinateSystem(
        location=P3(1,0,0),
        x_direction=-P3.x_direct(),
        z_direction=P3.y_direct()
    ),
    length=0.27,
    gradient=5,
    second_gradient=20,
    aperture_radius=100*MM
)
qs2 = QS(
    local_coordinate_system=LocalCoordinateSystem(
        location=P3(1,1,0),
        x_direction=-P3.x_direct(),
        z_direction=P3.y_direct()
    ),
    length=0.27,
    gradient=5,
    second_gradient=20,
    aperture_radius=100*MM
)
point = P3(0.95,0,0)
print("magnet_at_qs_dates3 ")
print("CPU计算结果：",Magnet.combine(qs1,qs2).magnetic_field_at(point))
print("GPU32计算结果：",ga32.magnet_at_qs_dates([qs1,qs2],point))
print("GPU64计算结果：",ga64.magnet_at_qs_dates([qs1,qs2],point))
point = P3(0.91,1.05,0)
print("magnet_at_qs_dates3 ")
print("CPU计算结果：",Magnet.combine(qs1,qs2).magnetic_field_at(point))
print("GPU32计算结果：",ga32.magnet_at_qs_dates([qs1,qs2],point))
print("GPU64计算结果：",ga64.magnet_at_qs_dates([qs1,qs2],point))