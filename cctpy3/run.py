from cctpy import *
from ccpty_cuda import *
import time
import numpy as np

momentum_dispersions = [-0.05, 0.0, 0.05]
particle_number_per_plane_per_dp = 6

# 优化匹配的变量数目
VARIABLE_NUMBER: int = 10


def run(params:np.ndarray):
    start_time = time.time()

    gantry_number = params.shape[0]

    beamlines :List[Beamline]= []

    for i in range(gantry_number):
        param = params[i]
        



def read_param() -> np.ndarray:
    """
    读取输入，格式如下：
    2 输入数目
    1 编号，从 1 开始
    -9.208 以下是 10 个参数
    -53.455
    80.
    88.
    92.
    107.1
    83.355
    77.87
    -9507.95
    -5608.6
    2 第二组
    -9.208
    53.455
    80.
    88.
    92.
    107.1
    83.355
    77.87
    -9507.95
    -5708.6



    转为如下格式：
    [[-9.20800e+00 -5.34550e+01  8.00000e+01  8.80000e+01  9.20000e+01
       1.07100e+02  8.33550e+01  7.78700e+01 -9.50795e+03 -5.60860e+03]
     [-9.20800e+00  5.34550e+01  8.00000e+01  8.80000e+01  9.20000e+01
       1.07100e+02  8.33550e+01  7.78700e+01 -9.50795e+03 -5.70860e+03]]

    用于 cct345_data_generator.py 处理
    -------

    """
    input_file = np.loadtxt('input.txt', dtype=np.float64)
    gantry_number = int(input_file[0])
    data = np.empty((gantry_number, VARIABLE_NUMBER), dtype=np.float64)
    for i in range(gantry_number):
        if int(input_file[i * (VARIABLE_NUMBER + 1) + 1]) != i + 1:
            raise ValueError("输入文件不合法")

        data[i, :] = input_file[i * (VARIABLE_NUMBER + 1) + 2:(i + 1) * (VARIABLE_NUMBER + 1) + 1]

    return data
