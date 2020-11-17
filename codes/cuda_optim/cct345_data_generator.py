from typing import List

import numpy as np

from cctpy.constant import MM
from cctpy.baseutils import Vectors

NUMBER_OF_VARIABLES_PER_CCT: int = 11
PI: float = np.pi
CUDA_THREAD_SIZE_X: float = 1024


# 2020年11月16日 验证无误
def __cct_data_generate(param: np.ndarray) -> np.ndarray:
    """
    生成机架参数 CUDA_THREAD_SIZE_X * NUMBER_OF_VARIABLES_PER_CCT 长度的 float32 数组

    param 表示第 2 到第 9 个参数，各个参数意义如下

    0 qs3梯度 Tm-1
    1 qs3六极梯度 Tm-2
    2 二极CCT 四极倾角 deg
    3 二极CCT 六极倾角 deg
    4 二极CCT 八极倾角 deg
    5 四极CCT 二极倾角 deg
    6 四极CCT 六极倾角 deg
    7 四极CCT 八极倾角 deg
    8 二极CCT电流 A
    9 四极CCT电流 A
    """
    data = np.zeros((CUDA_THREAD_SIZE_X * NUMBER_OF_VARIABLES_PER_CCT,), dtype=np.float32)
    i: int = 0
    k: List[float] = [0.0] * 4

    # init
    bigR: float = 0.95
    dicct_innerSmallR: float = 83 * MM + 15 * MM * 2
    dicct_outerSmall: float = 83 * MM + 15 * MM * 3
    dicct_bendingAngle_deg: float = 67.5
    dicct_bendingRadian: float = dicct_bendingAngle_deg / 180.0 * PI
    dicct_tiltAngles: List[float] = [30.0, 80.0, 90.0, 90.0]
    dicct_windingNumber: int = 128
    dicct_current: float = -9664.0
    dicct_phi0: float = dicct_bendingRadian / dicct_windingNumber

    agcct_innerSmallR: float = 83 * MM + 15 * MM * 0
    agcct_outerSmall: float = 83 * MM + 15 * MM * 1
    agcct_bendingAngle_degs = [11.716404, 27.93897, 27.844626]
    agcct_bendingRadians = [agcct_bendingAngle_degs[0] / 180.0 * PI, agcct_bendingAngle_degs[1] / 180.0 * PI,
                            agcct_bendingAngle_degs[2] / 180.0 * PI]
    agcct_tiltAngles: List[float] = [90.0, 30.0, 90.0, 90.0]
    agcct_windingNumbers: List[int] = [21, 50, 50]
    agcct_current: float = -6000.0
    agcct_phi0s: List[float] = [
        agcct_bendingRadians[0] / agcct_windingNumbers[0],
        agcct_bendingRadians[1] / agcct_windingNumbers[1],
        agcct_bendingRadians[2] / agcct_windingNumbers[2]
    ]

    # read params
    dicct_tiltAngles[1] = param[2]
    dicct_tiltAngles[2] = param[3]
    dicct_tiltAngles[3] = param[4]

    agcct_tiltAngles[0] = param[5]
    agcct_tiltAngles[2] = param[6]
    agcct_tiltAngles[3] = param[7]

    dicct_current = param[8]
    agcct_current = param[9]

    # fill data
    if True:  # bi cct inner
        a = np.sqrt(bigR * bigR - dicct_innerSmallR * dicct_innerSmallR)
        eta0 = 0.5 * np.log((bigR + a) / (bigR - a))
        ch_eta0 = np.cosh(eta0)
        sh_eta0 = np.sinh(eta0)

        k[0] = (1.0 / np.tan(dicct_tiltAngles[0] / 180.0 * PI)) / ((0 + 1) * sh_eta0)
        k[1] = (1.0 / np.tan(dicct_tiltAngles[1] / 180.0 * PI)) / ((1 + 1) * sh_eta0)
        k[2] = (1.0 / np.tan(dicct_tiltAngles[2] / 180.0 * PI)) / ((2 + 1) * sh_eta0)
        k[3] = (1.0 / np.tan(dicct_tiltAngles[3] / 180.0 * PI)) / ((3 + 1) * sh_eta0)

        ksi_deg0: int = 0
        while i < dicct_windingNumber:
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 0] = ksi_deg0 + 360 * i
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 1] = dicct_phi0

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 2] = k[0]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 3] = k[1]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 4] = k[2]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 5] = k[3]

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 6] = a

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 7] = ch_eta0
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 8] = sh_eta0

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 9] = dicct_current

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 10] = 0.0

            i += 1

    if True:  # bi cct outer
        dicct_current *= -1

        a = np.sqrt(bigR * bigR - dicct_outerSmall * dicct_outerSmall)
        eta0 = 0.5 * np.log((bigR + a) / (bigR - a))
        ch_eta0 = np.cosh(eta0)
        sh_eta0 = np.sinh(eta0)

        k[0] = -(1.0 / np.tan(dicct_tiltAngles[0] / 180.0 * PI)) / ((0 + 1) * sh_eta0)
        k[1] = -(1.0 / np.tan(dicct_tiltAngles[1] / 180.0 * PI)) / ((1 + 1) * sh_eta0)
        k[2] = -(1.0 / np.tan(dicct_tiltAngles[2] / 180.0 * PI)) / ((2 + 1) * sh_eta0)
        k[3] = -(1.0 / np.tan(dicct_tiltAngles[3] / 180.0 * PI)) / ((3 + 1) * sh_eta0)

        dicct_phi0 *= -1

        ksi_deg0 = -360 * dicct_windingNumber
        while i < dicct_windingNumber * 2:
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 0] = ksi_deg0 + 360 * (i - dicct_windingNumber)
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 1] = dicct_phi0

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 2] = k[0]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 3] = k[1]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 4] = k[2]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 5] = k[3]

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 6] = a

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 7] = ch_eta0
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 8] = sh_eta0

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 9] = dicct_current

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 10] = 0.0

            i += 1

    if True:  # ag cct1 inner
        a = np.sqrt(bigR * bigR - agcct_innerSmallR * agcct_innerSmallR)
        eta0 = 0.5 * np.log((bigR + a) / (bigR - a))
        ch_eta0 = np.cosh(eta0)
        sh_eta0 = np.sinh(eta0)

        k[0] = (1.0 / np.tan(agcct_tiltAngles[0] / 180.0 * PI)) / ((0 + 1) * sh_eta0)
        k[1] = (1.0 / np.tan(agcct_tiltAngles[1] / 180.0 * PI)) / ((1 + 1) * sh_eta0)
        k[2] = (1.0 / np.tan(agcct_tiltAngles[2] / 180.0 * PI)) / ((2 + 1) * sh_eta0)
        k[3] = (1.0 / np.tan(agcct_tiltAngles[3] / 180.0 * PI)) / ((3 + 1) * sh_eta0)

        ksi_deg0 = 0

        while i < dicct_windingNumber * 2 + agcct_windingNumbers[0]:
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 0] = ksi_deg0 + 360 * (i - dicct_windingNumber * 2)
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 1] = agcct_phi0s[0]

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 2] = k[0]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 3] = k[1]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 4] = k[2]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 5] = k[3]

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 6] = a

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 7] = ch_eta0
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 8] = sh_eta0

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 9] = agcct_current

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 10] = 0.0
            i += 1

    if True:  # ag cct2 inner
        agcct_current *= -1
        agcct_phi0s[1] *= -1

        a = np.sqrt(bigR * bigR - agcct_innerSmallR * agcct_innerSmallR)
        eta0 = 0.5 * np.log((bigR + a) / (bigR - a))
        ch_eta0 = np.cosh(eta0)
        sh_eta0 = np.sinh(eta0)

        k[0] = (1.0 / np.tan(agcct_tiltAngles[0] / 180.0 * PI)) / ((0 + 1) * sh_eta0)
        k[1] = (1.0 / np.tan(agcct_tiltAngles[1] / 180.0 * PI)) / ((1 + 1) * sh_eta0)
        k[2] = (1.0 / np.tan(agcct_tiltAngles[2] / 180.0 * PI)) / ((2 + 1) * sh_eta0)
        k[3] = (1.0 / np.tan(agcct_tiltAngles[3] / 180.0 * PI)) / ((3 + 1) * sh_eta0)

        ksi_deg0 = -360 * agcct_windingNumbers[1]

        while i < dicct_windingNumber * 2 + agcct_windingNumbers[0] + agcct_windingNumbers[1]:
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 0] = ksi_deg0 + 360 * (
                    i - dicct_windingNumber * 2 - agcct_windingNumbers[0])
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 1] = agcct_phi0s[1]

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 2] = k[0]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 3] = k[1]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 4] = k[2]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 5] = k[3]

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 6] = a

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 7] = ch_eta0
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 8] = sh_eta0

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 9] = agcct_current

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 10] = agcct_bendingRadians[0] + agcct_phi0s[0]
            i += 1

        agcct_current *= -1
        agcct_phi0s[1] *= -1

    if True:  # ag cct3 inner
        a = np.sqrt(bigR * bigR - agcct_innerSmallR * agcct_innerSmallR)
        eta0 = 0.5 * np.log((bigR + a) / (bigR - a))
        ch_eta0 = np.cosh(eta0)
        sh_eta0 = np.sinh(eta0)

        k[0] = (1.0 / np.tan(agcct_tiltAngles[0] / 180.0 * PI)) / ((0 + 1) * sh_eta0)
        k[1] = (1.0 / np.tan(agcct_tiltAngles[1] / 180.0 * PI)) / ((1 + 1) * sh_eta0)
        k[2] = (1.0 / np.tan(agcct_tiltAngles[2] / 180.0 * PI)) / ((2 + 1) * sh_eta0)
        k[3] = (1.0 / np.tan(agcct_tiltAngles[3] / 180.0 * PI)) / ((3 + 1) * sh_eta0)

        ksi_deg0 = 0

        while i < dicct_windingNumber * 2 + agcct_windingNumbers[0] + agcct_windingNumbers[1] + agcct_windingNumbers[2]:
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 0] = ksi_deg0 + 360 * (
                    i - dicct_windingNumber * 2 - agcct_windingNumbers[0] - agcct_windingNumbers[1])
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 1] = agcct_phi0s[2]

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 2] = k[0]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 3] = k[1]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 4] = k[2]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 5] = k[3]

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 6] = a

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 7] = ch_eta0
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 8] = sh_eta0

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 9] = agcct_current

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 10] = agcct_bendingRadians[0] + agcct_phi0s[0] + \
                                                         agcct_bendingRadians[1] + agcct_phi0s[1]
            i += 1

    if True:  # ag cct1 outer
        a = np.sqrt(bigR * bigR - agcct_outerSmall * agcct_outerSmall)
        eta0 = 0.5 * np.log((bigR + a) / (bigR - a))
        ch_eta0 = np.cosh(eta0)
        sh_eta0 = np.sinh(eta0)

        k[0] = -(1.0 / np.tan(agcct_tiltAngles[0] / 180.0 * PI)) / ((0 + 1) * sh_eta0)
        k[1] = -(1.0 / np.tan(agcct_tiltAngles[1] / 180.0 * PI)) / ((1 + 1) * sh_eta0)
        k[2] = -(1.0 / np.tan(agcct_tiltAngles[2] / 180.0 * PI)) / ((2 + 1) * sh_eta0)
        k[3] = -(1.0 / np.tan(agcct_tiltAngles[3] / 180.0 * PI)) / ((3 + 1) * sh_eta0)

        agcct_phi0s[0] *= -1
        agcct_current *= -1

        ksi_deg0 = -360 * agcct_windingNumbers[0]

        while i < dicct_windingNumber * 2 + agcct_windingNumbers[0] * 2 + agcct_windingNumbers[1] + \
                agcct_windingNumbers[2]:
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 0] = ksi_deg0 + 360 * (
                    i - dicct_windingNumber * 2 - agcct_windingNumbers[0] - agcct_windingNumbers[1] -
                    agcct_windingNumbers[2])
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 1] = agcct_phi0s[0]

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 2] = k[0]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 3] = k[1]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 4] = k[2]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 5] = k[3]

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 6] = a

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 7] = ch_eta0
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 8] = sh_eta0

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 9] = agcct_current

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 10] = 0.0
            i += 1

        agcct_phi0s[0] *= -1
        agcct_current *= -1

    if True:  # ag cct2 outer
        a = np.sqrt(bigR * bigR - agcct_outerSmall * agcct_outerSmall)
        eta0 = 0.5 * np.log((bigR + a) / (bigR - a))
        ch_eta0 = np.cosh(eta0)
        sh_eta0 = np.sinh(eta0)

        k[0] = -(1.0 / np.tan(agcct_tiltAngles[0] / 180.0 * PI)) / ((0 + 1) * sh_eta0)
        k[1] = -(1.0 / np.tan(agcct_tiltAngles[1] / 180.0 * PI)) / ((1 + 1) * sh_eta0)
        k[2] = -(1.0 / np.tan(agcct_tiltAngles[2] / 180.0 * PI)) / ((2 + 1) * sh_eta0)
        k[3] = -(1.0 / np.tan(agcct_tiltAngles[3] / 180.0 * PI)) / ((3 + 1) * sh_eta0)

        ksi_deg0 = 0

        while i < dicct_windingNumber * 2 + agcct_windingNumbers[0] * 2 + agcct_windingNumbers[1] * 2 + \
                agcct_windingNumbers[2]:
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 0] = ksi_deg0 + 360 * (
                    i - dicct_windingNumber * 2 - agcct_windingNumbers[0] * 2 - agcct_windingNumbers[1] -
                    agcct_windingNumbers[2])
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 1] = agcct_phi0s[1]

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 2] = k[0]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 3] = k[1]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 4] = k[2]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 5] = k[3]

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 6] = a

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 7] = ch_eta0
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 8] = sh_eta0

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 9] = agcct_current

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 10] = agcct_bendingRadians[0] + agcct_phi0s[0]
            i += 1

    if True:  # ag cct3 outer
        a = np.sqrt(bigR * bigR - agcct_outerSmall * agcct_outerSmall)
        eta0 = 0.5 * np.log((bigR + a) / (bigR - a))
        ch_eta0 = np.cosh(eta0)
        sh_eta0 = np.sinh(eta0)

        k[0] = -(1.0 / np.tan(agcct_tiltAngles[0] / 180.0 * PI)) / ((0 + 1) * sh_eta0)
        k[1] = -(1.0 / np.tan(agcct_tiltAngles[1] / 180.0 * PI)) / ((1 + 1) * sh_eta0)
        k[2] = -(1.0 / np.tan(agcct_tiltAngles[2] / 180.0 * PI)) / ((2 + 1) * sh_eta0)
        k[3] = -(1.0 / np.tan(agcct_tiltAngles[3] / 180.0 * PI)) / ((3 + 1) * sh_eta0)

        agcct_current *= -1
        agcct_phi0s[2] *= -1

        ksi_deg0 = -360 * agcct_windingNumbers[2]

        while i < dicct_windingNumber * 2 + agcct_windingNumbers[0] * 2 + agcct_windingNumbers[1] * 2 + \
                agcct_windingNumbers[2] * 2:
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 0] = ksi_deg0 + 360 * (
                    i - dicct_windingNumber * 2 - agcct_windingNumbers[0] * 2 - agcct_windingNumbers[1] * 2 -
                    agcct_windingNumbers[2])
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 1] = agcct_phi0s[2]

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 2] = k[0]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 3] = k[1]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 4] = k[2]
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 5] = k[3]

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 6] = a

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 7] = ch_eta0
            data[i * NUMBER_OF_VARIABLES_PER_CCT + 8] = sh_eta0

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 9] = agcct_current

            data[i * NUMBER_OF_VARIABLES_PER_CCT + 10] = agcct_bendingRadians[0] + agcct_phi0s[0] + \
                                                         agcct_bendingRadians[1] + agcct_phi0s[1]
            i += 1

        agcct_current *= -1
        agcct_phi0s[2] *= -1

    return data


def list_cct_data_generate(param_list: np.ndarray) -> np.ndarray:
    gantry_number = param_list.shape[0]  # 机架数目
    all_data = np.empty((gantry_number * CUDA_THREAD_SIZE_X * NUMBER_OF_VARIABLES_PER_CCT,), dtype=np.float32)
    for i in range(gantry_number):
        all_data[i * CUDA_THREAD_SIZE_X * NUMBER_OF_VARIABLES_PER_CCT:
                 (i + 1) * CUDA_THREAD_SIZE_X * NUMBER_OF_VARIABLES_PER_CCT] = \
            __cct_data_generate(param_list[i])

    return all_data


def __qs_data_generate(param: np.ndarray) -> np.ndarray:
    return np.array([
        param[0], param[1]
    ], dtype=np.float32)
    # qs_data = np.array([-7.3733, -45.31 * 2.0], dtype=np.float32)


def list_qs_data_generate(param_list: np.ndarray) -> np.ndarray:
    gantry_number = param_list.shape[0]  # 机架数目
    qs_data = np.empty((gantry_number * 2,), dtype=np.float32)
    for i in range(gantry_number):
        qs_data[i * 2:(i + 1) * 2] = __qs_data_generate(param_list[i])

    return qs_data


if __name__ == '__main__':
    param: np.ndarray = np.array([-7.3733, -45.31 * 2.0, 80., 90., 90., 90., 90., 90., -9664., -6000.])

    data = __cct_data_generate(param)

    for i in range(data.shape[0]):
        print(f"{i}  {data[i]}")

    data = __qs_data_generate(param)

    for i in range(data.shape[0]):
        print(f"{i}  {data[i]}")
