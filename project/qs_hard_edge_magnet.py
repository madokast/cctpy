"""
QS 硬边磁铁
"""
from typing import List, Tuple

import numpy as np

from abstract_classes import Magnet, Plotable, LocalCoordinateSystem
from constant import ZERO3


class QsHardEdgeMagnet(Magnet, Plotable):
    """
    硬边 QS 磁铁，由以下参数完全确定：

    length 磁铁长度 / m
    gradient 四极场梯度 / Tm-1
    second_gradient 六极场梯度 / Tm-2
    aperture_radius 孔径（半径） / m
    local_coordinate_system 局部坐标系

    局部坐标系的含义见下：

                      ------------------------------------------
     　               |        ②                               |
                 ① ->|       ---->           ③ ↑              |
      　              |                                        |
                      ------------------------------------------

    ① QS 磁铁入口中心位置，是局部坐标系的原心
    ② 理想粒子运动方向，是局部坐标系 z 方向
    ③ 像空间中 x 方向
    另外 y 方向有 x 方向和 z 方向确定

    """

    def __init__(self, length: float, gradient: float, second_gradient: float,
                 aperture_radius: float, local_coordinate_system: LocalCoordinateSystem):
        self.length = length
        self.gradient = gradient
        self.second_gradient = second_gradient
        self.aperture_radius = aperture_radius
        self.local_coordinate_system = local_coordinate_system

    def magnetic_field_at(self, point: np.ndarray) -> np.ndarray:
        """
        qs 磁铁在全局坐标系点 point 处产生的磁场
        Parameters
        ----------
        point 全局坐标系点
        Returns qs 磁铁在全局坐标系点 point 处产生的磁场
        -------
        """
        # point 转为局部坐标，并拆包
        x, y, z = self.local_coordinate_system.point_to_local_coordinate(point)

        # z < 0 or z > self.length 表示点 point 位于磁铁外部
        if z < 0 or z > self.length:
            return ZERO3
        else:
            # 以下判断点 point 是不是在孔径外，前两个 or 是为了快速短路判断，避免不必要的开方计算
            if np.abs(x) > self.aperture_radius or np.abs(y) > self.aperture_radius or np.sqrt(
                    x ** 2 + y ** 2) > self.aperture_radius:
                return ZERO3
            else:
                # bx 和 by 分别是局部坐标系中 x 和 y 方向的磁场（局部坐标系中 z 方向是束流方向，不会产生磁场）
                bx = self.gradient * y - self.second_gradient * (x * y)
                by = self.gradient * x + 0.5 * self.second_gradient * (x ** 2 - y ** 2)

                # 转移到全局坐标系中
                return bx * self.local_coordinate_system.XI + by * self.local_coordinate_system.YI

    def line_and_color(self, describe=None) -> List[Tuple[np.ndarray, str]]:
        """
        画图相关
        """
        front_circle_local = np.array([
            [self.aperture_radius * np.cos(i / 180 * np.pi),
             self.aperture_radius * np.sin(i / 180 * np.pi),
             0.]
            for i in range(360)])
        mid_circle_local = front_circle_local + np.array([0, 0, self.length / 2])
        back_circle_local = front_circle_local + np.array([0, 0, self.length])

        front_circle = self.local_coordinate_system.line_to_local_coordinate(front_circle_local)
        mid_circle = self.local_coordinate_system.line_to_local_coordinate(mid_circle_local)
        back_circle = self.local_coordinate_system.line_to_local_coordinate(back_circle_local)

        axial_direction_line_0 = np.array([front_circle[0], back_circle[0]])
        axial_direction_line_1 = np.array([front_circle[90], back_circle[90]])
        axial_direction_line_2 = np.array([front_circle[180], back_circle[180]])
        axial_direction_line_3 = np.array([front_circle[270], back_circle[270]])

        return [
            (front_circle, describe),
            (mid_circle, describe),
            (back_circle, describe),
            (axial_direction_line_0, describe),
            (axial_direction_line_1, describe),
            (axial_direction_line_2, describe),
            (axial_direction_line_3, describe),
        ]
