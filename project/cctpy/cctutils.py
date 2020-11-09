"""
CCT 工具类
"""

from cctpy.constant import ZERO3, ZI
from numpy import ndarray, inner, cross
from cctpy.baseutils import length, normalize_locally



class QS(Magnet, Plotable):
    """
    硬边 QS 磁铁，详见__init__方法
    """

    def __init__(self, length: float, gradient: float, second_gradient: float,
                 aperture_radius: float, location: ndarray, direct: ndarray) -> None:
        """
        硬边 QS 磁铁，由以下参数完全确定：

        length 磁铁长度 / m
        gradient 四极场梯度 / Tm-1
        second_gradient 六极场梯度 / Tm-2
        aperture_radius 孔径（半径） / m
        location QS 磁铁入口处中心位置，三维矢量 / [m,m,m]
        direct QS 磁铁朝向，三维矢量 / [m,m,m]

        //                  ------------------------------------------
        // 　               |     下箭头表示朝向                      |
        // 这里就是位置点  ->|       ---->                            |
        //  　              |                                        |
        //                  ------------------------------------------
        """
        super(QS, self).__init__()
        self.length = length
        self.gradient = gradient
        self.second_gradient = second_gradient
        self.aperture_radius = aperture_radius
        self.location = location
        self.direct = direct

        # 局部坐标系
        self.zi = normalize_locally(direct.copy())
        self.yi = ZI
        self.xi = cross(self.yi, self.zi)

    def magnetic_field_at(self, point: ndarray) -> ndarray:
        """
        实现 Magnet 接口
        """
        local_to_point = point - self.location  # 磁铁起点到 point
        z = inner(self.zi, local_to_point)  # 磁铁起点定义的坐标系中，point 的 z 坐标值
        if z < 0 or z > self.length:  # 在磁铁边界外
            return ZERO3
        else:
            project_point = self.location + self.direct * z
            if length(point - project_point) > self.aperture_radius:  # 超出孔径
                return ZERO3
            else:
                x = inner(self.xi, local_to_point)  # 磁铁起点定义的坐标系中，point 的 x 坐标值
                y = inner(self.yi, local_to_point)  # 磁铁起点定义的坐标系中，point 的 y 坐标值

                by_quad = -self.gradient * x
                bx_quad = -self.gradient * y

                by_sext = self.second_gradient * (x * x - y * y)
                bx_sext = self.second_gradient * (2. * x * y)

                By = (by_quad + by_sext) * self.yi
                Bx = (bx_quad + bx_sext) * self.xi

                return Bx + By
