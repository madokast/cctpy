"""
抽象 / 一般性对象
"""

import numpy as np
from typing import List, Tuple

from cctpy.baseutils import Vectors, Equal, Stream, Converter, Circle, Debug
from cctpy.constant import ORIGIN3, XI, ZI, ZERO3, MM


class Magnet:
    """
    表示一个可以求磁场的对象，如 CCT 、 QS 磁铁
    所有实现此接口的类，可以计算出它在某一点的磁场

    本类（接口）只有一个接口方法 magnetic_field_at(self, point: np.ndarray)
    """

    def magnetic_field_at(self, point: np.ndarray) -> np.ndarray:
        """
        获得本对象 self 在点 point 处产生的磁场
        这个方法需要在子类中实现/重写
        ----------
        point 三维笛卡尔坐标系中的点，即一个三维矢量，如 [0,0,0]

        Returns 本对象 self 在点 point 处的磁场，用三维矢量表示
        -------
        """
        raise NotImplementedError

    def magnetic_field_along(self, line: np.ndarray) -> np.ndarray:
        """
        计算本对象在三维曲线 line 上的磁场分布
        ----------
        line 由离散点组成的三维曲线，即三维矢量的数组，如 [[0,0,0], [1,0,0], [2,0,0]]

        Returns 本对象在三维曲线 line 上的磁场分布，用三维矢量的数组表示
        -------
        """
        length = line.shape[0]  # 曲线上离散点的数目
        fields = np.empty((length, 3), dtype=np.float64)  # 提前开辟空间
        for i in range(length):
            fields[i, :] = self.magnetic_field_at(line[i, :])
        return fields


class LocalCoordinateSystem:
    """
    局部坐标系，各种磁铁需要指定它所在的局部坐标系才能产生磁场，同时也便于磁铁调整位置

    局部坐标系由位置 location 、主方向 main_direction 和次方向 second_direction 确定

    一般而言，默认元件入口中心处，即元件的位置

    一般而言，主方向 main_direction 表示理想粒子运动方向，一般是 z 方向
    次方向 second_direction 垂直于主方向，并且在相空间分析中看作 x 方向
    """

    def __init__(self, location: np.ndarray = ORIGIN3, main_direction: np.ndarray = ZI,
                 second_direction: np.ndarray = XI):
        """
        指定实体的位置和朝向
        Parameters
        ----------
        location 全局坐标系中实体位置，默认全局坐标系的远点
        main_direction 主朝向，默认全局坐标系 z 方向
        second_direction 次朝向，默认全局坐标系 x 方向
        """
        Equal.require_float_equal(
            np.inner(main_direction, second_direction), 0.0,
            f"创建 LocalCoordinateSystem 对象异常，main_direction{main_direction}和second_direction{second_direction}不正交"
        )

        # 局部坐标系，原心
        self.location = location.copy()

        # 局部坐标系的 x y z 三方向
        self.ZI = Vectors.normalize_self(main_direction.copy())
        self.XI = Vectors.normalize_self(second_direction.copy())
        self.YI = np.cross(self.ZI, self.XI)

    def point_to_local_coordinate(self, global_coordinate_point: np.ndarray) -> np.ndarray:
        """
        全局坐标系 -> 局部坐标系
        Parameters
        ----------
        global_coordinate_point 全局坐标系中的点

        Returns 这一点在局部坐标系中的坐标
        -------
        """
        location_to_global_coordinate = global_coordinate_point - self.location
        x = np.inner(self.XI, location_to_global_coordinate)
        y = np.inner(self.YI, location_to_global_coordinate)
        z = np.inner(self.ZI, location_to_global_coordinate)
        return np.array([x, y, z], dtype=np.float64)

    def point_to_global_coordinate(self, local_coordinate_point: np.ndarray) -> np.ndarray:
        """
        局部坐标系 -> 全局坐标系
        Parameters
        ----------
        local_coordinate_point 局部坐标系

        Returns 全局坐标系
        -------

        """

        return self.location + (
                local_coordinate_point[0] * self.XI +
                local_coordinate_point[1] * self.YI +
                local_coordinate_point[2] * self.ZI
        )

    def line_to_local_coordinate(self, global_coordinate_line: np.ndarray) -> np.ndarray:
        """
        全局坐标系中的 线/点集 坐标转为局部坐标系
        线/点集，为 n*3 的矩阵，矩阵每一行代表一个点
        Parameters
        ----------
        global_coordinate_line 全局坐标系中的 线/点集

        Returns 转为局部坐标系
        -------

        """
        length = global_coordinate_line.shape[0]  # 点数目
        location_line = np.empty((length, 3), dtype=np.float64)  # 提前开辟空间
        for i in range(length):
            location_line[i, :] = self.point_to_local_coordinate(global_coordinate_line[i, :])
        return location_line

    def line_to_global_coordinate(self, local_coordinate_line: np.ndarray) -> np.ndarray:
        """
        局部坐标系中的 线/点集 坐标转为全局坐标系
        线/点集，为 n*3 的矩阵，矩阵每一行代表一个点
        Parameters
        ----------
        local_coordinate_line 局部坐标系中的 线/点集

        Returns 转为全局坐标系
        -------

        """
        length = local_coordinate_line.shape[0]  # 点数目
        global_line = np.empty((length, 3), dtype=np.float64)  # 提前开辟空间
        for i in range(length):
            global_line[i, :] = self.point_to_global_coordinate(local_coordinate_line[i, :])
        return global_line

    def set_location(self, location: np.ndarray):
        self.location = location.copy()

    def set_direction(self, main_direction: np.ndarray, second_direction: np.ndarray) -> None:
        self.ZI = Vectors.normalize_self(main_direction.copy())
        self.XI = Vectors.normalize_self(second_direction.copy())
        self.YI = np.cross(self.ZI, self.XI)

    def to_float32(self):
        self.location = self.location.astype(np.float32)
        self.XI = self.XI.astype(np.float32)
        self.YI = self.YI.astype(np.float32)
        self.ZI = self.ZI.astype(np.float32)
        return self

    def __str__(self) -> str:
        return f"ORIGIN={self.location}, xi={self.XI}, yi={self.YI}, zi={self.ZI}"

    @staticmethod
    def create_by_y_and_z_direction(location: np.ndarray, y_direction: np.ndarray, z_direction: np.ndarray):
        """
        由 原点 location y方向 y_direction 和 z方向 z_direction 创建坐标系
        Parameters
        ----------
        location 原点
        y_direction y方向
        z_direction z方向

        Returns 坐标系
        -------

        """
        Equal.require_float_equal(
            np.inner(y_direction, z_direction), 0.0,
            f"创建 LocalCoordinateSystem 对象异常，y_direction{y_direction}和z_direction{z_direction}不正交"
        )

        x_direction = np.cross(y_direction, z_direction)
        return LocalCoordinateSystem(location, z_direction, x_direction)

    @staticmethod
    def global_coordinate_system():
        """
        获取全局坐标系
        Returns 全局坐标系
        -------

        """
        return LocalCoordinateSystem()


class Plotable:
    """
    表示一个可以进行绘图的对象
    """

    def line_and_color(self, describe='r') -> List[Tuple[np.ndarray, str]]:
        """
        返回用于绘图的 线 和 绘图选项（如线型、线颜色、线粗细，默认红色）
        Returns [(线，绘图选项)]
        -------
        """
        raise NotImplementedError


class Line2(Plotable):
    """
    二维 xy 平面的一条有起点和终点的连续曲线段，可以是直线、圆弧
    本类包含 3 个抽象方法，需要实现：
    get_length 获得曲线长度
    point_at 从曲线起点出发，s 位置处的点
    direct_at 从曲线起点出发，s 位置处曲线方向

    说明：这个类主要用于构建 “理想轨道”，理想轨道的用处很多：
    1. 获取理想轨道上的理想粒子；
    2. 研究理想轨道上的磁场分布

    """

    def get_length(self) -> float:
        """
        获得曲线的长度
        Returns 曲线的长度
        -------

        """
        raise NotImplementedError

    def point_at(self, s: float) -> np.ndarray:
        """
        获得曲线 s 位置处的点 (x,y)
        即从曲线起点出发，运动 s 长度后的位置
        Parameters
        ----------
        s 长度量，曲线上 s 位置

        Returns 曲线 s 位置处的点 (x,y)
        -------

        """
        raise NotImplementedError

    def direct_at(self, s: float) -> np.ndarray:
        """
        获得 s 位置处，曲线的方向
        Parameters
        ----------
        s 长度量，曲线上 s 位置

        Returns s 位置处，曲线的方向
        -------

        """
        raise NotImplementedError

    def right_hand_side_point(self, s: float, d: float) -> np.ndarray:
        """
        位于 s 处的点，它右手边 d 处的点

         1    5    10     15
         -----------------@------>
         |2
         |4               *
        如上图，一条直线，s=15，d=4 ,即点 @ 右手边 4 距离处的点 *

        说明：这个方法，主要用于四极场、六极场的计算，因为需要涉及轨道横向位置的磁场

        Parameters
        ----------
        s 长度量，曲线上 s 位置
        d 长度量，d 距离远处

        Returns 位于 s 处的点，它右手边 d 处的点
        -------

        """
        ps = self.point_at(s)

        # 方向
        ds = self.direct_at(s)

        return ps + Vectors.update_length(
            Vectors.rotate_self_z_axis(ds.copy(), -np.pi / 2),
            d
        )

    def left_hand_side_point(self, s: float, d: float) -> np.ndarray:
        """
        位于 s 处的点，它左手边 d 处的点
        说明见 right_hand_side_point 方法
        Parameters
        ----------
        s 长度量，曲线上 s 位置
        d 长度量，d 距离远处

        Returns 位于 s 处的点，它左手边 d 处的点
        -------

        """
        return self.right_hand_side_point(s, -d)

    # ------------------------------端点性质-------------------- #
    def point_at_start(self):
        return self.point_at(0.0)

    def point_at_end(self):
        return self.point_at(self.get_length())

    def direct_at_start(self):
        return self.direct_at(0.0)

    def direct_at_end(self):
        return self.direct_at(self.get_length())

    # ------------------------------平移-------------------- #
    def __add__(self, v2: np.ndarray):
        """
        Line2 的平移， v2 表示移动的方向和距离
        Parameters
        ----------
        v2 二维向量

        Returns 平移后的 Line2
        -------

        """

        class MovedLine2(Line2):
            def __init__(self, hold):
                self.hold = hold

            def get_length(self) -> float:
                return self.hold.get_length()

            def point_at(self, s: float) -> np.ndarray:
                return self.hold.point_at(s) + v2

            def direct_at(self, s: float) -> np.ndarray:
                return self.hold.direct_at(s)

        return MovedLine2(self)

    # ------------------------------ 离散 ------------------------#
    def disperse2d(self, step: float = 1.0 * MM) -> np.ndarray:
        """
        二维离散轨迹点
        Parameters
        ----------
        step 步长

        Returns 二维离散轨迹点
        -------

        """
        number: int = int(self.get_length() / step)
        return Stream.linspace(0, self.get_length(), number).map(
            lambda t: self.point_at(t)
        ).to_vector()

    def disperse3d(self, step: float = 1.0 * MM) -> np.ndarray:
        """
        三维离散轨迹点，其中第三维 z == 0.0
        Parameters
        ----------
        step 步长

        Returns 二维离散轨迹点
        -------

        """
        number: int = int(self.get_length() / step)
        return Stream.linspace(0, self.get_length(), number).map(
            lambda t: np.hstack([self.point_at(t), [0.0]])
        ).to_vector()

    # ------------------------------画图-------------------- #
    def line_and_color(self, describe='r') -> List[Tuple[np.ndarray, str]]:
        line3 = self.disperse3d()
        return [(line3, describe)]

    def __str__(self):
        return f"Line2，起点{self.point_at_start()}，长度{self.get_length()}"


class StraightLine2(Line2):
    """
    二维直线段，包含三个参数：长度、方向、起点
    """

    def __init__(self, length: float, direct: np.ndarray, start_point: np.ndarray):
        self.length = length
        self.direct = direct
        self.start_point = start_point

    def get_length(self) -> float:
        return self.length

    def point_at(self, s: float) -> np.ndarray:
        return self.start_point + Vectors.update_length(
            self.direct.copy(),
            s
        )

    def direct_at(self, s: float) -> np.ndarray:
        return self.direct

    def __str__(self):
        return f"直线段，起点{self.start_point}，方向{self.direct}，长度{self.length}"


class ArcLine2(Line2):
    """
    二维圆弧段
    借助极坐标的思想来描述圆弧
    基础属性： 圆弧的半径 radius、圆弧的圆心 center
    起点描述：极坐标 phi 值
    弧长：len = radius * totalPhi

    起点start_point、圆心center、半径radius、旋转方向clockwise、角度totalPhi 五个自由度
    起点弧度值 starting_phi、起点处方向、半径radius、旋转方向clockwise、角度totalPhi 五个自由度

    如图： *1 表示起点方向，@ 是圆心，上箭头 ↑ 是起点处方向，旋转方向是顺时针，*5 是终点，因此角度大约是 80 deg
                *5
           *4
       *3
     *2
    *1     ↑       @

    """

    def __init__(self, starting_phi: float, center: np.ndarray, radius: float, total_phi: float, clockwise: bool):
        self.starting_phi = starting_phi
        self.center = center
        self.radius = radius
        self.total_phi = total_phi
        self.clockwise = clockwise
        self.length = radius * total_phi

    def get_length(self) -> float:
        return self.length

    def point_at(self, s: float) -> np.ndarray:
        phi = s / self.radius
        current_phi = self.starting_phi - phi if self.clockwise else self.starting_phi + phi

        uc = Circle.unit_circle(current_phi)

        return Vectors.update_length(uc, self.radius) + self.center

    def direct_at(self, s: float) -> np.ndarray:
        phi = s / self.radius
        current_phi = self.starting_phi - phi if self.clockwise else self.starting_phi + phi

        uc = Circle.unit_circle(current_phi)

        return Vectors.rotate_self_z_axis(
            uc,
            -np.pi / 2 if self.clockwise else np.pi / 2
        )

    @staticmethod
    def create(start_point: np.ndarray, start_direct: np.ndarray, radius: float, clockwise: bool, total_deg: float):
        center: np.ndarray = start_point + Vectors.update_length(
            Vectors.rotate_self_z_axis(start_direct.copy(), -np.pi / 2 if clockwise else np.pi / 2),
            radius
        )

        starting_phi = Vectors.angle_to_x_axis(start_point - center)

        total_phi = Converter.angle_to_radian(total_deg)

        return ArcLine2(starting_phi, center, radius, total_phi, clockwise)

    def __str__(self):
        return f"弧线段，起点{self.point_at_start()}，方向{self.direct_at_start()}，顺时针{self.clockwise}，半径{self.radius}，角度{self.total_phi}"


class Trajectory(Line2):
    """
    二维轨迹，由直线+圆弧组成
    """

    def __init__(self, first_line2: Line2):
        """
        构造器，传入第一条线 first_line2
        Parameters
        ----------
        first_line2 第一条线
        -------

        """
        self.__trajectoryList = [first_line2]
        self.__length = first_line2.get_length()
        self.__point_at_error_happen = False  # 是否发生 point_at 错误

    def add_strait_line(self, length: float):
        """
        尾接直线
        Parameters
        ----------
        length 直线长度

        Returns self
        -------

        """
        last_line = self.__trajectoryList[-1]
        sp = last_line.point_at_end()
        sd = last_line.direct_at_end()

        sl = StraightLine2(length, sd, sp)

        self.__trajectoryList.append(sl)
        self.__length += length

        return self

    def add_arc_line(self, radius: float, clockwise: bool, angle_deg: float):
        """
        尾接圆弧
        Parameters
        ----------
        radius 半径
        clockwise 顺时针？
        angle_deg 角度

        Returns self
        -------

        """
        last_line = self.__trajectoryList[-1]
        sp = last_line.point_at_end()
        sd = last_line.direct_at_end()

        al = ArcLine2.create(sp, sd, radius, clockwise, angle_deg)

        self.__trajectoryList.append(al)
        self.__length += al.get_length()

        return self

    def get_length(self) -> float:
        return self.__length

    def point_at(self, s: float) -> np.ndarray:
        s0 = s

        for line in self.__trajectoryList:
            if s <= line.get_length():
                return line.point_at(s)
            else:
                s -= line.get_length()

        last_line = self.__trajectoryList[-1]

        # 2020年4月2日
        # 解决了一个因为浮点数产生的巨大bug
        if np.abs(s) <= 1e-8:
            return last_line.point_at_end()

        if not self.__point_at_error_happen:
            self.__point_at_error_happen = True
            print(f"ERROR Trajectory::point_at{s0}")
            Debug.print_traceback()

        return last_line.point_at(s)

    def direct_at(self, s: float) -> np.ndarray:
        s0 = s

        for line in self.__trajectoryList:
            if s <= line.get_length():
                return line.direct_at(s)
            else:
                s -= line.get_length()

        last_line = self.__trajectoryList[-1]

        # 2020年4月2日
        # 解决了一个因为浮点数产生的巨大bug
        if np.abs(s) <= 1e-8:
            return last_line.direct_at_end()

        if not self.__point_at_error_happen:
            self.__point_at_error_happen = True
            print(f"ERROR Trajectory::direct_at{s0}")
            Debug.print_traceback()

        return last_line.direct_at(s)

    def __str__(self):
        details = '\t\n'.join(self.__trajectoryList.__str__())
        return f"Trajectory:{details}"
