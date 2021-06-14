"""
CCT 建模优化代码
OPERA 扩展，主要包括两个功能：
1. CCT线圈生成导体 cond 文件，可以导入 OPERA
    使用 OperaConductor.create_by_cct(cct,槽宽,槽深,标签,每周分段数目) 
2. OPERA 磁铁导入

作者：赵润晓
日期：2021年6月3日
"""
from typing import List
from packages.point import P3
from packages.magnets import Magnet
from packages.cct import CCT
from packages.constants import M


# OPERA 导体文件头
OPERA_CONDUCTOR_SCRIPT_HEAD: str = "CONDUCTOR\n"
# OPERA 导体文件尾
OPERA_CONDUCTOR_SCRIPT_TAIL: str = "QUIT\nEOF\n"


class Brick8:
    """
    opera 中 8 点导线立方体
    对应 opera 中脚本：
    --------------------------------------------------
    DEFINE BR8
    0.0 0.0 0.0 0.0 0.0
    0.0 0.0 0.0
    0.0 0.0 0.0
    1.054000e+00 5.651710e-04 -8.249738e-04
    1.046000e+00 5.651710e-04 -8.249738e-04
    1.046000e+00 -5.651710e-04 8.249738e-04
    1.054000e+00 -5.651710e-04 8.249738e-04
    1.004041e+00 1.474080e-01 1.026480e-01
    9.981663e-01 1.465494e-01 9.728621e-02
    9.973407e-01 1.451229e-01 9.841917e-02
    1.003216e+00 1.459815e-01 1.037810e-01
    3.4575E8 1  'layer1'
    0 0 0
    1.0
    --------------------------------------------------
    脚本中各参数的意义如下：
    --------------------------------------------------
    XCENTRE, YCENTRE, ZCENTRE, PHI1, THETA1, PSI1           # Local coordinate system 1
    XCEN2, YCEN2, ZCEN2                                     # Local coordinate system 2 (origin)
    THETA2, PHI2, PSI2                                      # Local coordinate system 2 (Euler angles)
    XP1, YP1, ZP1                                           #  Bottom right corner of front face
    XP2, YP2, ZP2                                           #  Top right corner of front face
    XP3, YP3, ZP3                                           #  Top left corner of front face
    XP4, YP4, ZP4                                           #  Bottom left corner of front face
    XP5, YP5, ZP5                                           #  Bottom right corner of back face
    XP6, YP6, ZP6                                           #  Top right corner of back face
    XP7, YP7, ZP7                                           #  Top left corner of back face
    XP8, YP8, ZP8                                           #  Bottom left corner of back face
    CURD, SYMMETRY, DRIVELABEL                              #  Current density, symmetry and drive label
    IRXY, IRYZ,IRZX                                         #  Reflections in local coordinate system 1 coordinate planes
    TOLERANCE Flux                                          #  density tolerance
    --------------------------------------------------
    """

    HEAD = 'DEFINE BR8\n0.0 0.0 0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n'
    TAIL = '0 0 0\n1.0\n'

    def __init__(self,
                 front_face_point1: P3,
                 front_face_point2: P3,
                 front_face_point3: P3,
                 front_face_point4: P3,
                 back_face_point1: P3,
                 back_face_point2: P3,
                 back_face_point3: P3,
                 back_face_point4: P3,
                 current_density: float,
                 label: str
                 ) -> None:
        """
        front_face_point1234 立方体前面的四个点
        back_face_point1234 立方体后面的四个点

        所谓的前面/后面，指的是按照电流方向（电流从前面流入，后面流出，参考 opera ref-3d 手册）
        前面  -> 电流 -> 后面
        """
        self.front_face_point1 = front_face_point1
        self.front_face_point2 = front_face_point2
        self.front_face_point3 = front_face_point3
        self.front_face_point4 = front_face_point4
        self.back_face_point1 = back_face_point1
        self.back_face_point2 = back_face_point2
        self.back_face_point3 = back_face_point3
        self.back_face_point4 = back_face_point4
        self.current_density = current_density
        self.label = label

    def to_opera_cond(self) -> str:
        def p3_str(p: P3) -> str:
            return f'{p.x} {p.y} {p.z}\n'
        front_face_point1_str = p3_str(self.front_face_point1)
        front_face_point2_str = p3_str(self.front_face_point2)
        front_face_point3_str = p3_str(self.front_face_point3)
        front_face_point4_str = p3_str(self.front_face_point4)

        back_face_point1_str = p3_str(self.back_face_point1)
        back_face_point2_str = p3_str(self.back_face_point2)
        back_face_point3_str = p3_str(self.back_face_point3)
        back_face_point4_str = p3_str(self.back_face_point4)

        current_label_str = f"{self.current_density} 1 '{self.label}'\n"

        return "".join((
            Brick8.HEAD,
            front_face_point1_str,
            front_face_point2_str,
            front_face_point3_str,
            front_face_point4_str,

            back_face_point1_str,
            back_face_point2_str,
            back_face_point3_str,
            back_face_point4_str,

            current_label_str,
            Brick8.TAIL
        ))


class Brick8s:
    """
    opera 中连续的 8 点导体立方体
    所谓连续，指的是前后两个立方体，各有一个面重合
    """

    def __init__(self,
                 line1: List[P3],
                 line2: List[P3],
                 line3: List[P3],
                 line4: List[P3],
                 current_density: float,
                 label: str
                 ) -> None:
        """
        line1 ~ line4 截面为矩形的导体四条棱
        current_density 电流密度
        label 导体标签（drive label）
        """
        self.line1 = line1
        self.line2 = line2
        self.line3 = line3
        self.line4 = line4
        self.current_density = current_density
        self.label = label

    def to_brick8_list(self) -> List[Brick8]:
        """
        转换为 Brick8 数组
        """
        bricks_list = []
        size = len(self.line1)
        for i in range(size-1):
            bricks_list.append(Brick8(
                self.line1[i],
                self.line2[i],
                self.line3[i],
                self.line4[i],
                self.line1[i+1],
                self.line2[i+1],
                self.line3[i+1],
                self.line4[i+1],
                self.current_density,
                self.label
            ))

        return bricks_list

    def to_opera_cond_string(self) -> str:
        bricks_list = self.to_brick8_list()
        return "\n".join([e.to_opera_cond() for e in bricks_list])

    @staticmethod
    def create_by_cct(cct: CCT, channel_width: float, channel_depth: float,
                      label: str, disperse_number_per_winding: int) -> 'Brick8s':
        """
        从 CCT 创建 Brick8s
        channel_width channel_depth 槽的宽度和深度
        label 标签
        disperse_number_per_winding 每匝分段数目

        注意：转为 Brick8s 时，没有进行坐标转换，即在 CCT 的局部坐标系中建模
        """
        delta = 1e-6

        # 路径方程
        def path3(ksi):
            return cct.p3_function(ksi)

        # 切向 正则归一化
        def tangential_direct(ksi):
            return ((path3(ksi+delta)-path3(ksi))/delta).normalize()

        # 主法线方向 注意：已正则归一化
        def main_normal_direct(ksi):
            return cct.bipolar_toroidal_coordinate_system.main_normal_direction_at(cct.p2_function(ksi))

        # 副法线方向
        def second_normal_direc(ksi):
            return (tangential_direct(ksi)@main_normal_direct(ksi)).normalize()

        def channel_path1(ksi):
            return (path3(ksi)
                    + (channel_depth/2) * main_normal_direct(ksi)
                    + (channel_width/2) * second_normal_direc(ksi)
                    )

        def channel_path2(ksi):
            return (path3(ksi)
                    - (channel_depth/2) * main_normal_direct(ksi)
                    + (channel_width/2) * second_normal_direc(ksi)
                    )

        def channel_path3(ksi):
            return (path3(ksi)
                    - (channel_depth/2) * main_normal_direct(ksi)
                    - (channel_width/2) * second_normal_direc(ksi)
                    )

        def channel_path4(ksi):
            return (path3(ksi)
                    + (channel_depth/2) * main_normal_direct(ksi)
                    - (channel_width/2) * second_normal_direc(ksi)
                    )

        start_ksi = cct.starting_point_in_ksi_phi_coordinate.x
        end_ksi = cct.end_point_in_ksi_phi_coordinate.x
        # +1 为了 linspace 获得正确分段结果
        total_disperse_number = disperse_number_per_winding * cct.winding_number + 1

        ksi_list = BaseUtils.linspace(
            start_ksi, end_ksi, total_disperse_number)

        return Brick8s(
            [channel_path1(ksi) for ksi in ksi_list],
            [channel_path2(ksi) for ksi in ksi_list],
            [channel_path3(ksi) for ksi in ksi_list],
            [channel_path4(ksi) for ksi in ksi_list],
            current_density=cct.current / (channel_width*channel_depth),
            label=label
        )

    def get_brick8_number(self) -> int:
        """
        返回 brick8 导体数目
        """
        return len(self.line1)-1


class OperaConductorScript:
    """
    一个纯静态类
    """
    @staticmethod
    def to_opera_cond_script(brick8s_list: List[Brick8s]) -> str:
        ret = [OPERA_CONDUCTOR_SCRIPT_HEAD]
        for b in brick8s_list:
            ret.append(b.to_opera_cond_string())

        ret.append(OPERA_CONDUCTOR_SCRIPT_TAIL)

        return '\n'.join(ret)

    @staticmethod
    def to_opera_cond_file(brick8s_list: List[Brick8s], file_name="cond.cond") -> None:
        num = 0
        for b in brick8s_list:
            num += b.get_brick8_number()

        print(f"共检测到 {num} 个 Brick8 导体，开始导出...")

        f = open(file_name, "w")
        f.write(OperaConductorScript.to_opera_cond_script(brick8s_list))
        f.close()

        print(f"opera 导体文件 {file_name} 导出成功")


class FieldWIthPosion:
    """
    磁场及其所在位置
    用于 OperaFieldTableMagnet 内部
    """

    def __init__(self, position: P3, field: P3) -> None:
        self.position = position
        self.field = field

    def __str__(self) -> str:
        return f"pos={self.position}, field={self.field}"

    def __repr__(self) -> str:
        return self.__str__()


class OperaFieldTableMagnet(Magnet):
    """
    利用 opera 磁场表格文件生成的磁场
    文件主体应为 6 列，分别是 x y z bx by bz
    """

    def __init__(self, file_name: str,
                 first_corner_x: float, first_corner_y: float, first_corner_z: float,
                 step_between_points_x: float, step_between_points_y: float, step_between_points_z: float,
                 number_of_points_x: int, number_of_points_y: int, number_of_points_z: int,
                 unit_of_length: float = 1*M, unit_of_field: float = 1
                 ) -> None:
        """
        first_corner_x / y / z 即 opera 导出磁场时 First corner 填写的值。单位由 unit_of_length 指定
        step_between_points_x / y / z 即 opera 导出磁场时 Step between points 填写的值。单位由 unit_of_length 指定
        number_of_points_x / y / z 即 opera 导出磁场时 Number of points 填写的值
        unit_of_length 是数据中长度单位，默认米。（注意 opera 中默认毫米，如果 opera 未修改单位，需要设为毫米 MM）
        unit_of_field 是磁场单位，默认特斯拉 T，即 1 T。如果是高斯，则输入 1e-4

        磁场文件格式：

        5 461 311 2
        1 X [METRE]
        2 Y [METRE]
        3 Z [METRE]
        4 BX [TESLA]
        5 BY [TESLA]
        6 BZ [TESLA]
        0
        -0.550000000000      -1.10000000000     -0.100000000000E-01  0.500216216871E-04  0.431668985488E-05 -0.180396818407E-02
        -0.550000000000      -1.10000000000     -0.500000000000E-02  0.422312886145E-04 -0.426162472137E-05 -0.180415005970E-02
        ... ...

        成员变量： xyz_0 gap_xyz number_of_points_xyz total_point_number
            table_position_data_xyz table_field_data_xyz
        """

        # 顶点
        self.x0 = first_corner_x*unit_of_length
        self.y0 = first_corner_y*unit_of_length
        self.z0 = first_corner_z*unit_of_length

        # 三方向步长
        self.gap_x = step_between_points_x*unit_of_length
        self.gap_y = step_between_points_y*unit_of_length
        self.gap_z = step_between_points_z*unit_of_length

        # 步数
        self.number_of_points_x = int(number_of_points_x)
        self.number_of_points_y = int(number_of_points_y)
        self.number_of_points_z = int(number_of_points_z)

        # 总点数
        self.total_point_number = self.number_of_points_x * \
            self.number_of_points_y*self.number_of_points_z

        print(f"给定参数的 table 文件包含 {self.total_point_number} 个节点")

        # 读取数据，使用 numpy
        data = numpy.loadtxt(fname=file_name, skiprows=8)  # 跳过 8 行，因为前 8 行非数据
        # 找出代表bx by bz 的列
        self.table_position_data_x = data[:, 0]
        self.table_position_data_y = data[:, 1]
        self.table_position_data_z = data[:, 2]

        self.table_field_data_x = data[:, 3]
        self.table_field_data_y = data[:, 4]
        self.table_field_data_z = data[:, 5]

        print(f"实际 table 文件包含 {len(data[:, 0])} 个节点")

        if not BaseUtils.equal(unit_of_field, 1):
            self.table_field_data_x = self.table_field_data_x * unit_of_field
            self.table_field_data_y = self.table_field_data_y * unit_of_field
            self.table_field_data_z = self.table_field_data_z * unit_of_field

    def magnetic_field_at(self, point: P3) -> P3:
        """
        核心方法，point 处的磁场，采用朗格朗日插值法计算
        """
        # 表格 table 上最近的磁场
        nearest_field_position = self.position_to_nearest_index(point)

        # x 為最後方向，研究 yz 平面.第二層，nearest_field_position 所在的那一層
        # 第 2 行，nearest_field_position 所在的那一行
        m1 = self.table_field_move(nearest_field_position, 0, 0, -1)
        m2 = nearest_field_position  # 不移動
        m3 = self.table_field_move(nearest_field_position, 0, 0, 1)
        m4 = self.table_field_move(nearest_field_position, 0, 0, 2)
        interpolate_point = P3(m1.position.x, m1.position.y, point.z)  # 插值点
        mm2 = OperaFieldTableMagnet.field_interpolate_lagrange(
            interpolate_point, m1, m2, m3, m4, OperaFieldTableMagnet.DIRECTION_Z
        )

        # 第 1 行，nearest_field_position 所在的那一行的下一行
        m1 = self.table_field_move(nearest_field_position, 0, -1, -1)
        m2 = self.table_field_move(nearest_field_position, 0, -1, 0)
        m3 = self.table_field_move(nearest_field_position, 0, -1, 1)
        m4 = self.table_field_move(nearest_field_position, 0, -1, 2)
        interpolate_point = P3(m1.position.x, m1.position.y, point.z)  # 插值点
        mm1 = OperaFieldTableMagnet.field_interpolate_lagrange(
            interpolate_point, m1, m2, m3, m4, OperaFieldTableMagnet.DIRECTION_Z
        )

        # 第 3 行，nearest_field_position 所在的那一行的上一行
        m1 = self.table_field_move(nearest_field_position, 0, 1, -1)
        m2 = self.table_field_move(nearest_field_position, 0, 1, 0)
        m3 = self.table_field_move(nearest_field_position, 0, 1, 1)
        m4 = self.table_field_move(nearest_field_position, 0, 1, 2)
        interpolate_point = P3(m1.position.x, m1.position.y, point.z)  # 插值点
        mm3 = OperaFieldTableMagnet.field_interpolate_lagrange(
            interpolate_point, m1, m2, m3, m4, OperaFieldTableMagnet.DIRECTION_Z
        )

        # 第 4 行，nearest_field_position 所在的那一行的上 2 行
        m1 = self.table_field_move(nearest_field_position, 0, 2, -1)
        m2 = self.table_field_move(nearest_field_position, 0, 2, 0)
        m3 = self.table_field_move(nearest_field_position, 0, 2, 1)
        m4 = self.table_field_move(nearest_field_position, 0, 2, 2)
        interpolate_point = P3(m1.position.x, m1.position.y, point.z)  # 插值点
        mm4 = OperaFieldTableMagnet.field_interpolate_lagrange(
            interpolate_point, m1, m2, m3, m4, OperaFieldTableMagnet.DIRECTION_Z
        )

        # 第二層的結合！y方向
        interpolate_point = P3(mm1.position.x, point.y, mm1.position.z)  # 插值点
        mmm2 = OperaFieldTableMagnet.field_interpolate_lagrange(
            interpolate_point, mm1, mm2, mm3, mm4, OperaFieldTableMagnet.DIRECTION_Y
        )

        # -------------------------------------------------- #

        # x為最後方向，研究yz平面.第1層，nearest_field_position所在的那一層的下一層
        # 第2行，nr所在的那一行
        m1 = self.table_field_move(nearest_field_position, -1, 0, -1)
        m2 = self.table_field_move(nearest_field_position, -1, 0, 0)
        m3 = self.table_field_move(nearest_field_position, -1, 0, 1)
        m4 = self.table_field_move(nearest_field_position, -1, 0, 2)
        interpolate_point = P3(m1.position.x, m1.position.y, point.z)  # 插值点
        mm2 = OperaFieldTableMagnet.field_interpolate_lagrange(
            interpolate_point, m1, m2, m3, m4, OperaFieldTableMagnet.DIRECTION_Z
        )

        # 第1行，nearest_field_position所在的那一行的下一行
        m1 = self.table_field_move(nearest_field_position, -1, -1, -1)
        m2 = self.table_field_move(nearest_field_position, -1, -1, 0)
        m3 = self.table_field_move(nearest_field_position, -1, -1, 1)
        m4 = self.table_field_move(nearest_field_position, -1, -1, 2)
        interpolate_point = P3(m1.position.x, m1.position.y, point.z)  # 插值点
        mm1 = OperaFieldTableMagnet.field_interpolate_lagrange(
            interpolate_point, m1, m2, m3, m4, OperaFieldTableMagnet.DIRECTION_Z
        )

        # 第3行，nearest_field_position 所在的那一行的上一行
        m1 = self.table_field_move(nearest_field_position, -1, 1, -1)
        m2 = self.table_field_move(nearest_field_position, -1, 1, 0)
        m3 = self.table_field_move(nearest_field_position, -1, 1, 1)
        m4 = self.table_field_move(nearest_field_position, -1, 1, 2)
        interpolate_point = P3(m1.position.x, m1.position.y, point.z)  # 插值点
        mm3 = OperaFieldTableMagnet.field_interpolate_lagrange(
            interpolate_point, m1, m2, m3, m4, OperaFieldTableMagnet.DIRECTION_Z
        )

        # 第4行，nearest_field_position 所在的那一行的上2行
        m1 = self.table_field_move(nearest_field_position, -1, 2, -1)
        m2 = self.table_field_move(nearest_field_position, -1, 2, 0)
        m3 = self.table_field_move(nearest_field_position, -1, 2, 1)
        m4 = self.table_field_move(nearest_field_position, -1, 2, 2)
        interpolate_point = P3(m1.position.x, m1.position.y, point.z)  # 插值点
        mm4 = OperaFieldTableMagnet.field_interpolate_lagrange(
            interpolate_point, m1, m2, m3, m4, OperaFieldTableMagnet.DIRECTION_Z
        )

        # 第1層的結合！y方向
        interpolate_point = P3(mm1.position.x, point.y, mm1.position.z)  # 插值点
        mmm1 = OperaFieldTableMagnet.field_interpolate_lagrange(
            interpolate_point, mm1, mm2, mm3, mm4, OperaFieldTableMagnet.DIRECTION_Y
        )

        # -------------------------------------------------- #

        # x為最後方向，研究yz平面.第3層，nearest_field_position所在的那一層的下一層
        # 第2行，nr所在的那一行
        m1 = self.table_field_move(nearest_field_position, 1, 0, -1)
        m2 = self.table_field_move(nearest_field_position, 1, 0, 0)
        m3 = self.table_field_move(nearest_field_position, 1, 0, 1)
        m4 = self.table_field_move(nearest_field_position, 1, 0, 2)
        interpolate_point = P3(m1.position.x, m1.position.y, point.z)  # 插值点
        mm2 = OperaFieldTableMagnet.field_interpolate_lagrange(
            interpolate_point, m1, m2, m3, m4, OperaFieldTableMagnet.DIRECTION_Z
        )

        # 第1行，nearest_field_position所在的那一行的下一行
        m1 = self.table_field_move(nearest_field_position, 1, -1, -1)
        m2 = self.table_field_move(nearest_field_position, 1, -1, 0)
        m3 = self.table_field_move(nearest_field_position, 1, -1, 1)
        m4 = self.table_field_move(nearest_field_position, 1, -1, 2)
        interpolate_point = P3(m1.position.x, m1.position.y, point.z)  # 插值点
        mm1 = OperaFieldTableMagnet.field_interpolate_lagrange(
            interpolate_point, m1, m2, m3, m4, OperaFieldTableMagnet.DIRECTION_Z
        )

        # 第3行，nearest_field_position 所在的那一行的上一行
        m1 = self.table_field_move(nearest_field_position, 1, 1, -1)
        m2 = self.table_field_move(nearest_field_position, 1, 1, 0)
        m3 = self.table_field_move(nearest_field_position, 1, 1, 1)
        m4 = self.table_field_move(nearest_field_position, 1, 1, 2)
        interpolate_point = P3(m1.position.x, m1.position.y, point.z)  # 插值点
        mm3 = OperaFieldTableMagnet.field_interpolate_lagrange(
            interpolate_point, m1, m2, m3, m4, OperaFieldTableMagnet.DIRECTION_Z
        )

        # 第4行，nearest_field_position 所在的那一行的上2行
        m1 = self.table_field_move(nearest_field_position, 1, 2, -1)
        m2 = self.table_field_move(nearest_field_position, 1, 2, 0)
        m3 = self.table_field_move(nearest_field_position, 1, 2, 1)
        m4 = self.table_field_move(nearest_field_position, 1, 2, 2)
        interpolate_point = P3(m1.position.x, m1.position.y, point.z)  # 插值点
        mm4 = OperaFieldTableMagnet.field_interpolate_lagrange(
            interpolate_point, m1, m2, m3, m4, OperaFieldTableMagnet.DIRECTION_Z
        )

        # 第3層的結合！y方向
        interpolate_point = P3(mm1.position.x, point.y, mm1.position.z)  # 插值点
        mmm3 = OperaFieldTableMagnet.field_interpolate_lagrange(
            interpolate_point, mm1, mm2, mm3, mm4, OperaFieldTableMagnet.DIRECTION_Y
        )

        # -------------------------------------------------- #

        # x為最後方向，研究yz平面.第4層，nearest_field_position所在的那一層的下一層
        # 第2行，nr所在的那一行
        m1 = self.table_field_move(nearest_field_position, 2, 0, -1)
        m2 = self.table_field_move(nearest_field_position, 2, 0, 0)
        m3 = self.table_field_move(nearest_field_position, 2, 0, 1)
        m4 = self.table_field_move(nearest_field_position, 2, 0, 2)
        interpolate_point = P3(m1.position.x, m1.position.y, point.z)  # 插值点
        mm2 = OperaFieldTableMagnet.field_interpolate_lagrange(
            interpolate_point, m1, m2, m3, m4, OperaFieldTableMagnet.DIRECTION_Z
        )

        # 第1行，nearest_field_position所在的那一行的下一行
        m1 = self.table_field_move(nearest_field_position, 2, -1, -1)
        m2 = self.table_field_move(nearest_field_position, 2, -1, 0)
        m3 = self.table_field_move(nearest_field_position, 2, -1, 1)
        m4 = self.table_field_move(nearest_field_position, 2, -1, 2)
        interpolate_point = P3(m1.position.x, m1.position.y, point.z)  # 插值点
        mm1 = OperaFieldTableMagnet.field_interpolate_lagrange(
            interpolate_point, m1, m2, m3, m4, OperaFieldTableMagnet.DIRECTION_Z
        )

        # 第3行，nearest_field_position 所在的那一行的上一行
        m1 = self.table_field_move(nearest_field_position, 2, 1, -1)
        m2 = self.table_field_move(nearest_field_position, 2, 1, 0)
        m3 = self.table_field_move(nearest_field_position, 2, 1, 1)
        m4 = self.table_field_move(nearest_field_position, 2, 1, 2)
        interpolate_point = P3(m1.position.x, m1.position.y, point.z)  # 插值点
        mm3 = OperaFieldTableMagnet.field_interpolate_lagrange(
            interpolate_point, m1, m2, m3, m4, OperaFieldTableMagnet.DIRECTION_Z
        )

        # 第4行，nearest_field_position 所在的那一行的上2行
        m1 = self.table_field_move(nearest_field_position, 2, 2, -1)
        m2 = self.table_field_move(nearest_field_position, 2, 2, 0)
        m3 = self.table_field_move(nearest_field_position, 2, 2, 1)
        m4 = self.table_field_move(nearest_field_position, 2, 2, 2)
        interpolate_point = P3(m1.position.x, m1.position.y, point.z)  # 插值点
        mm4 = OperaFieldTableMagnet.field_interpolate_lagrange(
            interpolate_point, m1, m2, m3, m4, OperaFieldTableMagnet.DIRECTION_Z
        )

        # 第3層的結合！y方向
        interpolate_point = P3(mm1.position.x, point.y, mm1.position.z)  # 插值点
        mmm4 = OperaFieldTableMagnet.field_interpolate_lagrange(
            interpolate_point, mm1, mm2, mm3, mm4, OperaFieldTableMagnet.DIRECTION_Y
        )

        # ------------------------------------------------------- #

        # 64元集合！！X方向
        interpolate_point = P3(point.x, mmm4.position.y,
                               mmm4.position.z)  # 插值点
        return OperaFieldTableMagnet.field_interpolate_lagrange(
            interpolate_point, mmm1, mmm2, mmm3, mmm4, OperaFieldTableMagnet.DIRECTION_X
        ).field

        # 毫無把握得把代碼轉移到這裏了
        # 純看天命

    # -------------------------  静态变量 ------------------------ #
    DIRECTION_X = 1  # 表示插值的方向
    DIRECTION_Y = 2
    DIRECTION_Z = 3

    @staticmethod
    def field_interpolate_lagrange(
        position: P3,
        field_postion0: FieldWIthPosion,
        field_postion1: FieldWIthPosion,
        field_postion2: FieldWIthPosion,
        field_postion3: FieldWIthPosion,
        direction: int
    ) -> FieldWIthPosion:
        """
        磁场的拉格朗日差值
        position 为要计算磁场的点
        field_postion0-3 为四个磁场点，每个磁场点包含磁场大小和点位置
        direction 确定差值方向

        position 和 field_postion0-3.position 位于同一条直线上
        当 direction == DIRECTION_X 时，直线于 x 轴平行
        当 direction == DIRECTION_Y 时，直线于 y 轴平行
        当 direction == DIRECTION_Z 时，直线于 z 轴平行
        """
        if direction == OperaFieldTableMagnet.DIRECTION_X:
            return FieldWIthPosion(
                position=position, field=P3(
                    BaseUtils.interpolate_lagrange(position.x,
                                                   field_postion0.position.x, field_postion0.field.x,
                                                   field_postion1.position.x, field_postion1.field.x,
                                                   field_postion2.position.x, field_postion2.field.x,
                                                   field_postion3.position.x, field_postion3.field.x),
                    BaseUtils.interpolate_lagrange(position.x,
                                                   field_postion0.position.x, field_postion0.field.y,
                                                   field_postion1.position.x, field_postion1.field.y,
                                                   field_postion2.position.x, field_postion2.field.y,
                                                   field_postion3.position.x, field_postion3.field.y),
                    BaseUtils.interpolate_lagrange(position.x,
                                                   field_postion0.position.x, field_postion0.field.z,
                                                   field_postion1.position.x, field_postion1.field.z,
                                                   field_postion2.position.x, field_postion2.field.z,
                                                   field_postion3.position.x, field_postion3.field.z),
                ))
        if direction == OperaFieldTableMagnet.DIRECTION_Y:
            return FieldWIthPosion(
                position=position, field=P3(
                    BaseUtils.interpolate_lagrange(position.y,
                                                   field_postion0.position.y, field_postion0.field.x,
                                                   field_postion1.position.y, field_postion1.field.x,
                                                   field_postion2.position.y, field_postion2.field.x,
                                                   field_postion3.position.y, field_postion3.field.x),
                    BaseUtils.interpolate_lagrange(position.y,
                                                   field_postion0.position.y, field_postion0.field.y,
                                                   field_postion1.position.y, field_postion1.field.y,
                                                   field_postion2.position.y, field_postion2.field.y,
                                                   field_postion3.position.y, field_postion3.field.y),
                    BaseUtils.interpolate_lagrange(position.y,
                                                   field_postion0.position.y, field_postion0.field.z,
                                                   field_postion1.position.y, field_postion1.field.z,
                                                   field_postion2.position.y, field_postion2.field.z,
                                                   field_postion3.position.y, field_postion3.field.z),
                ))
        if direction == OperaFieldTableMagnet.DIRECTION_Z:
            return FieldWIthPosion(
                position=position, field=P3(
                    BaseUtils.interpolate_lagrange(position.z,
                                                   field_postion0.position.z, field_postion0.field.x,
                                                   field_postion1.position.z, field_postion1.field.x,
                                                   field_postion2.position.z, field_postion2.field.x,
                                                   field_postion3.position.z, field_postion3.field.x),
                    BaseUtils.interpolate_lagrange(position.z,
                                                   field_postion0.position.z, field_postion0.field.y,
                                                   field_postion1.position.z, field_postion1.field.y,
                                                   field_postion2.position.z, field_postion2.field.y,
                                                   field_postion3.position.z, field_postion3.field.y),
                    BaseUtils.interpolate_lagrange(position.z,
                                                   field_postion0.position.z, field_postion0.field.z,
                                                   field_postion1.position.z, field_postion1.field.z,
                                                   field_postion2.position.z, field_postion2.field.z,
                                                   field_postion3.position.z, field_postion3.field.z),
                ))
        raise ValueError("direction只能是 OperaFieldTableMagnet.DIRECTION_X/Y/Z")

    def index_to_field_position(self, index: int) -> FieldWIthPosion:
        """
        获取 table 表上第 index 行对应的磁场，包括坐标
        """
        if index < 0 or index >= self.total_point_number:
            raise ValueError(f"index={index}，非法")

        field = P3(
            self.table_field_data_x[index],
            self.table_field_data_y[index],
            self.table_field_data_z[index],
        )

        # 下面的方法复杂
        # step_x = int(k/(self.number_of_points_y*self.number_of_points_z))
        # step_y = int((k - step_x * (self.number_of_points_y *
        #                             self.number_of_points_z)) / self.number_of_points_z)
        # step_z = int((k - step_x * (self.number_of_points_y *
        #                             self.number_of_points_z)) % self.number_of_points_z)

        # position = P3(
        #     self.x0 + self.gap_x * step_x,
        #     self.y0 + self.gap_y * step_y,
        #     self.z0 + self.gap_z * step_z,
        # )

        position_from_table = P3(
            self.table_position_data_x[index],
            self.table_position_data_y[index],
            self.table_position_data_z[index],
        )

        return FieldWIthPosion(position=position_from_table, field=field)

    def position_to_index(self, position: P3) -> int:
        """
        上述函数 index_to_field_position 的逆函数
        位置得到 table 编号
        """
        step_x = int(round((position.x-self.x0+1e-10)/self.gap_x))
        step_y = int(round((position.y-self.y0+1e-10)/self.gap_y))
        step_z = int(round((position.z-self.z0+1e-10)/self.gap_z))

        return (step_x*self.number_of_points_y*self.number_of_points_z +
                step_y*self.number_of_points_z +
                step_z)

    def position_to_nearest_index(self, position: P3) -> FieldWIthPosion:
        """
        任意一点，得到离它最近的已知磁场点
        """
        step_x = int(math.floor((position.x-self.x0+1e-10)/self.gap_x))
        step_y = int(math.floor((position.y-self.y0+1e-10)/self.gap_y))
        step_z = int(math.floor((position.z-self.z0+1e-10)/self.gap_z))

        return self.index_to_field_position(
            step_x*self.number_of_points_y*self.number_of_points_z +
            step_y*self.number_of_points_z +
            step_z
        )

    def table_field_move(self, field_position_in_table: FieldWIthPosion, x: int, y: int, z: int) -> FieldWIthPosion:
        """
        表格上的磁场点 field_position_in_table
        将其向 x y z 方向移动 x y z 格，得到新的磁场点
        """
        index = self.position_to_index(field_position_in_table.position)

        moved_index = (index +
                       x * self.number_of_points_y*self.number_of_points_z +
                       y * self.number_of_points_z +
                       z)

        if moved_index < 0 or moved_index >= self.total_point_number:
            print(f"移出边界")
            new_position = field_position_in_table.position + P3(
                self.gap_x * x, self.gap_y*y, self.gap_z*z
            )
            return FieldWIthPosion(position=new_position, field=field_position_in_table.field)

        return self.index_to_field_position(moved_index)
