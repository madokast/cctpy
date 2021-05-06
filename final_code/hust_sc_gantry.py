"""
CCT 建模优化代码
HUST 超导机架

作者：赵润晓
日期：2021年4月24日
"""

from cctpy import *  # 导入模块


class HUST_SC_GANTRY:
    def __init__(
        self,
        # ------------------ 前偏转段 ---------------#
        # 漂移段
        DL1=0.8001322,
        GAP1=0.1765959,
        GAP2=0.2960518,
        # qs 磁铁
        qs1_length=0.2997797,
        qs1_aperture_radius=30 * MM,
        qs1_gradient=28.33,
        qs1_second_gradient=-140.44 * 2.0,
        qs2_length=0.2585548,
        qs2_aperture_radius=30 * MM,
        qs2_gradient=-12.12,
        qs2_second_gradient=316.22 * 2.0,
        # cct 偏转半径
        cct12_big_r=0.95,
        # cct 孔径
        agcct12_inner_small_r=35 * MM,
        agcct12_outer_small_r=35 * MM + 15 * MM,
        dicct12_inner_small_r=35 * MM + 15 * MM * 2,
        dicct12_outer_small_r=35 * MM + 15 * MM * 3,
        # cct 匝数1
        agcct1_winding_number=30,
        agcct2_winding_number=39,
        dicct12_winding_number=71,
        # cct 角度
        dicct12_bending_angle=22.5,
        agcct1_bending_angle=9.782608695652174,
        agcct2_bending_angle=12.717391304347826,
        # cct 倾斜角（倾角 90 度表示不倾斜）
        dicct12_tilt_angles=[30, 80],
        agcct12_tilt_angles=[90, 30],
        # cct 电流
        dicct12_current=-6192,
        agcct12_current=-3319,
        # ------------------ 后偏转段 ---------------#
        # 漂移段
        DL2=2.1162209,
        GAP3=0.1978111,
        # qs 磁铁
        qs3_length=0.2382791,
        qs3_aperture_radius=60 * MM,
        qs3_gradient=-7.3733,
        qs3_second_gradient=-45.31 * 2,
        # cct 偏转半径
        cct345_big_r=0.95,
        # cct 孔径
        agcct345_inner_small_r=83 * MM,
        agcct345_outer_small_r=83 * MM + 15 * MM,
        dicct345_inner_small_r=83 * MM + 15 * MM * 2,
        dicct345_outer_small_r=83 * MM + 15 * MM * 3,
        # cct 匝数
        agcct3_winding_number=21,
        agcct4_winding_number=50,
        agcct5_winding_number=50,
        dicct345_winding_number=128,
        # cct 角度（负数表示顺时针偏转）
        dicct345_bending_angle=-67.5,
        agcct3_bending_angle=-(8 + 3.716404),
        agcct4_bending_angle=-(8 + 19.93897),
        agcct5_bending_angle=-(8 + 19.844626),
        # cct 倾斜角（倾角 90 度表示不倾斜）
        dicct345_tilt_angles=[30, 80],
        agcct345_tilt_angles=[90, 30],
        # cct 电流
        dicct345_current=9664,
        agcct345_current=-6000,

        part_per_winding=120,
    ) -> None:
        # ------------------ 前偏转段 ---------------#
        # 漂移段
        self.DL1 = DL1
        self.GAP1 = GAP1
        self.GAP2 = GAP2
        # qs 磁铁
        self.qs1_length = qs1_length
        self.qs1_aperture_radius = qs1_aperture_radius
        self.qs1_gradient = qs1_gradient
        self.qs1_second_gradient = qs1_second_gradient
        self.qs2_length = qs2_length
        self.qs2_aperture_radius = qs2_aperture_radius
        self.qs2_gradient = qs2_gradient
        self.qs2_second_gradient = qs2_second_gradient
        # cct 偏转半径
        self.cct12_big_r = cct12_big_r
        # cct 孔径
        self.agcct12_inner_small_r = agcct12_inner_small_r
        self.agcct12_outer_small_r = agcct12_outer_small_r
        self.dicct12_inner_small_r = dicct12_inner_small_r
        self.dicct12_outer_small_r = dicct12_outer_small_r
        # cct 匝数
        self.agcct1_winding_number = agcct1_winding_number
        self.agcct2_winding_number = agcct2_winding_number
        self.dicct12_winding_number = dicct12_winding_number
        # cct 角度
        self.dicct12_bending_angle = dicct12_bending_angle
        self.agcct1_bending_angle = agcct1_bending_angle
        self.agcct2_bending_angle = agcct2_bending_angle
        # cct 倾斜角（倾角 90 度表示不倾斜）
        self.dicct12_tilt_angles = dicct12_tilt_angles
        self.agcct12_tilt_angles = agcct12_tilt_angles
        # cct 电流
        self.dicct12_current = dicct12_current
        self.agcct12_current = agcct12_current
        # ------------------ 后偏转段 ---------------#
        # 漂移段
        self.DL2 = DL2
        self.GAP3 = GAP3
        # qs 磁铁
        self.qs3_length = qs3_length
        self.qs3_aperture_radius = qs3_aperture_radius
        self.qs3_gradient = qs3_gradient
        self.qs3_second_gradient = qs3_second_gradient
        # cct 偏转半径
        self.cct345_big_r = cct345_big_r
        # cct 孔径
        self.agcct345_inner_small_r = agcct345_inner_small_r
        self.agcct345_outer_small_r = agcct345_outer_small_r
        self.dicct345_inner_small_r = dicct345_inner_small_r
        self.dicct345_outer_small_r = dicct345_outer_small_r
        # cct 匝数
        self.agcct3_winding_number = agcct3_winding_number
        self.agcct4_winding_number = agcct4_winding_number
        self.agcct5_winding_number = agcct5_winding_number
        self.dicct345_winding_number = dicct345_winding_number
        # cct 角度（负数表示顺时针偏转）
        self.dicct345_bending_angle = dicct345_bending_angle
        self.agcct3_bending_angle = agcct3_bending_angle
        self.agcct4_bending_angle = agcct4_bending_angle
        self.agcct5_bending_angle = agcct5_bending_angle
        # cct 倾斜角（倾角 90 度表示不倾斜）
        self.dicct345_tilt_angles = dicct345_tilt_angles
        self.agcct345_tilt_angles = agcct345_tilt_angles
        # cct 电流
        self.dicct345_current = dicct345_current
        self.agcct345_current = agcct345_current

        self.part_per_winding = part_per_winding

        # -------- object  ---------
        self.__total_beamline: Beamline = None
        self.__first_bending_part_beamline: Beamline = None
        self.__second_bending_part_beamline: Beamline = None

    def create_first_bending_part_beamline(self) -> Beamline:
        """
        创建第一偏转段，并缓存
        """
        if self.__first_bending_part_beamline is None:
            self.__first_bending_part_beamline: Beamline = (
                Beamline.set_start_point(P2.origin())  # 设置束线的起点
                # 设置束线中第一个漂移段（束线必须以漂移段开始）
                .first_drift(direct=P2.x_direct(), length=self.DL1)
                .append_agcct(  # 尾接 acgcct
                    big_r=self.cct12_big_r,  # 偏转半径
                    # 二极 CCT 和四极 CCT 孔径
                    small_rs=[self.dicct12_outer_small_r, self.dicct12_inner_small_r,
                              self.agcct12_outer_small_r, self.agcct12_inner_small_r],
                    bending_angles=[self.agcct1_bending_angle,
                                    self.agcct2_bending_angle],  # agcct 每段偏转角度
                    tilt_angles=[self.dicct12_tilt_angles,
                                 self.agcct12_tilt_angles],  # 二极 CCT 和四极 CCT 倾斜角
                    winding_numbers=[[self.dicct12_winding_number], [
                        self.agcct1_winding_number, self.agcct2_winding_number]],  # 二极 CCT 和四极 CCT 匝数
                    # 二极 CCT 和四极 CCT 电流
                    currents=[self.dicct12_current, self.agcct12_current],
                    disperse_number_per_winding=self.part_per_winding  # 每匝分段数目
                )
                .append_drift(self.GAP1)  # 尾接漂移段
                .append_qs(  # 尾接 QS 磁铁
                    length=self.qs1_length,
                    gradient=self.qs1_gradient,
                    second_gradient=self.qs1_second_gradient,
                    aperture_radius=self.qs1_aperture_radius
                )
                .append_drift(self.GAP2)
                .append_qs(
                    length=self.qs2_length,
                    gradient=self.qs2_gradient,
                    second_gradient=self.qs2_second_gradient,
                    aperture_radius=self.qs2_aperture_radius
                )
                .append_drift(self.GAP2)
                .append_qs(
                    length=self.qs1_length,
                    gradient=self.qs1_gradient,
                    second_gradient=self.qs1_second_gradient,
                    aperture_radius=self.qs1_aperture_radius
                )
                .append_drift(self.GAP1)
                .append_agcct(
                    big_r=self.cct12_big_r,
                    small_rs=[self.dicct12_outer_small_r, self.dicct12_inner_small_r,
                              self.agcct12_outer_small_r, self.agcct12_inner_small_r],
                    bending_angles=[self.agcct2_bending_angle,
                                    self.agcct1_bending_angle],
                    tilt_angles=[self.dicct12_tilt_angles,
                                 self.agcct12_tilt_angles],
                    winding_numbers=[[self.dicct12_winding_number], [
                        self.agcct2_winding_number, self.agcct1_winding_number]],
                    currents=[self.dicct12_current, self.agcct12_current],
                    disperse_number_per_winding=self.part_per_winding
                )
                .append_drift(self.DL1)
            )

        return self.__first_bending_part_beamline

    def create_second_bending_part_beamline(self) -> Beamline:
        """
        创建第二偏转段，并缓存
        """
        if self.__first_bending_part_beamline is None:
            self.create_first_bending_part_beamline()

        if self.__first_bending_part_beamline is None:
            raise Exception(
                "HUST_SC_GANTRY出现异常，创建create_first_bending_part_beamline失败")

        if self.__second_bending_part_beamline is None:
            self.__second_bending_part_beamline: Beamline = (
                Beamline.set_start_point(
                    start_point=self.__first_bending_part_beamline.point_at_end())
                .first_drift(direct=self.__first_bending_part_beamline.direct_at_end(), length=self.DL2)
                .append_agcct(
                    big_r=self.cct345_big_r,
                    small_rs=[self.dicct345_outer_small_r, self.dicct345_inner_small_r,
                              self.agcct345_outer_small_r, self.agcct345_inner_small_r],
                    bending_angles=[self.agcct3_bending_angle,
                                    self.agcct4_bending_angle, self.agcct5_bending_angle],
                    tilt_angles=[self.dicct345_tilt_angles,
                                 self.agcct345_tilt_angles],
                    winding_numbers=[[self.dicct345_winding_number], [
                        self.agcct3_winding_number, self.agcct4_winding_number, self.agcct5_winding_number]],
                    currents=[self.dicct345_current, self.agcct345_current],
                    disperse_number_per_winding=self.part_per_winding
                )
                .append_drift(self.GAP3)
                .append_qs(
                    length=self.qs3_length,
                    gradient=self.qs3_gradient,
                    second_gradient=self.qs3_second_gradient,
                    aperture_radius=self.qs3_aperture_radius
                )
                .append_drift(self.GAP3)
                .append_agcct(
                    big_r=self.cct345_big_r,
                    small_rs=[self.dicct345_outer_small_r, self.dicct345_inner_small_r,
                              self.agcct345_outer_small_r, self.agcct345_inner_small_r],
                    bending_angles=[self.agcct5_bending_angle,
                                    self.agcct4_bending_angle, self.agcct3_bending_angle],
                    tilt_angles=[self.dicct345_tilt_angles,
                                 self.agcct345_tilt_angles],
                    winding_numbers=[[self.dicct345_winding_number], [
                        self.agcct5_winding_number, self.agcct4_winding_number, self.agcct3_winding_number]],
                    currents=[self.dicct345_current, self.agcct345_current],
                    disperse_number_per_winding=self.part_per_winding
                )
                .append_drift(self.DL2)
            )

        return self.__second_bending_part_beamline

    def create_total_beamline(self) -> Beamline:
        """
        创建整段机架，并缓存
        """
        if self.__total_beamline is None:
            self.__total_beamline: Beamline = (
                Beamline.set_start_point(P2.origin())  # 设置束线的起点
                # 设置束线中第一个漂移段（束线必须以漂移段开始）
                .first_drift(direct=P2.x_direct(), length=self.DL1)
                .append_agcct(  # 尾接 acgcct
                    big_r=self.cct12_big_r,  # 偏转半径
                    # 二极 CCT 和四极 CCT 孔径
                    small_rs=[self.dicct12_outer_small_r, self.dicct12_inner_small_r,
                              self.agcct12_outer_small_r, self.agcct12_inner_small_r],
                    bending_angles=[self.agcct1_bending_angle,
                                    self.agcct2_bending_angle],  # agcct 每段偏转角度
                    tilt_angles=[self.dicct12_tilt_angles,
                                 self.agcct12_tilt_angles],  # 二极 CCT 和四极 CCT 倾斜角
                    winding_numbers=[[self.dicct12_winding_number], [
                        self.agcct1_winding_number, self.agcct2_winding_number]],  # 二极 CCT 和四极 CCT 匝数
                    # 二极 CCT 和四极 CCT 电流
                    currents=[self.dicct12_current, self.agcct12_current],
                    disperse_number_per_winding=self.part_per_winding  # 每匝分段数目
                )
                .append_drift(self.GAP1)  # 尾接漂移段
                .append_qs(  # 尾接 QS 磁铁
                    length=self.qs1_length,
                    gradient=self.qs1_gradient,
                    second_gradient=self.qs1_second_gradient,
                    aperture_radius=self.qs1_aperture_radius
                )
                .append_drift(self.GAP2)
                .append_qs(
                    length=self.qs2_length,
                    gradient=self.qs2_gradient,
                    second_gradient=self.qs2_second_gradient,
                    aperture_radius=self.qs2_aperture_radius
                )
                .append_drift(self.GAP2)
                .append_qs(
                    length=self.qs1_length,
                    gradient=self.qs1_gradient,
                    second_gradient=self.qs1_second_gradient,
                    aperture_radius=self.qs1_aperture_radius
                )
                .append_drift(self.GAP1)
                .append_agcct(
                    big_r=self.cct12_big_r,
                    small_rs=[self.dicct12_outer_small_r, self.dicct12_inner_small_r,
                              self.agcct12_outer_small_r, self.agcct12_inner_small_r],
                    bending_angles=[self.agcct2_bending_angle,
                                    self.agcct1_bending_angle],
                    tilt_angles=[self.dicct12_tilt_angles,
                                 self.agcct12_tilt_angles],
                    winding_numbers=[[self.dicct12_winding_number], [
                        self.agcct2_winding_number, self.agcct1_winding_number]],
                    currents=[self.dicct12_current, self.agcct12_current],
                    disperse_number_per_winding=self.part_per_winding
                )
                .append_drift(self.DL1)
                
                # 第二段
                .append_drift(self.DL2)
                .append_agcct(
                    big_r=self.cct345_big_r,
                    small_rs=[self.dicct345_outer_small_r, self.dicct345_inner_small_r,
                              self.agcct345_outer_small_r, self.agcct345_inner_small_r],
                    bending_angles=[self.agcct3_bending_angle,
                                    self.agcct4_bending_angle, self.agcct5_bending_angle],
                    tilt_angles=[self.dicct345_tilt_angles,
                                 self.agcct345_tilt_angles],
                    winding_numbers=[[self.dicct345_winding_number], [
                        self.agcct3_winding_number, self.agcct4_winding_number, self.agcct5_winding_number]],
                    currents=[self.dicct345_current, self.agcct345_current],
                    disperse_number_per_winding=self.part_per_winding
                )
                .append_drift(self.GAP3)
                .append_qs(
                    length=self.qs3_length,
                    gradient=self.qs3_gradient,
                    second_gradient=self.qs3_second_gradient,
                    aperture_radius=self.qs3_aperture_radius
                )
                .append_drift(self.GAP3)
                .append_agcct(
                    big_r=self.cct345_big_r,
                    small_rs=[self.dicct345_outer_small_r, self.dicct345_inner_small_r,
                              self.agcct345_outer_small_r, self.agcct345_inner_small_r],
                    bending_angles=[self.agcct5_bending_angle,
                                    self.agcct4_bending_angle, self.agcct3_bending_angle],
                    tilt_angles=[self.dicct345_tilt_angles,
                                 self.agcct345_tilt_angles],
                    winding_numbers=[[self.dicct345_winding_number], [
                        self.agcct5_winding_number, self.agcct4_winding_number, self.agcct3_winding_number]],
                    currents=[self.dicct345_current, self.agcct345_current],
                    disperse_number_per_winding=self.part_per_winding
                )
                .append_drift(self.DL2)
            )
        
        return self.__total_beamline

if __name__ == "__main__":
    g = HUST_SC_GANTRY()
    f = g.create_first_bending_part_beamline()
    s = g.create_second_bending_part_beamline()
    t = g.create_total_beamline()

    Plot2.plot_beamline(t)
    Plot2.show()