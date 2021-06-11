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
        DL1=0.9007765,
        GAP1=0.4301517,
        GAP2=0.370816,
        # qs 磁铁
        qs1_length=0.2340128,
        qs1_aperture_radius=60 * MM,
        qs1_gradient=5.67,
        qs1_second_gradient=-127.78,
        qs2_length=0.200139,
        qs2_aperture_radius=60 * MM,
        qs2_gradient=12.83,
        qs2_second_gradient=72.22,
        # cct 偏转半径
        cct12_big_r=0.95,
        # cct 孔径
        agcct12_inner_small_r=72.5 * MM,
        agcct12_outer_small_r=88.5 * MM,
        dicct12_inner_small_r=104.5 * MM,
        dicct12_outer_small_r=120.5 * MM,
        # cct 匝数1
        agcct1_winding_number=22,
        agcct2_winding_number=23,
        dicct12_winding_number=42,
        # cct 角度
        dicct12_bending_angle=22.5,
        agcct1_bending_angle=11,
        agcct2_bending_angle=11.5,
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


def beamline_phase_ellipse_multi_delta(bl: Beamline, particle_number: int,
                                       dps: List[float], describles: str = ['r-', 'y-', 'b-', 'k-', 'g-', 'c-', 'm-'],
                                       foot_step: float = 20*MM, report: bool = True):
    if len(dps) > len(describles):
        print(
            f'describles(size={len(describles)}) 长度应大于等于 dps(size={len(dps)})')
    xs = []
    ys = []
    for dp in dps:
        x, y = bl.track_phase_ellipse(
            x_sigma_mm=3.5, xp_sigma_mrad=7.5,
            y_sigma_mm=3.5, yp_sigma_mrad=7.5,
            delta=dp, particle_number=particle_number,
            kinetic_MeV=215, concurrency_level=16,
            footstep=foot_step,
            report=report
        )
        xs.append(x + [x[0]])
        ys.append(y + [y[0]])

    plt.subplot(121)

    for i in range(len(dps)):
        plt.plot(*P2.extract(xs[i]), describles[i])
    plt.xlabel(xlabel='x/mm')
    plt.ylabel(ylabel='xp/mr')
    plt.title(label='x-plane')
    plt.legend(['dp'+str(int(dp*100)) for dp in dps])
    plt.axis("equal")

    plt.subplot(122)
    for i in range(len(dps)):
        plt.plot(*P2.extract(ys[i]), describles[i])
    plt.xlabel(xlabel='y/mm')
    plt.ylabel(ylabel='yp/mr')
    plt.title(label='y-plane')
    plt.legend(['dp'+str(int(dp*100)) for dp in dps])
    plt.axis("equal")

    plt.show()


if __name__ == "__main__":
    BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main()
    # param = [4.994543592,	40.04650003	,85.82762698,	96.91909089	,95.33845506,	97.55636171	,60.01158632,	83.74210641,	9244.758463,	-7241.295297]

    param = [4.373233845,	40	,85.34567767,	97.46179759,	95.92615864,	97.49058727,	60.08368362,	83.65814899,	9243.737555,	-7364.730324]

    qs3_g = param[0]
    qs3_sg = param[1]

    dicct_tilt_1 = param[2]
    dicct_tilt_2 = param[3]
    dicct_tilt_3 = param[4]

    agcct_tilt_0 = param[5]
    agcct_tilt_2 = param[6]
    agcct_tilt_3 = param[7]

    dicct_current = param[8]
    agcct_current = param[9]

    agcct3_wn = 25
    agcct4_wn = 40
    agcct5_wn = 34

    g = HUST_SC_GANTRY(
        qs3_gradient=qs3_g,
        qs3_second_gradient=qs3_sg,
        dicct345_tilt_angles=[30, dicct_tilt_1, dicct_tilt_2, dicct_tilt_3],
        agcct345_tilt_angles=[agcct_tilt_0, 30, agcct_tilt_2, agcct_tilt_3],
        dicct345_current=dicct_current,
        agcct345_current=agcct_current,
        agcct3_winding_number=agcct3_wn,
        agcct4_winding_number=agcct4_wn,
        agcct5_winding_number=agcct5_wn,
        agcct3_bending_angle=-67.5 * (agcct3_wn / (agcct3_wn + agcct4_wn + agcct5_wn)),
        agcct4_bending_angle=-67.5 * (agcct4_wn / (agcct3_wn + agcct4_wn + agcct5_wn)),
        agcct5_bending_angle=-67.5 * (agcct5_wn / (agcct3_wn + agcct4_wn + agcct5_wn)),

        DL1=0.9007765,
        GAP1=0.4301517,
        GAP2=0.370816,
        qs1_length=0.2340128,
        qs1_aperture_radius=60 * MM,
        qs1_gradient=0.0,
        qs1_second_gradient=0.0,
        qs2_length=0.200139,
        qs2_aperture_radius=60 * MM,
        qs2_gradient=0.0,
        qs2_second_gradient=0.0,

        DL2=2.35011,
        GAP3=0.43188,
        qs3_length=0.24379,

        agcct345_inner_small_r=92.5 * MM + 17.1 * MM,# 92.5
        agcct345_outer_small_r=108.5 * MM + 17.1 * MM,  # 83+15
        dicct345_inner_small_r=124.5 * MM + 17.1 * MM,  # 83+30+1
        dicct345_outer_small_r=140.5 * MM + 17.1 * MM,  # 83+45 +2
    )
    # f = g.create_first_bending_part_beamline()
    s = g.create_second_bending_part_beamline()
    # t = g.create_total_beamline()



    beamline_phase_ellipse_multi_delta(
        s,8,[-0.05,0,0.05]
    )

    # Plot2.plot_beamline(t)
    # Plot2.show()