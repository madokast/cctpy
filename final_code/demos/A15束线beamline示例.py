"""
CCT 建模优化代码
A15 束线 beamline 示例


作者：赵润晓
日期：2021年5月2日
"""

from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))

from cctpy import *

# Beamline 表示一段竖线，它由 Line2 和 多个 Magnet 组成
# 因此它可以看作一条二维有向曲线段，也可以看作一个磁铁

# 下面以一个超导机架为例，介绍 Beamline 的使用

if __name__ == "__main__":
    BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main()

    #------------------ 前偏转段 ---------------#
    # 漂移段，包含 DL1 GAP1 GAP2
    DL1 = 0.8001322
    GAP1 = 0.1765959
    GAP2 = 0.2960518
    # qs 磁铁，两个 qs 磁铁 qs1 qs2，分别包括长度、孔径、梯度、二阶梯度
    qs1_length = 0.2997797
    qs1_aperture_radius = 30 * MM
    qs1_gradient = 28.33
    qs1_second_gradient = -140.44 * 2.0
    qs2_length = 0.2585548
    qs2_aperture_radius = 30 * MM
    qs2_gradient = -12.12
    qs2_second_gradient = 316.22 * 2.0
    # cct 偏转半径
    cct12_big_r = 0.95
    # cct 孔径，四层 cct，所以存在四个值
    agcct12_inner_small_r = 35 * MM
    agcct12_outer_small_r = 35 * MM + 15 * MM
    dicct12_inner_small_r = 35 * MM + 15 * MM * 2
    dicct12_outer_small_r = 35 * MM + 15 * MM * 3
    # cct 匝数，包括两段交变梯度 agcct 的匝数和一段弯曲二极 dicct 的匝数
    agcct1_winding_number = 30
    agcct2_winding_number = 39
    dicct12_winding_number = 71
    # cct 角度（偏转角度）
    dicct12_bending_angle = 22.5
    agcct1_bending_angle = 9.782608695652174
    agcct2_bending_angle = 12.717391304347826 # agcct1_bending_angle + agcct2_bending_angle = dicct12_bending_angle
    # cct 倾斜角（倾角 90 度表示不倾斜）
    dicct12_tilt_angles = [30, 80]
    agcct12_tilt_angles = [90, 30]
    # cct 电流
    dicct12_current = -6192
    agcct12_current = -3319
    #------------------ 后偏转段 ---------------#
    # 漂移段
    DL2 = 2.1162209
    GAP3 = 0.1978111
    # qs 磁铁
    qs3_length = 0.2382791
    qs3_aperture_radius = 60 * MM
    qs3_gradient = -7.3733
    qs3_second_gradient = -45.31 * 2
    # cct 偏转半径
    cct345_big_r = 0.95
    # cct 孔径
    agcct345_inner_small_r = 83 * MM
    agcct345_outer_small_r = 83 * MM + 15 * MM
    dicct345_inner_small_r = 83 * MM + 15 * MM * 2
    dicct345_outer_small_r = 83 * MM + 15 * MM * 3
    # cct 匝数
    agcct3_winding_number = 21
    agcct4_winding_number = 50
    agcct5_winding_number = 50
    dicct345_winding_number = 128
    # cct 角度（负数表示顺时针偏转）
    dicct345_bending_angle = -67.5
    agcct3_bending_angle = -(8 + 3.716404)
    agcct4_bending_angle = -(8 + 19.93897)
    agcct5_bending_angle = -(8 + 19.844626)
    # cct 倾斜角（倾角 90 度表示不倾斜）
    dicct345_tilt_angles = [30, 80]
    agcct345_tilt_angles = [90, 30]
    # cct 电流
    dicct345_current = 9664
    agcct345_current = -6000

    # 一匝 cct 离散的电流元数目，设为 36 个
    part_per_winding = 36


    #------------------ 使用 Beamline 构建束线 ---------------#
    #------------------ 前偏转段 ---------------#
    beamline = (
        Beamline.set_start_point(P2.origin()) # 设置束线的起点
        .first_drift(direct=P2.x_direct(), length=DL1) # 设置束线中第一个漂移段（束线必须以漂移段开始）
        .append_agcct( # 尾接 acgcct
            big_r=cct12_big_r, # 偏转半径
            # 二极 CCT 和四极 CCT 孔径
            small_rs=[dicct12_outer_small_r,dicct12_inner_small_r,agcct12_outer_small_r,agcct12_inner_small_r],
            bending_angles=[agcct1_bending_angle,agcct2_bending_angle], # agcct 每段偏转角度
            tilt_angles=[dicct12_tilt_angles,agcct12_tilt_angles], # 二极 CCT 和四极 CCT 倾斜角
            winding_numbers=[[dicct12_winding_number],[agcct1_winding_number,agcct2_winding_number]], # 二极 CCT 和四极 CCT 匝数
            currents=[dicct12_current,agcct12_current], # 二极 CCT 和四极 CCT 电流
            disperse_number_per_winding=part_per_winding # 每匝分段数目
        )
        .append_drift(GAP1) # 尾接漂移段
        .append_qs(  # 尾接 QS 磁铁
            length=qs1_length,
            gradient=qs1_gradient,
            second_gradient=qs1_second_gradient,
            aperture_radius=qs1_aperture_radius
        )
        .append_drift(GAP2)
        .append_qs(
            length=qs2_length,
            gradient=qs2_gradient,
            second_gradient=qs2_second_gradient,
            aperture_radius=qs2_aperture_radius
        )
        .append_drift(GAP2)
        .append_qs(
            length=qs1_length,
            gradient=qs1_gradient,
            second_gradient=qs1_second_gradient,
            aperture_radius=qs1_aperture_radius
        )
        .append_drift(GAP1)
        .append_agcct(
            big_r=cct12_big_r,
            small_rs=[dicct12_outer_small_r,dicct12_inner_small_r,agcct12_outer_small_r,agcct12_inner_small_r],
            bending_angles=[agcct2_bending_angle,agcct1_bending_angle],
            tilt_angles=[dicct12_tilt_angles,agcct12_tilt_angles],
            winding_numbers=[[dicct12_winding_number],[agcct2_winding_number,agcct1_winding_number]],
            currents=[dicct12_current,agcct12_current],
            disperse_number_per_winding=part_per_winding
        )
        .append_drift(DL1)
    )

    # 束线长度
    beamline_length_part1 = beamline.get_length()
    print(f"前偏转段束线长度为{beamline_length_part1}m")

    # 绘制前偏转段图
    # Plot2.equal()
    # Plot2.plot(beamline)
    # Plot2.show()

    #------------------ 后偏转段 ---------------#
    beamline = (
        beamline.append_drift(DL2)
        .append_agcct(
                big_r=cct345_big_r,
                small_rs=[dicct345_outer_small_r,dicct345_inner_small_r,agcct345_outer_small_r,agcct345_inner_small_r],
                bending_angles=[agcct3_bending_angle,agcct4_bending_angle,agcct5_bending_angle],
                tilt_angles=[dicct345_tilt_angles,agcct345_tilt_angles],
                winding_numbers=[[dicct345_winding_number], [agcct3_winding_number,agcct4_winding_number,agcct5_winding_number]],
                currents=[dicct345_current,agcct345_current],
                disperse_number_per_winding=part_per_winding
        )
        .append_drift(GAP3)
        .append_qs(
            length=qs3_length,
            gradient=qs3_gradient,
            second_gradient=qs3_second_gradient,
            aperture_radius=qs3_aperture_radius
        )
        .append_drift(GAP3)
        .append_agcct(
                big_r=cct345_big_r,
                small_rs=[dicct345_outer_small_r,dicct345_inner_small_r,agcct345_outer_small_r,agcct345_inner_small_r],
                bending_angles=[agcct5_bending_angle,agcct4_bending_angle,agcct3_bending_angle],
                tilt_angles=[dicct345_tilt_angles,agcct345_tilt_angles],
                winding_numbers=[[dicct345_winding_number], [agcct5_winding_number,agcct4_winding_number,agcct3_winding_number]],
                currents=[dicct345_current,agcct345_current],
                disperse_number_per_winding=part_per_winding
        )
        .append_drift(DL2)
    )

    beamline_length = beamline.get_length()
    print(f"总束线长度为{beamline_length}m")

    # 绘制后偏转段图
    # Plot2.equal()
    # Plot2.plot(beamline)
    # Plot2.show()

    #------------------ 定义束流，并进行粒子跟踪，绘制相椭圆 ---------------#

    # 设置束流参数，并进行粒子跟踪
    # 返回一个长度 2 的元素，表示相空间 x-xp 平面和 y-yp 平面上粒子投影
    xxp,yyp = beamline.track_phase_ellipse(
        x_sigma_mm=3.5,
        xp_sigma_mrad=7.5,
        y_sigma_mm=3.5,
        yp_sigma_mrad=7.5,
        delta=0.0,
        particle_number=6, # 粒子数目
        kinetic_MeV=215,
        s=beamline_length_part1, # 束流起点，设为 beamline_length_part1，即后偏转段的起点
        footstep=20*MM, # 粒子运动步长
        concurrency_level=16
    )

    # Plot2.info(x_label='x/mm',y_label='xp/mrad',title='x-xp')
    # Plot2.plot(xxp,describe='bo')
    # Plot2.show()

    # Plot2.info(x_label='y/mm',y_label='yp/mrad',title='y-yp')
    # Plot2.plot(yyp,describe='bo')
    # Plot2.show()


    # 查看磁场分布
    b = beamline.magnetic_field_bz_along(step=20*MM)
    # Plot2.plot(b,describe='r-')
    # Plot2.show()

    # ------------------------------ 下面简单介绍 Beamline 的各个函数 --------------------- #

    # Beamline 是 Line2, Magnet, ApertureObject 三个类的子类
    # 因此这三个父类的函数都可以在 Beamline 对象中使用，如 magnetic_field_at()

    # Beamline 的直接构造函数 Beamline() 不推荐使用
    # 它传传入一个 Line2 对象当作设计轨道 trajectory
    # 构造后，轨道上不存在磁场

    # Magnet 相关的函数
    # magnetic_field_along
    # magnetic_field_bz_along
    # graident_field_along
    # second_graident_field_along

    # ApertureObject 相关的函数
    # is_out_of_aperture
    # trace_is_out_of_aperture 新增函数，用于确定一段轨迹 P3 数组，是否超出孔径

    # Line2 相关的函数
    # get_length
    # point_at
    # direct_at


    # 函数 track_ideal_particle 用于理想粒子的粒子跟踪
    # 参数如下
    # kinetic_MeV 粒子动能，单位 MeV
    # s 起点位置，以束线起点处 s 距离作为粒子起点。默认 0，即在束线的起点
    # length 运动路程。默认运动到束线尾部
    # footstep 粒子运动步长，默认 5*MM
    # 返回粒子运动轨迹，P3 数组
    track_ideal_p = beamline.track_ideal_particle(kinetic_MeV=215)
    # Plot3.plot_beamline(beamline)
    # Plot3.plot_p3s(track_ideal_p)
    # Plot3.show()


    # 函数 track_phase_ellipse() 束流跟踪，运行一个相椭圆
    # 返回一个长度 2 的元组
    # 元素元素分别表示相空间 x-xp 平面和 y-yp 平面上粒子投影（单位 mm / mrad），元素类型为 P2 数组
    # 参数如下：
    # x_sigma_mm σx 单位 mm
    # xp_sigma_mrad σxp 单位 mrad
    # y_sigma_mm σy 单位 mm
    # yp_sigma_mrad σyp 单位 mrad
    # delta 动量分散 单位 1
    # particle_number 粒子数目
    # kinetic_MeV 动能 单位 MeV
    # s 起点位置
    # length 粒子运行长度，默认运行到束线尾部
    # footstep 粒子运动步长
    # concurrency_level 并发等级（使用多少个核心进行粒子跟踪）
    # report 是否打印并行任务计划
    
    xxp,yyp = beamline.track_phase_ellipse(
        x_sigma_mm=3.5,
        xp_sigma_mrad=7.5,
        y_sigma_mm=3.5,
        yp_sigma_mrad=7.5,
        delta=0.0,
        particle_number=6, # 粒子数目
        kinetic_MeV=250,
        s = 0,
        length=beamline_length_part1,
        footstep=20*MM, # 粒子运动步长
        concurrency_level=16
    )

    # Plot2.plot_p2s(xxp,describe='r.')
    # Plot2.plot_p2s(yyp,describe='b.')
    # Plot2.show()

    # 内部类 __BeamlineBuilder 用于方便的构建 beamline
    # 构建 beamline 的步轴为
    # 第一步 指定起点
    beamline = Beamline.set_start_point(P2.origin())
    # 第二步 设定第一条偏移段，方向和长度
    beamline = beamline.first_drift(direct=P2.x_direct(),length=1)
    # 第三步 不断地尾解元件，可以是以下四种
    #   append_drift
    #   append_qs
    #   append_dipole_cct
    #   append_agcct
    # 下面一一介绍

    # 函数 append_drift 尾接一条漂移段
    # 参数只有一个 length，即漂移段的长度

    # 函数 append_qs 尾接一个 qs 磁铁
    # 参数为 
    # length: float QS 磁铁长度
    # gradient: float 梯度 T/m
    # second_gradient: float 二阶梯度（六极场） T/m^2
    # aperture_radius: float 半孔径 单位 m

    # 函数 append_dipole_cct 尾接一个 二极CCT 磁铁
    # 参数如下
    # big_r: float 偏转半径
    # small_r_inner: float 内层半孔径
    # small_r_outer: float 外层半孔径
    # bending_angle: float 偏转角度（正数表示逆时针、负数表示顺时针）
    # tilt_angles: List[float] 各极倾斜角
    # winding_number: int 匝数
    # current: float 电流
    # disperse_number_per_winding: int 每匝分段数目，越大计算越精确

    # 函数 append_agcct 尾接一个 四极CCT 磁铁
    # 参数为 
    # big_r: float 偏转半径，单位 m
    # small_rs: List[float] 各层 CCT 的孔径，一共四层，从大到小排列。分别是二极CCT外层、内层，四极CCT外层、内层
    # bending_angles: List[float] 交变四极 CCT 每个 part 的偏转半径（正数表示逆时针、负数表示顺时针），要么全正数，要么全负数。不需要传入二极 CCT 偏转半径，因为它就是 sum(bending_angles)
    # tilt_angles: List[List[float]] 二极 CCT 和四极 CCT 的倾斜角，典型值 [[30],[90,30]]，只有两个元素的二维数组
    # winding_numbers: List[List[int]], 二极 CCT 和四极 CCT 的匝数，典型值 [[128],[21,50,50]] 表示二极 CCT 128匝，四极交变 CCT 为 21、50、50 匝
    # currents: List[float] 二极 CCT 和四极 CCT 的电流，典型值 [8000,9000]
    # disperse_number_per_winding: int 每匝分段数目，越大计算越精确




    # 函数 __str__ 将 beamline 转为字符串
    print(beamline) # beamline(magnet_size=0, traj_len=1.0)
