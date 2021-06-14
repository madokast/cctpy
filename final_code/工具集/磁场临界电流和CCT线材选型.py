"""
CCT 建模优化代码
工具集

作者：赵润晓
日期：2021年6月12日
"""

# 超导线材临界电流关键点
magnet_field_critical_points = [6, 7, 8, 9]
current_critical_points = [795, 620, 445, 275]

# cct 单线电流和表面最大磁场
current = 444 # 588
max_magnet_field = 4.39 # 4.00

# ---------------------------------------------------------------------------------------- #

import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)

from cctpy import *

# 临界电流关键点拟合
coefficient_list = BaseUtils.polynomial_fitting(
    xs = magnet_field_critical_points,
    ys = current_critical_points,
    order = 2
)
fitted_func = BaseUtils.polynomial_fitted_function(coefficient_list)

Plot2.plot_function(fitted_func,2,10,describe="g--")
Plot2.plot_xy_array(magnet_field_critical_points,current_critical_points,"kx")
Plot2.plot_p2s([P2.origin(),P2(max_magnet_field,current).change_length(100000)],"r")
Plot2.plot_p2(P2(max_magnet_field,current),"ro")
Plot2.xlim(0,10)
Plot2.ylim(0,1200)
Plot2.info("B/T","I/A")
Plot2.show()