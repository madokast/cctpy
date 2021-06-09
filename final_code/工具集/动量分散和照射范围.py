"""
CCT 建模优化代码
工具集

作者：赵润晓
日期：2021年6月7日
"""


import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)

from cctpy import *
import numpy as np


print(Protons.get_kinetic_energy_MeV_after_momentum_dispersion(200,-0.15))
print(Protons.get_kinetic_energy_MeV_after_momentum_dispersion(200,0.15))

dp_max = 0.05
dp_min = -0.05

# 动能
kinetic_energy_MeV = BaseUtils.linspace(1,250,250)
# 动能上限
kinetic_energy_MeV_max = [Protons.get_kinetic_energy_MeV_after_momentum_dispersion(k,dp_max) for k in kinetic_energy_MeV]
kinetic_energy_MeV_min = [Protons.get_kinetic_energy_MeV_after_momentum_dispersion(k,dp_min) for k in kinetic_energy_MeV]

# 照射深度
ranges = [0.0022*k**1.77 for k in kinetic_energy_MeV]
ranges_max = [0.0022*k**1.77 for k in kinetic_energy_MeV_max]
ranges_min = [0.0022*k**1.77 for k in kinetic_energy_MeV_min]

if False: # 绘制中心能量下对应的水深
    Plot2.plot_xy_array(kinetic_energy_MeV,ranges,describe='k-')
    Plot2.plot_xy_array(kinetic_energy_MeV,ranges_max,describe='k--')
    Plot2.plot_xy_array(kinetic_energy_MeV,ranges_min,describe='k--')

    dp_max = 0.10
    dp_min = -0.10

    # 动能
    kinetic_energy_MeV = BaseUtils.linspace(1,250,250)
    # 动能上限
    kinetic_energy_MeV_max = [Protons.get_kinetic_energy_MeV_after_momentum_dispersion(k,dp_max) for k in kinetic_energy_MeV]
    kinetic_energy_MeV_min = [Protons.get_kinetic_energy_MeV_after_momentum_dispersion(k,dp_min) for k in kinetic_energy_MeV]

    # 照射深度
    ranges = [0.0022*k**1.77 for k in kinetic_energy_MeV]
    ranges_max = [0.0022*k**1.77 for k in kinetic_energy_MeV_max]
    ranges_min = [0.0022*k**1.77 for k in kinetic_energy_MeV_min]
    Plot2.plot_xy_array(kinetic_energy_MeV,ranges_max,describe='k--')
    Plot2.plot_xy_array(kinetic_energy_MeV,ranges_min,describe='k--')
    
    dp_max = 0.15
    dp_min = -0.15

    # 动能
    kinetic_energy_MeV = BaseUtils.linspace(1,250,250)
    # 动能上限
    kinetic_energy_MeV_max = [Protons.get_kinetic_energy_MeV_after_momentum_dispersion(k,dp_max) for k in kinetic_energy_MeV]
    kinetic_energy_MeV_min = [Protons.get_kinetic_energy_MeV_after_momentum_dispersion(k,dp_min) for k in kinetic_energy_MeV]

    # 照射深度
    ranges = [0.0022*k**1.77 for k in kinetic_energy_MeV]
    ranges_max = [0.0022*k**1.77 for k in kinetic_energy_MeV_max]
    ranges_min = [0.0022*k**1.77 for k in kinetic_energy_MeV_min]
    Plot2.plot_xy_array(kinetic_energy_MeV,ranges_max,describe='k--')
    Plot2.plot_xy_array(kinetic_energy_MeV,ranges_min,describe='k--')

    Plot2.info("能量/MeV","等效水深/cm","",25,font_family="SimHei")

    Plot2.xlim(0,300)

    Plot2.show()

if True: # 绘制水深和能扩展的布拉格峰
    Plot2.plot_xy_array(ranges,[ranges_max[i]-ranges_min[i] for i in range(len(ranges_max))],describe='k-')
    print(BaseUtils.polynomial_fitting(ranges,[ranges_max[i]-ranges_min[i] for i in range(len(ranges_max))],1))

    dp_max = 0.10
    dp_min = -0.10

    # 动能
    kinetic_energy_MeV = BaseUtils.linspace(1,250,250)
    # 动能上限
    kinetic_energy_MeV_max = [Protons.get_kinetic_energy_MeV_after_momentum_dispersion(k,dp_max) for k in kinetic_energy_MeV]
    kinetic_energy_MeV_min = [Protons.get_kinetic_energy_MeV_after_momentum_dispersion(k,dp_min) for k in kinetic_energy_MeV]

    # 照射深度
    ranges = [0.0022*k**1.77 for k in kinetic_energy_MeV]
    ranges_max = [0.0022*k**1.77 for k in kinetic_energy_MeV_max]
    ranges_min = [0.0022*k**1.77 for k in kinetic_energy_MeV_min]

    Plot2.plot_xy_array(ranges,[ranges_max[i]-ranges_min[i] for i in range(len(ranges_max))],describe='k-')
    print(BaseUtils.polynomial_fitting(ranges,[ranges_max[i]-ranges_min[i] for i in range(len(ranges_max))],1))


    dp_max = 0.15
    dp_min = -0.15

    # 动能
    kinetic_energy_MeV = BaseUtils.linspace(1,250,250)
    # 动能上限
    kinetic_energy_MeV_max = [Protons.get_kinetic_energy_MeV_after_momentum_dispersion(k,dp_max) for k in kinetic_energy_MeV]
    kinetic_energy_MeV_min = [Protons.get_kinetic_energy_MeV_after_momentum_dispersion(k,dp_min) for k in kinetic_energy_MeV]

    # 照射深度
    ranges = [0.0022*k**1.77 for k in kinetic_energy_MeV]
    ranges_max = [0.0022*k**1.77 for k in kinetic_energy_MeV_max]
    ranges_min = [0.0022*k**1.77 for k in kinetic_energy_MeV_min]
    Plot2.plot_xy_array(ranges,[ranges_max[i]-ranges_min[i] for i in range(len(ranges_max))],describe='k-')
    print(BaseUtils.polynomial_fitting(ranges,[ranges_max[i]-ranges_min[i] for i in range(len(ranges_max))],1))


    Plot2.info("中心深度/cm","最大照射范围/cm","",25,font_family="SimHei")
    Plot2.xlim(0,50)

    Plot2.show()

