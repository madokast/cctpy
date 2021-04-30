"""
CCT 建模优化代码
A07质子相关常量和计算 Protons 示例

作者：赵润晓
日期：2021年4月29日
"""

from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

# Protons 类包含了与质子相关的很多常量和计算

print("质子静止质量，单位kg",Protons.STATIC_MASS_KG) # 1.672621898e-27
print("质子静止能量，单位J",Protons.STATIC_ENERGY_J) # 1.5032775928961053e-10
print("质子静止能量，单位eV",Protons.STATIC_ENERGY_eV) # 938272081.4796858
print("质子静止能量，单位MeV",Protons.STATIC_ENERGY_MeV) # 938.2720813
print("质子电荷量，单位C",Protons.CHARGE_QUANTITY) # 1.6021766208e-19

# 函数 get_total_energy_MeV(kinetic_energy_MeV)，计算质子总能量
# 传入动能 kinetic_energy_MeV
# 单位都是 MeV
print("计算质子总能量 MeV，动能 250 MeV",Protons.get_total_energy_MeV(250)) # 1188.2720813

# 函数 get_total_energy_J(kinetic_energy_MeV)，计算质子总能量
# 传入动能 kinetic_energy_MeV，单位 MeV
# 返回值单位 焦耳
print("计算质子总能量 J，动能 250 MeV",Protons.get_total_energy_J(250)) # 1.903821747808217e-10

# 函数 get_relativistic_mass(kinetic_energy_MeV)，计算质子计算动质量
# 传入动能 kinetic_energy_MeV，单位 MeV
# 返回值单位 Kg
print("计算质子计算动质量 Kg，动能 250 MeV",Protons.get_relativistic_mass(250)) # 2.1182873744149107e-27

# 函数 get_speed_m_per_s(kinetic_energy_MeV)，计算质子速度
# 传入动能 kinetic_energy_MeV，单位 MeV
# 返回值单位 m/s
print("计算质子速度 m/s，动能 250 MeV",Protons.get_speed_m_per_s(250)) # 183955177.96913892

# 函数 get_momentum_kg_m_pre_s(kinetic_energy_MeV)，计算质子动量 kg m/s
# 传入动能 kinetic_energy_MeV，单位 MeV
# 返回值单位 kg m/s
print("计算质子动量 kg m/s，动能 250 MeV",Protons.get_momentum_kg_m_pre_s(250)) # 3.896699309502749e-19

# 函数 getMomentum_MeV_pre_c(kinetic_energy_MeV)，计算质子动量 MeV/c
# 传入动能 kinetic_energy_MeV，单位 MeV
# 返回值单位 MeV/c
print("计算质子动量 MeV/c，动能 250 MeV",Protons.getMomentum_MeV_pre_c(250)) # 729.1337833520677

# 函数 get_magnetic_stiffness(kinetic_energy_MeV)，计算质子磁钢度 T/m
# 传入动能 kinetic_energy_MeV，单位 MeV
# 返回值单位 T/m
print("计算质子磁钢度 T/m，动能 250 MeV",Protons.get_magnetic_stiffness(250)) # 2.4321284301084396

# 函数 get_kinetic_energy_MeV(momentum_KG_M_PER_S)，质子动量 kg m/s 转动能 MeV
# 传入动量 momentum_KG_M_PER_S kg m/s
# 返回值单位 MeV
print("质子动量 kg m/s 转动能 MeV，动量 3.896699309502749e-19 kg m/s",Protons.get_kinetic_energy_MeV(3.896699309502749e-19)) # 249.99999982031446

# 函数 get_kinetic_energy_MeV_after_momentum_dispersion(old_kinetic_energy_MeV,momentum_dispersion)
# 计算动量分散后的动能 MeV
# 参数：
# old_kinetic_energy_MeV 原动能_MeV
# momentum_dispersion 动量分散 0~1
print("计算动量分散后的动能 MeV，原动能 250 MeV，动量分散 -20%",Protons.get_kinetic_energy_MeV_after_momentum_dispersion(250,-0.2)) # 166.53630221606724

# 函数 convert_momentum_dispersion_to_energy_dispersion(momentum_dispersion,kinetic_energy_MeV)
# 将动量分散转为能量分散
# 参数
# momentum_dispersion 动量分散
# kinetic_energy_MeV 动能_MeV
print("将 20% 动量分散转为能量分散，中心动能为250 MeV",Protons.convert_momentum_dispersion_to_energy_dispersion(0.2,250)) # 0.3579220947905309

# 函数 convert_energy_dispersion_to_momentum_dispersion(energyDispersion,kineticEnergy_MeV)
# 将能量分散转为动量分散
# 参数
# energyDispersion 能量分散
# kinetic_energy_MeV 动能_MeV
print("将0.3579220947905309能量分散转为动量分散，中心动能为250 MeV",Protons.convert_energy_dispersion_to_momentum_dispersion(0.3579220947905309,250)) # 0.2

