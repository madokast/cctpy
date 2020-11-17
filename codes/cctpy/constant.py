"""
常量
"""

from numpy import zeros, ndarray, array, float64, sqrt

# 单位 米，默认的长度单位
M: float = 1.

# 单位 毫米
MM: float = 0.001 * M

RAD: float = 1.

MRAD: float = 0.001 * RAD

# 光速
LIGHT_SPEED: float = 299792458.0 * M

# 注意以下 ndarray 量，如果需要原地改变，必须复制一下

# 三维坐标系原点
ORIGIN3: ndarray = zeros((3,))

# 三维零矢量
ZERO3: ndarray = ORIGIN3

# 坐标系各轴单位向量
XI: ndarray = array([1, 0, 0], dtype=float64)
YI: ndarray = array([0, 1, 0], dtype=float64)
ZI: ndarray = array([0, 0, 1], dtype=float64)

# 焦耳 能力单位
J: float = 1.0

# 电子伏特
eV = 1.6021766208e-19 * J
MeV = 1000 * 1000 * eV

# 动量单位 1 MeV/c = 5.3442857792E-22 kg m/s
MeV_PER_C = 5.3442857792E-22


class Protons:
    """
    质子相关常量和计算
    """

    # 静止质量
    STATIC_MASS_KG = 1.672621898e-27

    # 静止能量 = m0 * c ^ 2 单位焦耳
    STATIC_ENERGY_J = STATIC_MASS_KG * LIGHT_SPEED * LIGHT_SPEED

    # 静止能量 eV 为单位
    STATIC_ENERGY_eV = STATIC_ENERGY_J / eV

    # 静止能量 MeV 为单位，应该是 STATIC_ENERGY_J / MeV。但是写成字面量
    STATIC_ENERGY_MeV = 938.2720813

    # 电荷量 库伦
    CHARGE_QUANTITY = 1.6021766208e-19

    @classmethod
    def get_total_energy_MeV(cls, kinetic_energy_MeV: float) -> float:
        """
        计算总能量 MeV = 静止能量 + 动能
        Parameters
        ----------
        kinetic_energy_MeV 动能 MeV 一般为 250 Mev

        Returns 总能量 MeV
        -------

        """
        return cls.STATIC_ENERGY_MeV + kinetic_energy_MeV

    @classmethod
    def get_total_energy_J(cls, kinetic_energy_MeV: float) -> float:
        """
        计算总能量 焦耳
        Parameters
        ----------
        kinetic_energy_MeV 动能 MeV 一般为 250 Mev

        Returns 总能量 焦耳
        -------

        """
        return cls.get_total_energy_MeV(kinetic_energy_MeV) * MeV

    @classmethod
    def get_relativistic_mass(cls, kinetic_energy_MeV: float) -> float:
        """
        计算动质量 kg = 动能 / (c^2)
        Parameters
        ----------
        kinetic_energy_MeV 动能 MeV 一般为 250 Mev

        Returns 动质量 kg
        -------

        """
        return cls.get_total_energy_J(kinetic_energy_MeV) / LIGHT_SPEED / LIGHT_SPEED

    @classmethod
    def get_speed_m_per_s(cls, kinetic_energy_MeV: float) -> float:
        """
        计算速度 m/s = c * sqrt( 1 - (m0/m)^2 )
        Parameters
        ----------
        kinetic_energy_MeV 动能 MeV 一般为 250 Mev

        Returns 速度 m/s
        -------

        """
        return LIGHT_SPEED * sqrt(
            1 - (cls.STATIC_MASS_KG / cls.get_relativistic_mass(kinetic_energy_MeV)) ** 2
        )

    @classmethod
    def get_momentum_kg_m_pre_s(cls, kinetic_energy_MeV: float) -> float:
        """
        动量 kg m/s
        Parameters
        ----------
        kinetic_energy_MeV 动能 MeV 一般为 250 Mev

        Returns 动量 kg m/s
        -------

        """
        return cls.get_relativistic_mass(kinetic_energy_MeV) * cls.get_speed_m_per_s(kinetic_energy_MeV)

    @classmethod
    def getMomentum_MeV_pre_c(cls, kinetic_energy_MeV: float) -> float:
        """
        动量 MeV/c
        Parameters 动能 MeV 一般为 250 Mev
        ----------
        kinetic_energy_MeV

        Returns 动量 MeV/c
        -------

        """
        return cls.get_momentum_kg_m_pre_s(kinetic_energy_MeV) / MeV_PER_C

    @classmethod
    def get_magnetic_stiffness(cls, kinetic_energy_MeV: float) -> float:
        """
        磁钢度 T/m
        Parameters
        ----------
        kinetic_energy_MeV 动能 MeV 一般为 250 Mev

        Returns 磁钢度 T/m
        -------

        """
        return cls.get_momentum_kg_m_pre_s(kinetic_energy_MeV) / cls.CHARGE_QUANTITY

    # ------------------  动量分散相关  ----------------------
    @classmethod
    def get_kinetic_energy_MeV(cls, momentum_KG_M_PER_S: float) -> float:
        # 速度
        speed = momentum_KG_M_PER_S / sqrt(cls.STATIC_MASS_KG ** 2 + (momentum_KG_M_PER_S / LIGHT_SPEED) ** 2)
        # 动质量
        relativistic_mass = cls.STATIC_MASS_KG / sqrt(1 - (speed / LIGHT_SPEED) ** 2)
        # 总能量 J
        total_energy_J = relativistic_mass * LIGHT_SPEED * LIGHT_SPEED
        # 动能 J
        k = total_energy_J - cls.STATIC_ENERGY_J

        return k / MeV

    @classmethod
    def get动量分散后的动能(cls, 原动能_MeV: float, 动量分散: float):
        """
        英文版见下
        Parameters
        ----------
        原动能_MeV
        动量分散

        Returns 动量分散后的动能 MeV
        -------

        """
        原动量 = cls.get_momentum_kg_m_pre_s(原动能_MeV)

        新动量 = 原动量 * (1 + 动量分散)

        新动能 = cls.get_kinetic_energy_MeV(新动量)

        return 新动能

    @classmethod
    def get_kinetic_energy_MeV_after_momentum_dispersion(cls, old_kinetic_energy_MeV: float,
                                                         momentum_dispersion: float) -> float:
        """
        中文版见上
        Parameters
        ----------
        old_kinetic_energy_MeV 原动能_MeV
        momentum_dispersion 动量分散

        Returns 动量分散后的动能 MeV
        -------

        """
        momentum0 = cls.get_momentum_kg_m_pre_s(old_kinetic_energy_MeV)

        momentum = momentum0 * (1 + momentum_dispersion)

        kinetic_energy = cls.get_kinetic_energy_MeV(momentum)

        return kinetic_energy

    @classmethod
    def convert动量分散_TO_能量分散(cls, 动量分散: float, 动能_MeV: float) -> float:
        """
        下方法的中文版
        Parameters
        ----------
        动量分散
        动能_MeV

        Returns convert动量分散_TO_能量分散
        -------

        """
        k = (动能_MeV + cls.STATIC_ENERGY_MeV) / (动能_MeV + 2 * cls.STATIC_ENERGY_MeV)

        return 动量分散 / k

    @classmethod
    def convert_momentum_dispersion_to_energy_dispersion(cls, momentum_dispersion: float,
                                                         kinetic_energy_MeV: float) -> float:
        """
        上方法的英文版
        Parameters
        ----------
        momentum_dispersion 动量分散
        kinetic_energy_MeV 动能_MeV

        Returns convert动量分散_TO_能量分散
        -------

        """
        k = (kinetic_energy_MeV + cls.STATIC_ENERGY_MeV) / (kinetic_energy_MeV + 2 * cls.STATIC_ENERGY_MeV)

        return momentum_dispersion / k

    @classmethod
    def convert能量分散_TO_动量分散(cls, 能量分散: float, 动能_MeV: float) -> float:
        k = (动能_MeV + cls.STATIC_ENERGY_MeV) / (动能_MeV + 2 * cls.STATIC_ENERGY_MeV)
        return 能量分散 * k

    @classmethod
    def convert_energy_dispersion_to_momentum_dispersion(cls, energyDispersion: float,
                                                         kineticEnergy_MeV: float) -> float:
        """
        上方法的英文版
        Parameters
        ----------
        energyDispersion 能量分散
        kineticEnergy_MeV 动能，典型值 250

        Returns 动量分散
        -------

        """
        k = (kineticEnergy_MeV + cls.STATIC_ENERGY_MeV) / (kineticEnergy_MeV + 2 * cls.STATIC_ENERGY_MeV)
        return energyDispersion * k
