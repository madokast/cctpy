"""
CCT 建模优化代码
COSY 扩展代码

作者：赵润晓
日期：2021年6月3日
"""

from packages.line3s import Line3
from packages.base_utils import BaseUtils
from packages.point import ValueWithDistance
from packages.constants import M, MM
from packages.line2s import Line2
from packages.magnets import Magnet
from typing import List, Optional, Union
from packages.particles import PhaseSpaceParticle


class CosyMap:
    """
    用于分析 cosy 输出的任意阶 map
    使用方法可见本文 if __name__ == "__main__": 后代码
    """
    # COSY map 输出时每项长度
    ITEM_LENGTH = len("-0.0000000E+00")

    def __init__(self, map: str):
        """
        初始化一个 map
        使用方法见 if __name__ == "__main__":
        """
        self.map = map
        self.contributionLinesString = map.split('\n')

    def apply(self, p0: PhaseSpaceParticle, order: int = 1, print_detail: bool = False, file=None) -> PhaseSpaceParticle:
        """
        map 作用于相空间粒子 p0，返回作用后的结果
        order 表示需要考虑的阶数，注意应不大于构造 CosyMap 时传入的阶数
        print_detail 表示打印每个矩阵项的详细贡献，默认不打印
        """
        x = 0.0
        xp = 0.0
        y = 0.0
        yp = 0.0

        for contributionLineString in self.contributionLinesString:
            x_contribution_string = contributionLineString[0:CosyMap.ITEM_LENGTH + 1]
            xp_contribution_string = contributionLineString[CosyMap.ITEM_LENGTH +
                                                            1: CosyMap.ITEM_LENGTH * 2 + 1]
            y_contribution_string = contributionLineString[CosyMap.ITEM_LENGTH *
                                                           2 + 1: CosyMap.ITEM_LENGTH * 3 + 1]
            yp_contribution_string = contributionLineString[CosyMap.ITEM_LENGTH *
                                                            3 + 1: CosyMap.ITEM_LENGTH * 4 + 1]

            contribution_describing = contributionLineString[CosyMap.ITEM_LENGTH * 5 + 2:]

            this_order = self.__order(contribution_describing)

            if this_order > order:
                break

            contributionBy = self.__contribution_by(
                contribution_describing, p0)

            xContribution = float(x_contribution_string)
            x += xContribution * contributionBy

            xpContribution = float(xp_contribution_string)
            xp += xpContribution * contributionBy

            yContribution = float(y_contribution_string)
            y += yContribution * contributionBy

            ypContribution = float(yp_contribution_string)
            yp += ypContribution * contributionBy

            if print_detail:
                if file is None:
                    print(f"{contribution_describing} {xContribution* contributionBy} {xpContribution* contributionBy} {yContribution* contributionBy} {ypContribution* contributionBy}")
                else:
                    print(f"{contribution_describing} {xContribution* contributionBy} {xpContribution* contributionBy} {yContribution* contributionBy} {ypContribution* contributionBy}", file=file)

        return PhaseSpaceParticle(x, xp, y, yp, p0.z, p0.delta)

    def __order(self, contributionDescribing: str) -> int:
        """
        内部方法
        由 contribution_describing 查看这行 map 的阶数
        """
        order = 0
        for i in range(6):
            if i == 4:
                continue
            order += int(contributionDescribing[i])

        return order

    def __contribution_by(self, contributionDescribing: str, pp: PhaseSpaceParticle) -> float:
        """
        内部方法
        当前行是如何作用于粒子的
        要看懂只需要研究清楚 cosy map 每行的意义和具体映射方法
        """
        by = 1.0
        x = pp.x
        xp = pp.xp
        y = pp.y
        yp = pp.yp
        delta = pp.delta

        by_x: int = int(contributionDescribing[0])
        by *= x ** by_x

        by_xp = int(contributionDescribing[1])
        by *= xp ** by_xp

        by_y = int(contributionDescribing[2])
        by *= y ** by_y

        by_yp = int(contributionDescribing[3])
        by *= yp ** by_yp

        by_delta = int(contributionDescribing[5])
        by *= delta ** by_delta

        return by

    def apply_phase_space_particles(self, ps: List[PhaseSpaceParticle], order: int = 1) -> List[PhaseSpaceParticle]:
        """
        作用于多个粒子
        """
        return [self.apply(p, order) for p in ps]

    def compute_chromatic_dispersion(self, delta_δ: float = 0.01,
                                     order: int = 1,
                                     centerKineticEnergy_MeV: float = 215.0,
                                     p0: Optional[PhaseSpaceParticle] = None
                                     ) -> float:
        """
        计算色散
        非常粗略的方法，不推荐使用
        计算方法如下：
        1. 生成一个相空间粒子 x=x'=y=y'=0，δ=1%
        2. 经 map 作用，得到作用后的粒子 x=x
        3. 则色散为 x/1%

        返回值即为 x，单位 m
        """
        if p0 is None:
            p0 = PhaseSpaceParticle(0, 0, 0, 0, 0, delta_δ)

        # 转为能量分散，用于 COSY
        p0 = PhaseSpaceParticle.convert_delta_from_momentum_dispersion_to_energy_dispersion(
            p0, centerKineticEnergy_MeV
        )

        # 作用
        p1 = self.apply(p0, order)

        # 转回
        p1 = PhaseSpaceParticle.convert_delta_from_energy_dispersion_to_momentum_dispersion(
            p1, centerKineticEnergy_MeV
        )

        return p1.x - p0.x


class SR:
    """
    粒子脚本生成
    """
    COLOR_BLACK = 1
    COLOR_BLUE = 2
    COLOR_RED = 3
    COLOR_YELLOW = 4
    COLOR_GREEN = 5
    COLOR_WHITE = 10

    @classmethod
    def to_cosy_sr(cls, phase_space_particle: Union[PhaseSpaceParticle, List[PhaseSpaceParticle]], color: int = COLOR_BLUE) -> Union[str, List[str]]:
        """
        将相空间粒子 PhaseSpaceParticle 转为 COSY 脚本 SR < x > < xp > < y > < yp > <T> <dp> <G> <Z> <color> ;

        color 表示 
        """
        if isinstance(phase_space_particle, PhaseSpaceParticle):
            return (
                f"SR {phase_space_particle.x} {phase_space_particle.xp} "
                + f"{phase_space_particle.y} {phase_space_particle.yp} 0 "
                + f"{phase_space_particle.delta} 0 0 {color} ;"
            )
        elif isinstance(phase_space_particle, List):
            return '\n'.join([cls.to_cosy_sr(p) for p in phase_space_particle])
        else:
            print(
                f"phase_space_particle = {phase_space_particle}输入不是 PhaseSpaceParticle 对象，或 PhaseSpaceParticle 对象数组")


class MagnetSlicer:
    """
    磁场切片
    """
    @staticmethod
    def slice_trajectory(
        magnet: Magnet,
        trajectory: Line2,
        Bp: float,
        aperture: float = 60*MM,
        good_field_area_width: float = 60*MM,
        min_step_length: float = 1*MM,
        tolerance: float = 0.1,
        ignore_radisu: float = 50*M,
        ignore_gradient: float = 0.1,
        ignore_second_gradient: float = 1.0,
    ) -> List[str]:
        """
        将磁铁切片，得到可以导入 cosy 的脚本
        magnet                   切片的磁铁/磁场，一般是 Beamline
        trajectory               设计轨道，切片磁场计算位于轨道上
        Bp                       磁钢度
        aperture                 元件孔径
        good_field_area_width    好长度总长度
        min_step_length          最小切片长度（如果相邻的切片参数几乎一致，代码会自动合并切片）
        tolerance                容忍差值，差值范围内的切片会合并，减少总切片数目
        ignore_radisu            偏转半径大于此值的偏转磁铁，将视为偏移段
        ignore_gradient          梯度小于此值的四极场，将置零
        ignore_second_gradient   二阶梯度小于此值的六极场，将置零

        返回值为 cosy 切片脚本数组
        """
        ret: List[str] = []

        multipole_field_along_trajectory: List[ValueWithDistance[List[float]]] = (
            magnet.multipole_field_along(
                line2=trajectory,
                order=2,
                good_field_area_width=good_field_area_width,
                step=min_step_length,
                point_number=10
            )
        )

        size = len(multipole_field_along_trajectory)

        # 实际步长
        real_step = multipole_field_along_trajectory[1].distance - \
            multipole_field_along_trajectory[0].distance

        i = 0

        total_length = 0.0

        while i < size-1:
            multipole_field0: List[float] = multipole_field_along_trajectory[i].value

            B0 = multipole_field0[0]
            if (abs(B0) < 1e-6 or abs(Bp / B0) > ignore_radisu):
                B0 = 0
            T0 = multipole_field0[1]
            if (abs(T0) < ignore_gradient):
                T0 = 0
            L0 = multipole_field0[2]
            if (abs(L0) < ignore_second_gradient):
                L0 = 0

            Bs = [B0]
            Ts = [T0]
            Ls = [L0]

            j = i+1  # 大漏洞，python 中的 for range 与其他语言的 for(;;) 不是完全相同的
            for j in range(i+1, size-1):
                multipole_field: List[float] = multipole_field_along_trajectory[j].value

                B = multipole_field[0]
                if (abs(B) < 1e-6 or abs(Bp / B) > ignore_radisu):
                    B = 0
                T = multipole_field[1]
                if (abs(T) < ignore_gradient):
                    T = 0
                L = multipole_field[2]
                if (abs(L) < ignore_second_gradient):
                    L = 0

                Bs.append(B)
                Ts.append(T)
                Ls.append(L)

                if (
                        BaseUtils.Statistic().add_all(Bs).undulate() > tolerance or
                        BaseUtils.Statistic().add_all(Ts).undulate() > tolerance or
                        BaseUtils.Statistic().add_all(Ls).undulate() > tolerance):
                    break

            B0 = BaseUtils.Statistic().add_all(Bs).average()
            T0 = BaseUtils.Statistic().add_all(Ts).average()
            L0 = BaseUtils.Statistic().add_all(Ls).average()

            length = real_step * (j-i)

            total_length += length

            i = j  # !!

            r = (ignore_radisu + 1) if abs(B0) < 1e-6 else Bp/B0  # 半径，有正负

            cosy_script = None

            if abs(r) > ignore_radisu:
                b2 = T0 * aperture
                b3 = L0 * aperture * aperture
                cosy_script = f"M5 {length} {b2} {b3} 0 0 0 {aperture} ;"
            else:
                angle = BaseUtils.radian_to_angle(length/abs(r))  # 偏转角度
                n1 = -r / B0 * T0
                n2 = -r * r / B0 * L0
                change_direct = 'CB ;' if r < 0 else ''  # r<0 时改变偏转方向
                cosy_script = f"{change_direct} MS {abs(r)} {angle} {aperture} {n1} {n2} 0 0 0 ; {change_direct}"

            ret.append(cosy_script)

        print(f"切片长度{len(ret)}")
        return ret

    @staticmethod
    def slice_track(
        magnet: Magnet,
        track: Line3,
        Bp: float,
        aperture: float = 60*MM,
        good_field_area_width: float = 60*MM,
        min_step_length: float = 1*MM,
        tolerance: float = 0.1,
        ignore_radisu: float = 50*M,
        ignore_gradient: float = 0.1,
        ignore_second_gradient: float = 1.0,
    ) -> List[str]:
        """
        将磁铁切片，得到可以导入 cosy 的脚本
        magnet                   切片的磁铁（磁场，一般是 Beamline）
        track                    粒子轨迹，三维曲线，切片磁场计算位于曲线上
        Bp                       磁钢度
        aperture                 元件孔径
        good_field_area_width    好长度总长度
        min_step_length          最小切片长度（如果相邻的切片参数几乎一致，代码会自动合并切片）
        tolerance                容忍差值，差值范围内的切片会合并，减少总切片数目
        ignore_radisu            偏转半径大于此值的偏转磁铁，将视为偏移段
        ignore_gradient          梯度小于此值的四极场，将置零
        ignore_second_gradient   二阶梯度小于此值的六极场，将置零

        返回值为 cosy 切片脚本数组
        """
        ret: List[str] = []

        multipole_field_along_trajectory: List[ValueWithDistance[List[float]]] = (
            magnet.multipole_field_along_line3(
                line3=track,
                order=2,
                good_field_area_width=good_field_area_width,
                step=min_step_length,
                point_number=6
            )
        )

        size = len(multipole_field_along_trajectory)

        # 实际步长
        real_step = multipole_field_along_trajectory[1].distance - \
            multipole_field_along_trajectory[0].distance

        i = 0

        total_length = 0.0

        while i < size-1:
            multipole_field0: List[float] = multipole_field_along_trajectory[i].value

            B0 = multipole_field0[0]
            if (abs(Bp / B0) > ignore_radisu):
                B0 = 0
            T0 = multipole_field0[1]
            if (abs(T0) < ignore_gradient):
                T0 = 0
            L0 = multipole_field0[2]
            if (abs(L0) < ignore_second_gradient):
                L0 = 0

            Bs = [B0]
            Ts = [T0]
            Ls = [L0]

            j = i+1
            for j in range(i+1, size-1):
                multipole_field: List[float] = multipole_field_along_trajectory[j].value

                B = multipole_field[0]
                if (abs(Bp / B) > ignore_radisu):
                    B = 0
                T = multipole_field[1]
                if (abs(T) < ignore_gradient):
                    T = 0
                L = multipole_field[2]
                if (abs(L) < ignore_second_gradient):
                    L = 0

                Bs.append(B)
                Ts.append(T)
                Ls.append(L)

                if (
                        BaseUtils.Statistic.add_all(Bs).undulate() > tolerance or
                        BaseUtils.Statistic.add_all(Ts).undulate() > tolerance or
                        BaseUtils.Statistic.add_all(Ls).undulate() > tolerance):
                    break

            B0 = BaseUtils.Statistic().add_all(Bs).average()
            T0 = BaseUtils.Statistic().add_all(Ts).average()
            L0 = BaseUtils.Statistic().add_all(Ls).average()

            length = real_step * (j-i)

            total_length += length

            i = j  # !!

            r = Bp/B0  # 半径，有正负

            angle = BaseUtils.radian_to_angle(length/abs(r))  # 偏转角度

            change_direct = 'CB ;' if r < 0 else ''  # r<0 时改变偏转方向

            n1 = -r / B0 * T0

            n2 = -r * r / B0 * L0

            b2 = T0 * aperture

            b3 = L0 * aperture * aperture

            cosy_script = None

            if abs(r) > ignore_radisu:
                cosy_script = f"M5 {length} {b2} {b3} 0 0 0 {aperture} ;"
            else:
                cosy_script = f"{change_direct} MS {abs(r)} {angle} {aperture} {n1} {n2} 0 0 0 ; {change_direct}"

            ret.append(cosy_script)

        print(f"切片长度{len(ret)}")
        return ret
