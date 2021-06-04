"""
CCT 建模优化代码
COSY 

作者：赵润晓
日期：2021年6月3日
"""

from cctpy import *


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

    def apply(self, p0: PhaseSpaceParticle, order: int = 1, print_detail:bool=False,file=None) -> PhaseSpaceParticle:
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
                    print(f"{contribution_describing} {xContribution* contributionBy} {xpContribution* contributionBy} {yContribution* contributionBy} {ypContribution* contributionBy}",file=file)

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

