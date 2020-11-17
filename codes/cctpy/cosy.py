"""
读取 COSY 任意阶矩阵
"""
from typing import List

from cctpy.baseutils import Stream
from cctpy.particle import PhaseSpaceParticle


class CosyMap:
    ITEM_LENGTH = len("-0.0000000E+00")

    def __init__(self, map: str):
        self.map = map
        self.contributionLinesString = map.split('\n')

    def apply(self, p0: PhaseSpaceParticle, order: int = 1) -> PhaseSpaceParticle:
        x = 0.0
        xp = 0.0
        y = 0.0
        yp = 0.0

        for contributionLineString in self.contributionLinesString:
            x_contribution_string = contributionLineString[0:CosyMap.ITEM_LENGTH + 1]
            xp_contribution_string = contributionLineString[CosyMap.ITEM_LENGTH + 1: CosyMap.ITEM_LENGTH * 2 + 1]
            y_contribution_string = contributionLineString[CosyMap.ITEM_LENGTH * 2 + 1: CosyMap.ITEM_LENGTH * 3 + 1]
            yp_contribution_string = contributionLineString[CosyMap.ITEM_LENGTH * 3 + 1: CosyMap.ITEM_LENGTH * 4 + 1]

            contribution_describing = contributionLineString[CosyMap.ITEM_LENGTH * 5 + 2:]

            this_order = self.__order(contribution_describing)

            if this_order > order:
                break

            contributionBy = self.__contribution_by(contribution_describing, p0)

            xContribution = float(x_contribution_string)
            x += xContribution * contributionBy

            xpContribution = float(xp_contribution_string)
            xp += xpContribution * contributionBy

            yContribution = float(y_contribution_string)
            y += yContribution * contributionBy

            ypContribution = float(yp_contribution_string)
            yp += ypContribution * contributionBy

        return PhaseSpaceParticle(x, xp, y, yp, p0.z, p0.delta)

    def __order(self, contributionDescribing: str) -> int:
        order = 0
        for i in range(6):
            if i == 4:
                continue
            order += int(contributionDescribing[i])

        return order

    def __contribution_by(self, contributionDescribing: str, pp: PhaseSpaceParticle) -> float:
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
        return Stream(ps).map(
            lambda p: self.apply(p, order)
        ).to_list()
