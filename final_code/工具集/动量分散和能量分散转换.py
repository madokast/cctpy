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


if __name__ == "__main__":
    for dp in BaseUtils.linspace(0.0,0.1,11):
        print(f"dp={dp}, dE={Protons.convert_momentum_dispersion_to_energy_dispersion(dp,250)}")


# dp=0.0, dE=0.0
# dp=0.010000000000000002, dE=0.017896104739526547
# dp=0.020000000000000004, dE=0.035792209479053094
# dp=0.030000000000000006, dE=0.05368831421857964
# dp=0.04000000000000001, dE=0.07158441895810619
# dp=0.05000000000000001, dE=0.08948052369763274
# dp=0.06000000000000001, dE=0.10737662843715928
# dp=0.07, dE=0.12527273317668583
# dp=0.08000000000000002, dE=0.14316883791621238
# dp=0.09000000000000002, dE=0.16106494265573895
# dp=0.10000000000000002, dE=0.17896104739526547