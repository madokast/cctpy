"""
Z01八极磁铁六个峰问题

作者：赵润晓
日期：2021年5月20日
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

def cf(e:P2,p:P2)->P2:
    r = (e-p).length()
    rr = (p-e).normalize()
    return r*rr



es = BaseUtils.Ellipse.create_standard_ellipse(1,1).uniform_distribution_points_along_edge(4)

ps = BaseUtils.Ellipse.create_standard_ellipse(0.01,0.01).uniform_distribution_points_along_edge(360)


degs = BaseUtils.linspace(0,359,360)

efs = []
for p in ps:
    ef = 0
    for e in es:
        ef += cf(e,p).x
    efs.append(ef)

Plot2.plot_xy_array(degs,efs)


Plot2.show()