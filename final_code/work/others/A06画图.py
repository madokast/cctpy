"""
[0.1,	0.02]
[0.2,	0.03]
[0.3,	0.08]
[0.4,	0.13]
[0.5,	0.15]
[0.6,	0.17]
[0.7,	0.22]
[0.8,	0.27]
[0.9,	0.28]
[1.0,	0.30]

[0.1, 0.5],
[0.2, 1.75],
[0.3, 2.50],
[0.4, 3.0],
[0.5, 4.5],
[0.6, 6.0],
[0.7, 7.0],
[0.8, 8.3],
[0.9, 10.5],
[1.0, 12.5],

"""

from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
sys.path.append(path.dirname(path.dirname(
    path.abspath(path.dirname(__file__)))))
from cctpy import *

data = [[0.1, 0.5],
[0.2, 1.75],
[0.3, 2.50],
[0.4, 3.0],
[0.5, 4.5],
[0.6, 6.0],
[0.7, 7.0],
[0.8, 8.3],
[0.9, 10.5],
[1.0, 12.5],]

p2s = P2.from_list(data)

Plot2.plot_p2s(p2s,describe='r-o')
Plot2.info(
    "Error range/mm",
    "Maximum fluctuation/mm",
)
Plot2.show()