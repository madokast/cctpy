"""
size location along delta
"""
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
sys.path.append(path.dirname(path.dirname(
    path.abspath(path.dirname(__file__)))))
from cctpy import *

data = """-0.1	189.36	325.43	-63.01	25.14
-0.09	8.66	23.7	-0.13	-3.48
-0.08	3.58	4.37	1.19	-5.63
-0.07	3.5	3.79	0.66	-5.5
-0.06	3.48	3.74	0.3	-5.24
-0.05	3.54	3.59	0	-5.02
-0.04	3.43	3.43	-0.24	-4.82
-0.03	3.46	3.47	-0.42	-4.58
-0.02	3.45	3.53	-0.57	-4.33
-0.01	3.43	3.54	-0.69	-4.11
0	3.43	3.56	-0.77	-3.88
0.01	3.37	3.53	-0.85	-3.66
0.02	3.49	3.52	-0.91	-3.41
0.03	3.36	3.62	-0.88	-3.11
0.04	3.33	3.66	-0.97	-2.83
0.05	3.31	3.42	-0.84	-2.54
0.06	3.6	3.39	-0.9	-2.2
0.07	10.44	19.16	-7.39	-3.25"""

lines  = data.split('\n')

p2ss = [[] for _ in range(4)]

for line in lines:
    # print(line)
    nums = line.split('\t')

    nums = [float(num) for num in nums]
    print(nums)
    delta = nums[0]*100
    for i in range(4):
        p2ss[i].append(P2(delta,nums[i+1]))


if True:
    # 绘制束斑大小
    Plot2.plot_p2s(p2ss[0],describe='r-o')
    Plot2.plot_p2s(p2ss[1],describe='b-o')
    Plot2.ylim(2,5)
    Plot2.info("Momentum dispersion/%","Beam spot size/mm")
    Plot2.legend("Horizontal beam spot size","Vertical beam spot size")

if False:
    # 绘制束斑位置
    Plot2.plot_p2s(p2ss[2],describe='r-o')
    Plot2.plot_p2s(p2ss[3],describe='b-o')
    Plot2.ylim(-8,6)
    Plot2.info("Momentum dispersion/%","Beam spot position/mm")
    Plot2.legend("Horizontal beam spot position","Vertical beam spot position")


Plot2.show()

    
