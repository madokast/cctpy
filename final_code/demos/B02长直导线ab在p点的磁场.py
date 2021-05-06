"""
CCT 建模优化代码
长直导线ab在p点的磁场，展示P3类的使用方法

作者：赵润晓
日期：2021年5月3日
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

DISPERSED_NUMBER = 10000 # 电流元分段数目
MIU0 = 4 * math.pi * 1e-7 # 真空磁导率

def magnetic_field(a,b,current,p):
    # 线段 ab 离散为 N 个点
    ab_dispersed = BaseUtils.linspace(a,b,DISPERSED_NUMBER)

    # 磁场，累加
    B = P3(0,0,0)

    # 累加
    for i in range(DISPERSED_NUMBER-1):
        pi = ab_dispersed[i]
        pii = ab_dispersed[i+1] # pii 即 p(i+1)
        dLi = pii-pi
        mi = (pi+pii)/2
        ri = p-mi

        dB = MIU0/(4*math.pi) * current * (dLi @ ri) / (ri.length()**3)

        B+=dB
    
    return B

b = magnetic_field(
    a = P3.y_direct(-1000),
    b = P3.y_direct(1000),
    current = 1000,
    p = P3.x_direct(1)
)

# 输出 (0.0, 0.0, -0.00019999990000014092)
print(b)

