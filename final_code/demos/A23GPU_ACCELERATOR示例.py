"""
CCT 建模优化代码
A23 GPU_ACCELERATOR 示例


作者：赵润晓
日期：2021年5月2日
"""
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

if __name__ == "__main__":
    ga = GPU_ACCELERATOR()
    ga_cpu = GPU_ACCELERATOR(cpu_mode=True)

    p = P3(1,1,1)

    print(ga.vct_length(p))
    print(ga_cpu.vct_length(p))