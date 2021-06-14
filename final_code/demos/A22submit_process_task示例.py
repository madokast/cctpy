"""
CCT 建模优化代码
A11 Baseutils 示例


作者：赵润晓
日期：2021年5月2日
"""

from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

print(multiprocessing.current_process().name)

def add(a,b):
    return a+b

# print(add(1,2)) # 3
# print(add(3,4)) # 7
# print(add(5,6)) # 11

def sub(a,b):
    return a-b

def factorial(a):
    if a<=1:
        return 1
    else:
        return a * factorial(a-1)

def random_p3():
    print(multiprocessing.current_process().name)
    return P3.random()

if __name__ == "__main__":
    BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main()
    print(BaseUtils.submit_process_task(
        task=random_p3,
        param_list=[[],[]]
    ))

