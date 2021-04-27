from __future__ import division
from itertools import product
import time

import numpy as np
import math
import os
from cuda_optim.main import run


def Inputfile_generator(ID, samples):
    Number = len(ID)
    input = open('input.txt', 'w', encoding='utf-8')
    input.write(str(Number) + '\n')
    for i in range(Number):
        input.write(str(ID[i]) + '\n')
        for j in range(len(samples[0])):
            input.write(str(samples[i][j]) + '\n')
    input.close()


def postprocess():
    while True:
        try:
            run()
            break
        except Exception as e:
            print("CUDA出现异常，30s 后重试", e)
            time.sleep(30)

    # out = np.loadtxt('output.txt', usecols=[1])[:, np.newaxis] # 只有一列 用这个
    out = np.loadtxt('output.txt', usecols=[1, 2, 3, 4, 5, 6])

    return out
