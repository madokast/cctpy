from __future__ import division
from itertools import product
import numpy as np
import math
import os

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

    os.system("run.cmd")
    out = np.loadtxt('output.txt', usecols=[1,2,3,4])

    # beamsize_x = out[:, [1,3,5]]
    # beamsize_y = out[:, [7,9,11]]
    # beamcenter_x = out[:, [0, 2, 4]].max(axis=1)
    # beamcenter_y = out[:, [6, 8, 10]].max(axis=1)
    # beamx_max = beamsize_x.max(axis=1)
    # beamx_min = beamsize_x.min(axis=1)
    # beamy_max = beamsize_y.max(axis=1)
    # beamy_min = beamsize_y.min(axis=1)
    # sizedata = np.vstack([beamx_max, beamy_max, abs(beamsize_x - beamsize_y).max(axis=1), beamx_max - beamx_min,
    #                       beamy_max - beamy_min, beamcenter_x, beamcenter_y]).T

    # np.savetxt('outputprocessed.txt', sizedata)

    return out
