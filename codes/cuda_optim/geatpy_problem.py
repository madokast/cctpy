import numpy as np
import math
import geatpy as ea
from cuda_optim.geatpy_utils import Inputfile_generator, postprocess
import os


class Myproblem(ea.Problem):
    def __init__(self):
        name = 'FirstSection'  # Name of the question,dosen't matter
        M = 6  # Number of targets
        maxormins = [1] * M  # Want the maximum or the minimun of the target,[1] for the min, [0] for the max
        Dim = 10  # Dimension of the features
        varTypes = [0] * Dim  # Type of the features, [0] for real number , [1] for integer
        # lower bounds of the features
        lb = [-10, -60, 60, 60, 80, 60, 60, 80, -12000, -10000]
        # upper bounds of the features
        ub = [10, 60, 120, 120, 100, 120, 120, 100, -6000, 10000]
        # Whether possible to get the boundary value, [1] for yes, [0] for no
        lbin = [1] * Dim
        ubin = [1] * Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        Vars = pop.Phen  # Generate populations
        Number = len(Vars)
        ID = list(range(1, Number + 1))
        Inputfile_generator(ID, Vars.tolist())
        f = postprocess()
        # pop.CV = np.vstack([f[:, 0] - 2, f[:, 1] - 2, f[:, 2] - 2, f[:, 3] - 2, f[:, 4] - 2, f[:, 5] - 2, ]).T
        pop.ObjV = f
