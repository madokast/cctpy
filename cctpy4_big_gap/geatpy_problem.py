import geatpy as ea
from run import run


class Myproblem(ea.Problem):
    def __init__(self):
        name = 'FirstSection'  # Name of the question,dosen't matter
        M = 11  # Number of targets
        maxormins = [1] * M  # Want the maximum or the minimun of the target,[1] for the min, [0] for the max
        Dim = 10  # Dimension of the features
        varTypes = ([0] * 10)  # Type of the features, [0] for real number , [1] for integer
        # lower bounds of the features
        lb = [-6, -100, 80, 80, 80, 60, 60, 80, 9000, -7500,]
        # upper bounds of the features
        ub = [6, 100, 100, 105, 100, 100, 100, 100, 10000, -7000]
        # Whether possible to get the boundary value, [1] for yes, [0] for no
        lbin = [1] * Dim
        ubin = [1] * Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        Vars = pop.Phen  # Generate populations
        pop.ObjV = run(Vars)
