import geatpy as ea
from optimization.run_total import run


class Myproblem(ea.Problem):
    def __init__(self):
        name = 'FirstSection'  # Name of the question,dosen't matter
        M = 11  # 目标维度
        maxormins = [1] * M  # 目标是要最大还是最小 Want the maximum or the minimun of the target,[1] for the min, [0] for the max
        Dim = 20  # 变量维度
        varTypes = ([0] * 12) + ([1] * 2) + ([0] * 2) + ([0]*4)  # 变量类型浮点/整数, [0] for real number , [1] for integer
        # 变量最小值
        lb = [-8, -13, -127, -72,    60, 60, 80,      60, 60, 80, -12000, 6000,15,15,-6,-100,   -8,-100,-20,-20]
        # lb = [-7.306812648,	0.550519664,	96.27929383,	-28.79607736,107.9136697,	64.40246184,	83.98949224,	105.4492682,64.2875468,	84.84352535,	766.2625325,	3965.557502,	21,	25]
        # 变量最大值
        ub = [8,  13,   127,  72,  120, 120, 100, 120, 120, 100, -8000, 14000,   25,25,6,100,    8,100,20,20]
        # ub = [-7.306812648,	0.550519664,	96.27929383,	-28.79607736,107.9136697,	64.40246184,	83.98949224,	105.4492682,64.2875468,	84.84352535,	766.2625325,	3965.557502,	21,	25]
        # 是否可以取变量的边界值
        lbin = [1] * Dim
        ubin = [1] * Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        Vars = pop.Phen  # Generate populations
        pop.ObjV = run(Vars)
