import geatpy as ea
from work.optim.A02run import run


class Myproblem(ea.Problem):
    def __init__(self):
        name = 'FirstSection'  # Name of the question,dosen't matter
        M = 9  # 目标维度
        maxormins = [1] * M  # 目标是要最大还是最小 Want the maximum or the minimun of the target,[1] for the min, [0] for the max
        Dim = 16+7  # 变量维度
        varTypes = ([0] * 12) + ([1] * 2) + ([0] * 2) + ([0] * 4) + ([1] * 3)  # 变量类型浮点/整数, [0] for real number , [1] for integer
        # 变量最小值
        lb = [-10, -10, -100, -100,    60, 60, 80,      60, 60, 80, -15000, 4000,15,15,-10,-10,  60, 60, 80,-7500,21 - 4, 50 - 15, 50 - 15]
        # lb = [-7.306812648,	0.550519664,	96.27929383,	-28.79607736,107.9136697,	64.40246184,	83.98949224,	105.4492682,64.2875468,	84.84352535,	766.2625325,	3965.557502,	21,	25]
        # 变量最大值
        ub = [10,  10,   100,  100,  120, 120, 100, 120, 120, 100, -6000, 16000,   25,25,10,10,  120, 120, 100,-5500,21 + 4, 50 + 15, 50 + 15]
        # ub = [-7.306812648,	0.550519664,	96.27929383,	-28.79607736,107.9136697,	64.40246184,	83.98949224,	105.4492682,64.2875468,	84.84352535,	766.2625325,	3965.557502,	21,	25]
        # 是否可以取变量的边界值
        lbin = [1] * Dim
        ubin = [1] * Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        Vars = pop.Phen  # Generate populations
        pop.ObjV = run(Vars)
