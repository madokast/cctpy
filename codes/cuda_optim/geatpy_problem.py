import numpy as np
import math
import geatpy as ea
from utils import Inputfile_generator,postprocess
import os

class Myproblem(ea.Problem):
    #Define the bacis parameters
    def __init__(self):
        name = 'FirstSection'        #Name of the question,dosen't matter
        M = 4              #Number of targets
        maxormins = [1]*M   #Want the maximum or the minimun of the target,[1] for the min, [0] for the max
        Dim = 8            #Dimension of the features
        varTypes = [0]*Dim  #Type of the features, [0] for real number , [1] for integer
        #lower bounds of the features
        lb = [70, 70, 70, 70, 70, 70, -10000, -7000]
        #upper bounds of the features
        ub = [110, 110, 110, 110, 110, 110, -9000, -5000]
        #Whether possible to get the boundary value, [1] for yes, [0] for no
        lbin = [1]*Dim
        ubin = [1]*Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        Vars = pop.Phen     #Generate populations
        Number = len(Vars)
        ID = list(range(1,Number+1))
        Inputfile_generator(ID, Vars.tolist())
        f = postprocess()
        #Limitations
        pop.CV = np.vstack([f[:,0]-2,f[:,1]-2,f[:,2]-2,f[:,3]-2,]).T
        pop.ObjV = f