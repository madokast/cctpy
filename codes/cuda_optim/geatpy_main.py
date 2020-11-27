import geatpy as ea
from problem import Myproblem

problem = Myproblem()
Encoding = 'RI'
NIND = 200
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
population = ea.Population(Encoding, Field, NIND)

myAlgorithm = ea.moea_NSGA3_templet(problem,population)
myAlgorithm.MAXGEN = 300
myAlgorithm.drawing = 0

[NDset, population] = myAlgorithm.run()
NDset.save()

print ('time: %f seconds'%(myAlgorithm.passTime))
print ('evaluation times: %d times'%(myAlgorithm.evalsNum))
print ('NDnum: %d'%(NDset.sizes))
print ('ParetoNum: %d'%(int(NDset.sizes / myAlgorithm.passTime)))