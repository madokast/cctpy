"""
2021年5月21日 

重新调整束线
前段一个 QS
后段漂移段 2.5 米

优化识别码 202105210001
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *
from work.A03run import *
from work.A03geatpy_problem import *


if __name__ == '__main__':
    # multiprocessing.Process(target=runviz).start()  # Start Visualization Server
    # time.sleep(15)

    BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main()
    problem = Myproblem()
    Encoding = 'RI'
    NIND = 24*5
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)

    myAlgorithm = ea.moea_NSGA3_templet(problem, population)
    myAlgorithm.MAXGEN = 500000
    myAlgorithm.drawing = 0

    [NDset, population] = myAlgorithm.run()
    NDset.save()

    print('time: %f seconds' % (myAlgorithm.passTime))
    print('evaluation times: %d times' % (myAlgorithm.evalsNum))
    print('NDnum: %d' % (NDset.sizes))
    print('ParetoNum: %d' % (int(NDset.sizes / myAlgorithm.passTime)))