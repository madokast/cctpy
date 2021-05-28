import multiprocessing
import os
import time

import geatpy as ea
from optimization.geatpy_problem_total import Myproblem
from cctpy import BaseUtils


# def runviz():
#     os.system('python -m visdom.server -p ' + str(VIZ_PORT))


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
