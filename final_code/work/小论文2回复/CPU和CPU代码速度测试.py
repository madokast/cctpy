"""
代码速度测试，三种并行等级

测试项目：16个粒子，100个机架。

测试一 纯 python，单进程
测试二 纯 python，多进程
测试三 CUDA 单个流多处理器（stream Stream Multiprocessor）且每个处理器 1 个线程
测试四 CUDA 单个流多处理器（stream Stream Multiprocessor）且每个处理器 1024 个线程
测试五 CUDA 多个流多处理器（stream Stream Multiprocessor）且每个处理器 1024 个线程
"""


from os import error, path
import sys

sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
sys.path.append(path.dirname(path.dirname(
    path.abspath(path.dirname(__file__)))))

from cctpy import *

# -------------------------------------------- 配置区 -----------------------------------------
#-----------测试开关，true 表示执行此次测试-------------------------
测试一 = not True
测试二 = not True
测试三 = not True
测试四 = True
测试五 = True
# -----------------------------------------------------------------
测试轮数 = 10
footstep = 2*MM # 调整测试复杂度
# 如果测试时间过长，实际耗时10分钟以上，则可以调大
# 如果测试时间过段，实际耗时1分钟以下，那么测试不准确，需要调小
# -------------------------------------------------------------------------------------------

if __name__ == "__main__":
    BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main()
    # 机架
    beamline = (
        Beamline.set_start_point(P2.origin())
        .first_drift(direct=P2.x_direct(),length=5)
        .append_agcct(
            big_r=1.0,
            small_rs=[200*MM,180*MM,160*MM,140*MM],
            bending_angles=[10,10,15,15],
            tilt_angles=[[30],[90,30]],
            winding_numbers=[[50],[10,10,15,15]],
            currents=[10000,10000]
        ).append_drift(2)
        .append_qs(length=1,gradient=0,second_gradient=0,aperture_radius=200*MM)
        .append_drift(2)
    )
    # 粒子
    particle = ParticleFactory.create_proton_along(beamline)

    beamlines = [beamline]*100
    particles = [particle.copy() for _ in range(16)]

    print(f"机架数目 = {len(beamlines)}")
    print(f"粒子数目 = {len(particles)}")
    
    if 测试一:
        print("\n------------执行 测试一 纯 python，单进程---------------")
        print("纯 python 不支持多机架同时运行，测试用时已乘以机架数目")
        for i in range(测试轮数):
            particles = [particle.copy() for _ in range(len(particles))]
            start = time.time()
            ParticleRunner.run_only(particles,beamline,beamline.get_length(),footstep=footstep,report=False,concurrency_level=1)
            print(f"第{i+1}轮，用时{(time.time()-start)*len(beamlines)}s")

    if 测试二:
        print("\n------------执行 测试二 纯 python，多进程---------------")
        print("纯 python 不支持多机架同时运行，测试用时已乘以机架数目")
        for i in range(测试轮数):
            particles = [particle.copy() for _ in range(len(particles))]
            start = time.time()
            ParticleRunner.run_only(particles,beamline,beamline.get_length(),footstep=footstep,report=False,concurrency_level=16)
            print(f"第{i+1}轮，用时{(time.time()-start)*len(beamlines)}s")


    if 测试三:
        print("\n------------测试三 CUDA 单个流多处理器（stream Stream Multiprocessor）且每个处理器 1 个线程---------------")
        print("单个流多处理器时不支持多机架同时运行，测试用时已乘以机架数目")
        gpu = GPU_ACCELERATOR(block_dim_x=1) # block_dim_x 取 1 表示每个处理器 1 个线程
        for i in range(测试轮数):
            particles = [particle.copy() for _ in range(len(particles))]
            start = time.time()
            gpu.track_multi_particle_beamline_for_magnet_with_multi_qs(
                bls=[beamline],ps=particles,distance=beamline.get_length(),footstep=footstep
            )
            print(f"第{i+1}轮，用时{(time.time()-start)*len(beamlines)}s")

    if 测试四:
        print("\n------------测试四 CUDA 单个流多处理器（stream Stream Multiprocessor）且每个处理器 1024 个线程---------------")
        print("单个流多处理器时不支持多机架同时运行，测试用时已乘以机架数目")
        gpu = GPU_ACCELERATOR(block_dim_x=1024) # block_dim_x 取 1 表示每个处理器 1 个线程
        for i in range(测试轮数):
            particles = [particle.copy() for _ in range(len(particles))]
            start = time.time()
            gpu.track_multi_particle_beamline_for_magnet_with_multi_qs(
                bls=[beamline],ps=particles,distance=beamline.get_length(),footstep=footstep
            )
            print(f"第{i+1}轮，用时{(time.time()-start)*len(beamlines)}s")


    if 测试五:
        print("\n------------测试五 CUDA 多个流多处理器（stream Stream Multiprocessor）且每个处理器 1024 个线程---------------")
        gpu = GPU_ACCELERATOR(block_dim_x=1024) # block_dim_x 取 1 表示每个处理器 1 个线程
        for i in range(测试轮数):
            particles = [particle.copy() for _ in range(len(particles))]
            start = time.time()
            gpu.track_multi_particle_beamline_for_magnet_with_multi_qs(
                bls=beamlines,ps=particles,distance=beamline.get_length(),footstep=footstep
            )
            print(f"第{i+1}轮，用时{(time.time()-start)}s")





#-------------------------------result------------------------------------
"""
机架数目 = 100
粒子数目 = 16

------------执行 测试一 纯 python，单进程---------------
纯 python 不支持多机架同时运行，测试用时已乘以机架数目
第1轮，用时8107.001519203186s
第2轮，用时8028.349423408508s
第3轮，用时8052.92227268219s
第4轮，用时8061.962080001831s
第5轮，用时8102.836394309998s
第6轮，用时8045.828056335449s
第7轮，用时8086.695861816406s
第8轮，用时8039.621663093567s
第9轮，用时8044.477415084839s
第10轮，用时8017.894601821899s

------------执行 测试二 纯 python，多进程---------------
纯 python 不支持多机架同时运行，测试用时已乘以机架数目
第1轮，用时1508.3383798599243s
第2轮，用时1536.2271308898926s
第3轮，用时1537.3384714126587s
第4轮，用时1520.224952697754s
第5轮，用时1523.5814094543457s
第6轮，用时1556.738042831421s
第7轮，用时1524.7166633605957s
第8轮，用时1525.917649269104s
第9轮，用时1557.9027891159058s
第10轮，用时1595.534324645996s

------------测试三 CUDA 单个流多处理器（stream Stream Multiprocessor）且每个处理器 1 个线程---------------
单个流多处理器时不支持多机架同时运行，测试用时已乘以机架数目
第1轮，用时22117.388820648193s
第2轮，用时22452.09310054779s
第3轮，用时21377.52501964569s
第4轮，用时22541.461658477783s
第5轮，用时20024.002528190613s
第6轮，用时22843.680357933044s
第7轮，用时21767.91431903839s
第8轮，用时19049.305725097656s
第9轮，用时24832.381081581116s
第10轮，用时24772.587490081787s

------------测试四 CUDA 单个流多处理器（stream Stream Multiprocessor）且每个处理器 1024 个线程---------------
单个流多处理器时不支持多机架同时运行，测试用时已乘以机架数目
第1轮，用时201.78251266479492s
第2轮，用时196.14384174346924s
第3轮，用时201.35140419006348s
第4轮，用时205.42361736297607s
第5轮，用时203.2546043395996s
第6轮，用时199.21634197235107s
第7轮，用时200.03154277801514s
第8轮，用时199.79910850524902s
第9轮，用时197.27671146392822s
第10轮，用时197.49228954315186s

------------测试五 CUDA 多个流多处理器（stream Stream Multiprocessor）且每个处理器 1024 个线程---------------
第1轮，用时20.19264054298401s
第2轮，用时20.242048740386963s
第3轮，用时20.257314682006836s
第4轮，用时20.18230152130127s
第5轮，用时20.051960706710815s
第6轮，用时20.303875207901s
第7轮，用时20.29099702835083s
第8轮，用时20.44864773750305s
第9轮，用时20.18773889541626s
第10轮，用时20.331196308135986s

汇总
t1 8058.76
t2 1538.65
t3 22177.83
t4 200
t5 20
"""