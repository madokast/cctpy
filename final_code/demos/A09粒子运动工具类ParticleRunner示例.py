"""
CCT 建模优化代码
A09粒子运动工具类 ParticleRunner 示例


作者：赵润晓
日期：2021年4月29日
"""

from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

# ParticleRunner 类提供了粒子在磁场中运动的函数
# 计算方法包括自己实现的 runge_kutta4 法（GPU加速时也是用这一方法）
# 另外还有 scipy 包提供的 ode 法，这个方法更智能，可以自动调整积分步长

# 首先创建一个粒子和磁铁来演示 ParticleRunner 类的函数
# 创建一个位于全局坐标系原点，沿着 x 轴正向运动，动能 250 MeV 的质子
proton250 = ParticleFactory.create_proton(
    position=P3.origin(),
    direct=P3.x_direct(),
    kinetic_MeV=250
) 

# 创建一个产生匀强磁场的磁铁，磁场沿 z 轴正方向，大小为 2.4321282996 T
uniform_magnet243 = Magnet.uniform_magnet(magnetic_field=P3(z=2.4321282996))

# 类私有函数 __callback_for_runge_kutta4()
# 将二阶微分方程转为一阶，返回一个函数 Callable[[float, numpy.ndarray], numpy.ndarray]
# 原先的粒子在磁场运动方程为
#   v = p'
#   a = (q/m)v×B
# 自变量 t
# 这是一个二阶微分方程
# 通过向量化转为一阶
# 令 Y = [v,p]
# 则方程变为 Y' = [a, v] = [(q/m)v×B, v]
# 这个函数仅在内部才能调用，因此不做示例


# 类私有函数 __callback_for_solve_ode()
# 函数功能和 __callback_for_runge_kutta4() 一致
# 不同点在于
# __callback_for_runge_kutta4() 返回的函数，入参是 t,[p,v] 返回值是 [v,a]，其中数组元素都是三维矢量 P3
# __callback_for_solve_ode() 返回的函数，入参是 t,[px,py,px,vx,vy,vz] 返回值是 [vx,vy,vz,ax,ay,az] 其中数组元素都是浮点数


# 函数 run_only(p,m,length,footstep,concurrency_level,report) 让粒子(群)在磁场中运动 length 距离
# 函数没有返回值。粒子运动完成后，粒子信息（位置、速度等）反映运动完成时的值
# 参数意义和可选值如下
# p 粒子 RunningParticle，或者粒子数组 [RunningParticle...]。注意运行后，粒子发生变化，返回运动后的坐标、速度值。
# m 磁场
# length 运动长度
# footstep 步长，默认 20 mm
# concurrency_level 并行度，默认值 1
# report 是否打印并行任务计划，默认 True 打印
# 下面让粒子 proton250 在 uniform_magnet243 下运动 pi/2 米
if __name__ == "__main__": 
    # 因为代码后面采用了多线程，所以需要将代码放在 __main__ 里面
    proton250_0 = proton250.copy() # 拷贝一下，为了后面再次使用
    ParticleRunner.run_only(
        p= proton250_0,
        m=uniform_magnet243,
        length= math.pi/2
    )
    print(proton250_0)
    # 输出为
    # p=(1.0000000541349048, -0.9999999670349127, 0.0),v=(16.0287304376252, -183955178.02123857, 0.0),v0=183955178.0274753


# 再举一个例子，取两个粒子，采用并行任务计算
    proton250_1 = proton250.copy() # 拷贝一下，为了后面再次使用
    BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main() # 使用并行计算时，需要这条语句（实际上没有任何意义，只是提醒代码需要被 if __name__ == "__main__": 包裹）
    ParticleRunner.run_only([proton250_0,proton250_1],uniform_magnet243,math.pi/2,concurrency_level=2)
    print(proton250_0) # p=(1.742677909061241e-07, -2.000000108269802, 0.0),v=(-183955178.01500052, -32.05746095441282, 0.0),v0=183955178.0274753
    print(proton250_1) # p=(1.0000000541349048, -0.9999999670349127, 0.0),v=(16.0287304376252, -183955178.02123857, 0.0),v0=183955178.0274753
    # 任务运行时会打印并行任务运行信息：
    # 处理并行任务，任务数目2，并行等级2
    # 任务完成，用时0.7770464420318604秒

# 再举一个例子，取两个粒子，不采用并行任务计算，只需要 concurrency_level 取 1 即可
    ParticleRunner.run_only([proton250_0,proton250_1],uniform_magnet243,math.pi/2,concurrency_level=1)
    print(proton250_0) # p=(-1.000000054067076, -1.0000003155704955, 0.0),v=(-48.086191406007856, 183955178.00876093, 0.0),v0=183955178.0274753
    print(proton250_1) # p=(1.742677909061241e-07, -2.000000108269802, 0.0),v=(-183955178.01500052, -32.05746095441282, 0.0),v0=183955178.0274753
    # 任务运行时会打印任务运行信息：
    # 当前使用单线程进行粒子跟踪，如果函数支持多线程并行，推荐使用多线程
    # 运行一个粒子需要0.00557秒，估计总耗时0.01114秒
    # 100.00%  finished
    # 实际用时0.01237秒

# run_only_ode() 让粒子(群)在磁场中运动 length 距离
# 这个函数的效果和 run_only() 一样。同样没有返回值
# 不同点在于前者使用 runge_kutta4 法计算，后者使用 scipy 包采用变步长的方法计算
# 参数含义如下：
# p 粒子 RunningParticle，或者粒子数组 [RunningParticle...]。注意运行后，粒子发生变化，返回运动后的坐标、速度值。
# m 磁场
# length 运动长度
# footstep 步长，默认 20 mm
# absolute_tolerance 绝对误差，默认 1e-8
# relative_tolerance 相对误差，默认 1e-8
    ParticleRunner.run_only_ode([proton250_0]*10,uniform_magnet243,math.pi/2)
    print(proton250_0) 
    # (1.742677909061241e-07, -2.000000108269802, 0.0),v=(-183955178.01500052, -32.05746095441282, 0.0),v0=183955178.0274753
    # 任务运行时会打印任务运行信息：
    # track 10 particles
    # ▇▇▇▇▇▇▇▇▇▇ finished


# 函数 run_get_trajectory() 运行一个粒子，并返回轨迹，轨迹是 P3 的数组
# 参数如下
# p 粒子
# m 磁场
# length 运动长度
# footstep 步长，默认 20 mm
    traj = ParticleRunner.run_get_trajectory(proton250_0,uniform_magnet243,math.pi)
    # 去除下面两行注释查看绘图结果
    plt.gca(projection="3d").plot(*P3.extract(traj))
    plt.show()

# 函数 run_get_all_info() 运行一个粒子，获取全部信息
# 所谓全部信息即每一步粒子的所有信息，包含位置、速度等，返回值是 RunningParticle 数组
    all_info = ParticleRunner.run_get_all_info(proton250.copy(),uniform_magnet243,1,0.1)
    for info in all_info:
        print(info.__str__()  + " distance = " + str(info.distance))
    # 输出如下：
    # p=(0.0, 0.0, 0.0),v=(183955178.0274753, 0.0, 0.0),v0=183955178.0274753 distance = 0.0
    # p=(0.0998333333513896, -0.004995833063166415, 0.0),v=(183036168.7167266, -18364857.6149287, 0.0),v0=183955178.0274753 distance = 0.1
    # p=(0.19866916542168667, -0.01993340168785551, 0.0),v=(180288325.76992166, -36546219.71405112, 0.0),v0=183955178.0274753 distance = 0.2
    # p=(0.2955199630138605, -0.044663454753014774, 0.0),v=(175739104.70211726, -54362424.462139845, 0.0),v0=183955178.0274753 distance = 0.3
    # p=(0.3894180267177238, -0.0789388980457223, 0.0),v=(169433959.7707788, -71635458.55155955, 0.0),v0=183955178.0274753 distance = 0.4
    # p=(0.47942515982512757, -0.12241726314185991, 0.0),v=(161435889.81242117, -88192735.84918669, 0.0),v0=183955178.0274753 distance = 0.5
    # p=(0.5646420424642986, -0.1746641292326529, 0.0),v=(151824808.7799325, -103868821.81686696, 0.0),v0=183955178.0274753 distance = 0.6000000000000001
    # p=(0.644217217290238, -0.2351574637073639, 0.0),v=(140696747.26995525, -118507086.47560275, 0.0),v0=183955178.0274753 distance = 0.7
    # p=(0.7173555969492972, -0.3032928381225379, 0.0),v=(128162893.01838718, -131961269.39762901, 0.0),v0=183955178.0274753 distance = 0.8
    # p=(0.7833264083143161, -0.37838946744163776, 0.0),v=(114348479.95103914, -144096941.08952317, 0.0),v0=183955178.0274753 distance = 0.9
    # p=(0.8414704941142988, -0.4596970122030806, 0.0),v=(99391536.88967079, -154792846.1647193, 0.0),v0=183955178.0274753 distance = 1.0


# 函数 run_only_deprecated() 已被废弃，因为没有使用 runge_kutta4 等控制误差的积分方法

# 函数 run_get_trajectory_deprecated() 已被废弃，因为没有使用 runge_kutta4 等控制误差的积分方法

# 函数 run_get_all_info_deprecated() 已被废弃，因为没有使用 runge_kutta4 等控制误差的积分方法

