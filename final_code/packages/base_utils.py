"""
CCT 建模优化代码
基本工具

作者：赵润晓
日期：2021年4月24日
"""

import multiprocessing  # since v0.1.1 多线程计算
import time  # since v0.1.1 统计计算时长
from typing import Callable, Dict, Generic, Iterable, List, NoReturn, Optional, Tuple, TypeVar, Union
import matplotlib.pyplot as plt
import math
import random  # since v0.1.1 随机数
import sys
import os  # since v0.1.1 查看CPU核心数
import numpy
from scipy.integrate import solve_ivp  # since v0.1.1 ODE45
import warnings  # since v0.1.1 提醒方法过时
from packages.constants import *
from packages.point import P2,P3

class BaseUtils:
    """
    这里存放一些简单的工具，如
    1. 判断两个对象是否相等
    2. numpy 中用于生成均匀分布的 linspace 方法
    3. 角度转弧度 angle_to_radian 和 弧度转角度 radian_to_angle
    4. 打印函数调用栈 print_traceback （这个主要用于 debug）
    5. 椭圆。用于生成椭圆圆周上均匀分布的若干点
    """

    @staticmethod
    def equal(
            a: Union[float, int, P2, P3],
            b: Union[float, int, P2, P3],
            err: float = 1e-6,
            msg: Optional[str] = None,
    ) -> bool:
        """
        判断 a b 是否相等，相等返回 true
        当 a b 不相等时，若 msg 为空，返回 flase，否则抛出异常，异常信息即 msg

        示例：
        """
        if (isinstance(a, float) or isinstance(a, int)) and (
                isinstance(b, float) or isinstance(b, int)
        ):
            if (
                    a == b
                    or abs(a - b) <= err
                    or ((a + b != 0.0) and ((2 * abs((a - b) / (a + b))) <= err))
            ):
                return True
            else:
                if msg is None:
                    return False
                else:
                    raise AssertionError(msg)
        elif (isinstance(a, P2) and isinstance(b, P2)) or (
                isinstance(a, P3) and isinstance(b, P3)
        ):
            if a.__eq__(b, err=err, msg=msg):
                return True
            else:
                if msg is None:
                    return False
                else:
                    raise AssertionError(msg)
        else:
            if a == b:
                return True
            else:
                if msg is None:
                    return False
                else:
                    raise AssertionError(msg)

    @staticmethod
    def linspace(
            start: Union[float, int, P2, P3], end: Union[float, int, P2, P3], number: int
    ) -> List[Union[float, P2, P3]]:
        """
        同 numpy 的 linspace
        """
        # 除法改成乘法以适应 P2 P3 对象
        d = (end - start) * (1 / (number - 1))
        # i 转为浮点以适应 P2 P3 对象
        return [start + d * float(i) for i in range(number)]

    @staticmethod
    def angle_to_radian(
            deg: Union[float, int, List[Union[float, int]]]
    ) -> Union[float, List[float]]:
        """
        角度值转弧度制
        对于单个角度，或者角度数组都可以使用
        """
        if isinstance(deg, float) or isinstance(deg, int):
            return deg / 180.0 * math.pi
        elif isinstance(deg, List):
            return [BaseUtils.angle_to_radian(d) for d in deg]
        else:
            raise NotImplementedError

    @staticmethod
    def radian_to_angle(
            rad: Union[float, int, List[Union[float, int]]]
    ) -> Union[float, List[float]]:
        """
        弧度制转角度制
        对于单个弧度，或者弧度数组都可以使用
        """
        if isinstance(rad, float) or isinstance(rad, int):
            return rad * 180.0 / math.pi
        elif isinstance(rad, List):
            return [BaseUtils.radian_to_angle(d) for d in rad]
        elif isinstance(rad, numpy.ndarray):
            return numpy.array([BaseUtils.radian_to_angle(d) for d in rad])
        else:
            raise NotImplementedError

    @staticmethod
    def circle_center_and_radius(p1: P2, p2: P2, p3: P2) -> Tuple[P2, float]:
        """
        已知三个二维点 p1 p2 p3
        求由这三个点组成的圆的圆心和半径
        方法来自：https://blog.csdn.net/liutaojia/article/details/83625151
        """
        x1 = p1.x
        x2 = p2.x
        x3 = p3.x
        y1 = p1.y
        y2 = p2.y
        y3 = p3.y
        z1 = x2 ** 2 + y2 ** 2 - x1 ** 2 - y1 ** 2
        z2 = x3 ** 2 + y3 ** 2 - x1 ** 2 - y1 ** 2
        z3 = x3 ** 2 + y3 ** 2 - x2 ** 2 - y2 ** 2
        A = numpy.array(
            [[(x2 - x1), (y2 - y1)], [(x3 - x1), (y3 - y1)], [(x3 - x2), (y3 - y2)]]
        )
        B = 0.5 * numpy.array([[z1], [z2], [z3]])
        c = numpy.linalg.inv(A.T @ A) @ A.T @ B
        c = P2.from_numpy_ndarry(c)
        # c = (A'*A)\A'*B;
        R1 = math.sqrt((c.x - x1) ** 2 + (c.y - y1) ** 2)
        R2 = math.sqrt((c.x - x2) ** 2 + (c.y - y2) ** 2)
        R3 = math.sqrt((c.x - x3) ** 2 + (c.y - y3) ** 2)
        R = (R1 + R2 + R3) / 3
        return c, R

    @staticmethod
    def polynomial_fitting(xs: List[float], ys: List[float], order: int) -> List[float]:
        """
        多项式拟合
        xs 自变量，ys 变量，拟合阶数为 order，返回一个数组
        数组第 0 项为拟合常数项
        数组第 i 项为拟合 i 次项
        """
        fit = numpy.polyfit(xs, ys, order)
        return fit[::-1].tolist()

    @staticmethod
    def list_multiply(
            li: Union[List[int], List[float], List[P2], List[P3]], number: Union[int, float]
    ) -> Union[List[int], List[float], List[P2], List[P3]]:
        """
        让数组中每个元素都乘以一个数
        """
        return [e * number for e in li]

    @staticmethod
    def derivative(func: Callable[[float], Union[float, P2, P3]],
                   delta: float = 1e-7) -> Callable[[float], Union[float, P2, P3]]:
        """
        函数 func 求导，微分
        delta 即 Δ，f' = (f(x+Δ)-f(x))/Δ
        """
        def d(x: float) -> Union[float, P2, P3]:
            return (func(x+delta/2)-func(x-delta/2))/delta

        return d

    @staticmethod
    def interpolate_lagrange(x: float, x0: float, y0: float,
                             x1: float, y1: float, x2: float, y2: float, x3: float, y3: float,
                             error: float = 1e-8) -> float:
        """
        拉格朗日插值法 4 个点
        利用四点 (x0,y0) (x1,y1) (x2,y2) (x3,y3) 多项式插值，f(x)
        返回 x 对应的 y，即 f(x)

        当 x 和 xi 的差小于 error 时，直接返回 yi，既是为了快速计算，也是为了防止后面公式中除0

        since v0.1.3 这个函数引入，为了计算 opera 导出的磁场表格数据，在任意一点的磁场
        """
        if abs(x-x0) < error:
            return y0
        if abs(x-x1) < error:
            return y1
        if abs(x-x2) < error:
            return y2
        if abs(x-x3) < error:
            return y3

        t0 = (x - x1)*(x - x2)*(x - x3)*y0 / ((x0 - x1)*(x0 - x2)*(x0 - x3))
        t1 = (x - x0)*(x - x2)*(x - x3)*y1 / ((x1 - x0)*(x1 - x2)*(x1 - x3))
        t2 = (x - x0)*(x - x1)*(x - x3)*y2 / ((x2 - x0)*(x2 - x1)*(x2 - x3))
        t3 = (x - x0)*(x - x1)*(x - x2)*y3 / ((x3 - x0)*(x3 - x1)*(x3 - x2))

        tt = t0 + t1 + t2 + t3

        if math.isnan(tt):
            print(
                f"error in interpolate_lagrange params={x},{x0},{y0},{x1},{y1},{x2},{y2},{x3},{y3}")
            return 0.0

        return tt

    @staticmethod
    def is_sorted(li: List) -> bool:
        """
        判断数组是否有序
        这个方法来自 https://www.zhihu.com/question/368573897
        虽然无法快速退出，但很简洁
        """
        return all([li[i] <= li[i + 1] for i in range(len(li) - 1)])

    @staticmethod
    def print_traceback() -> None:
        """
        打印函数调用栈
        用于 debug
        -------

        """
        f = sys._getframe()
        while f is not None:
            print(f)
            f = f.f_back

    @staticmethod
    def runge_kutta4(t0: float, t_end: float, Y0: T, y_derived_function: Callable[[float, T], T], dt: float,
                     record: bool = False) -> Union[T, Tuple[List[float], List[T]]]:
        """
        4 阶 runge kutta 法求解微分方程组
        since v0.1.1
        """
        number: int = math.ceil((t_end - t0) / dt)
        dt = (t_end - t0) / float(number)

        if record:
            ts = [t0]
            Ys = [Y0]
            for ignore in range(number):
                k1 = y_derived_function(t0, Y0)
                k2 = y_derived_function(t0 + dt / 2, Y0 + dt / 2 * k1)
                k3 = y_derived_function(t0 + dt / 2, Y0 + dt / 2 * k2)
                k4 = y_derived_function(t0 + dt, Y0 + dt * k3)

                t0 = t0 + dt
                Y0 = Y0 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                ts.append(t0)
                Ys.append(Y0)
            return (ts, Ys)
        else:
            for ignore in range(number):
                k1 = y_derived_function(t0, Y0)
                k2 = y_derived_function(t0 + dt / 2, Y0 + dt / 2 * k1)
                k3 = y_derived_function(t0 + dt / 2, Y0 + dt / 2 * k2)
                k4 = y_derived_function(t0 + dt, Y0 + dt * k3)

                t0 += dt
                Y0 += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            return Y0

    @staticmethod
    def solve_ode(t0: float, t_end: float, Y0: T, y_derived_function: Callable[[float, T], T], dt: float,
                  record: bool = False, absolute_tolerance: float = 1e-8, relative_tolerance: float = 1e-8) -> Union[
            T, Tuple[List[float], List[T]]]:
        """
        scipy 中 ode45
        即变步长 4 阶 runge kutta 法
        since v0.1.1
        """
        if record:
            raise NotImplementedError
            # number: int = math.ceil((t_end-t0)/dt)
            # t_eval = numpy.linspace(t0, t_end, number)
            # s = solve_ivp(y_derived_function, [
            #               t0, t_end], Y0, t_eval=t_eval, rtol=1e-8, atol=1e-8, first_step=dt, max_step=dt)
        else:
            s = solve_ivp(y_derived_function, [
                t0, t_end], Y0, rtol=1e-8, atol=1e-8, first_step=dt, max_step=dt)
            return s.y

    # 多进程安全提示 since v0.1.1
    __I_AM_SURE_MY_CODE_CLOSED_IN_IF_NAME_EQUAL_MAIN: bool = False

    @classmethod
    def i_am_sure_my_code_closed_in_if_name_equal_main(cls):
        """
        多线程安全提示
        since v0.1.1
        """
        cls.__I_AM_SURE_MY_CODE_CLOSED_IN_IF_NAME_EQUAL_MAIN = True

    @classmethod
    def submit_process_task(cls,
                            task: Callable[..., T],
                            param_list: List[List],
                            concurrency_level: Optional[int] = None,
                            report: bool = True
                            ) -> List[T]:
        """
        提交任务多进程并行
        task 要运行的任务，是一个函数
        T 任务返回值
        param_list 任务参数数组，数组每个元素表示一个 task 的输出组合
        concurrency_level 并发等级，默认为 CPU 核心数


        因为 python 具有全局解释器锁，所以 CPU 密集任务无法使用线程加速，只能使用进程
        see https://www.cnblogs.com/dragon-123/p/10247252.html

        since v0.1.1
        """
        if not cls.__I_AM_SURE_MY_CODE_CLOSED_IN_IF_NAME_EQUAL_MAIN:
            raise PermissionError(
                "在使用CPU并行计算前，应确保你的脚本写在if __name__ == '__main__':"
                + "代码块内部，并显式调用BaseUtils.i_am_sure_my_code_closed_in_if_name_equal_main()函数"
            )

        if concurrency_level is None:
            concurrency_level = os.cpu_count()
        if report:
            print(f"处理并行任务，任务数目{len(param_list)}，并行等级{concurrency_level}")
        start = time.time()
        pool = multiprocessing.Pool(processes=concurrency_level)  # 开启一次性进程池
        r = pool.starmap(task, param_list)  # 执行任务
        pool.close()  # 停止接受任务
        pool.join()  # 等待完成
        if report:
            print(f"任务完成，用时{time.time() - start}秒")
        return r

    class Ellipse:
        """
        椭圆类
        Ax^2+Bxy+Cy^2=D
        """

        def __init__(self, A: float, B: float, C: float, D: float):
            self.A = float(A)
            self.B = float(B)
            self.C = float(C)
            self.D = float(D)

        def __eq__(self, other: 'BaseUtils.Ellipse') -> bool:
            """
            椭圆相等判断
            注意：因为 A B C D 具有放大不变性，所以判断为不等的椭圆，有可能是相等的

            since v0.1.4
            """
            return (
                self.A == other.A and
                self.B == other.B and
                self.C == other.C and
                self.D == other.D
            )

        def __hash__(self) -> int:
            """
            hash 方法，因为需要将椭圆当作字典的键

            since v0.1.4
            """
            return hash((self.A, self.B, self.C, self.D))

        def point_at(self, theta: float) -> P2:
            """
            原点出发，方向th弧度的射线和椭圆Ax^2+Bxy+Cy^2=D的交点
            Parameters
            ----------
            theta 弧度

            Returns 方向th弧度的射线和椭圆Ax^2+Bxy+Cy^2=D的交点
            -------

            """
            d = P2()

            while theta < 0:
                theta += 2 * math.pi

            while theta > 2 * math.pi:
                theta -= 2 * math.pi

            if BaseUtils.equal(theta, 0) or BaseUtils.equal(theta, 2 * math.pi):
                d.x = math.sqrt(self.D / self.A)
                d.y = 0

            if BaseUtils.equal(theta, math.pi):
                d.x = -math.sqrt(self.D / self.A)
                d.y = 0

            t = 0.0

            if 0 < theta < math.pi:
                t = 1 / math.tan(theta)
                d.y = math.sqrt(
                    self.D / (self.A * t * t + self.B * t + self.C))
                d.x = t * d.y

            if math.pi < theta < 2 * math.pi:
                theta -= math.pi
                t = 1 / math.tan(theta)
                d.y = -math.sqrt(self.D / (self.A * t *
                                           t + self.B * t + self.C))
                d.x = t * d.y

            return d

        # circumference 方法缓存
        CIRCUMFERENCE_CACHE: Dict['BaseUtils.Ellipse', float] = dict()

        @property
        def circumference(self) -> float:
            """
            计算椭圆周长
            Returns 计算椭圆周长

            refactor v0.1.4 添加缓存
            -------

            """
            c: float = BaseUtils.Ellipse.CIRCUMFERENCE_CACHE.get(self)

            if c is None:
                num: int = 3600 * 4
                c: float = 0.0
                for i in range(num):
                    c += (
                        self.point_at(2.0 * math.pi / float(num) * (i + 1))
                        - self.point_at(2.0 * math.pi / float(num) * (i))
                    ).length()

                BaseUtils.Ellipse.CIRCUMFERENCE_CACHE[self] = c

            return c

        def point_after(self, length: float) -> P2:
            """
            在椭圆 Ax^2+Bxy+Cy^2=D 上行走 length，返回此时的点
            规定起点：椭圆与X轴正方向的交点
            规定行走方向：逆时针
            Parameters
            ----------
            length 行走距离

            Returns 椭圆 Ax^2+Bxy+Cy^2=D 上行走 length，返回此时的点
            -------

            """
            step_theta = BaseUtils.angle_to_radian(0.05)
            theta = 0.0
            while length > 0.0:
                length -= (
                    self.point_at(theta + step_theta) - self.point_at(theta)
                ).length()

                theta += step_theta

            return self.point_at(theta)

        def uniform_distribution_points_along_edge(self, num: int) -> List[P2]:
            """
            返回椭圆圆周上均匀分布的 num 个点
            """
            points = []
            c = self.circumference
            for i in range(num):
                points.append(self.point_after(c / num * i))

            return points

        @classmethod
        def create_standard_ellipse(cls, a: float, b: float) -> "BaseUtils.Ellipse":
            """
            构建标准椭圆
            x**2/a**2+y**2/b**2 = 1

            则 Ax^2+Bxy+Cy^2=D 中
            A = b**2
            B = 0
            C = a**2
            D = a**2 * b**2

            since 0.1.4
            """
            return BaseUtils.Ellipse(A=b**2, B=0.0, C=a**2, D=(a**2) * (b**2))

    class Statistic:
        """
        统计器
        since v0.1.1
        refactor v0.1.3 增加 add_all 和 helf_width 方法
        """

        def __init__(self):
            self.__data: List[float] = []

        def add(self, val: float) -> 'BaseUtils.Statistic':
            """
            添加元素
            """
            self.__data.append(val)
            return self

        def add_all(self, vals: Iterable[float]) -> 'BaseUtils.Statistic':
            """
            添加多个元素
            """
            self.__data.extend(vals)
            return self

        def max(self) -> float:
            """
            最大值
            """
            return numpy.max(self.__data)

        def min(self) -> float:
            """
            最小值
            """
            return numpy.min(self.__data)

        def var(self) -> float:
            """
            方差
            """
            return numpy.var(self.__data)

        def average(self) -> float:
            """
            均值
            """
            return sum(self.__data) / len(self.__data)

        def helf_width(self) -> float:
            """
            半宽
            即 (max-min)/2
            这个方法用于求束斑大小
            """
            return (self.max()-self.min())/2

        def clear(self):
            """
            清空
            """
            self.__data: List[float] = []
            return self

    class Random:
        """
        产生随机分布的类
        包括以下分布
            uniformly_distributed_along_circumference          单位圆的圆周均匀分布
            uniformly_distributed_in_circle                    单位圆内均匀分布
            uniformly_distributed_at_spherical_surface         单位球面均匀分布
            uniformly_distributed_in_sphere                    单位球内均匀分布
            uniformly_distributed_along_elliptic_circumference 椭圆的圆周均匀分布
            uniformly_distributed_in_ellipse                   椭圆内均匀分布
            uniformly_distributed_at_ellipsoidal_surface       椭球球面均匀分布
            uniformly_distributed_in_ellipsoid                 椭球球内均匀分布
            uniformly_distributed_at_hyperespherical_surface   超球体表面均匀分布
            uniformly_distributed_in_hyperesphere              超球体内均匀分布
            uniformly_distributed_at_hypereellipsoidal_surface 超椭球体表面均匀分布
            uniformly_distributed_in_hypereellipsoid           超椭球体内均匀分布

            gauss                                              高斯分布 / 正态分布 2021年2月26日 新增
            gauss_multi_dimension                              多维无关高斯分布（标准椭球） 2021年2月26日 新增

        辅助函数
            hypersphere_volume                                 超球体体积
            hypersphere_area                                   超球体面积



        since v0.1.4
        """
        @classmethod
        def uniformly_distributed_along_circumference(cls) -> P2:
            """
            单位圆的圆周均匀分布点
            原理：生成 [0, 2π) 的均与分布点，就是圆的方位角 azimuth，再转为二维点
            """
            azimuth = 2.0 * math.pi * random.random()  # [0, 2π)
            return P2(math.cos(azimuth), math.sin(azimuth))

        @staticmethod
        def uniformly_distributed_in_circle() -> P2:
            """
            单位圆内均匀分布
            原理：生成两个 [-1, 1] 分布的点 x y，
                若 (x,y) 在圆内则返回，否则重试
            """
            while True:
                x = random.uniform(-1, 1)
                y = random.uniform(-1, 1)
                if x**2+y**2 <= 1:
                    return P2(x, y)

        @staticmethod
        def uniformly_distributed_at_spherical_surface() -> P3:
            """
            单位球面均匀分布
            原理：天顶角 zenith / θ 的取值范围为 [0, π]，
                当具体取值为 θ0 时，对应的圆半径为 sin(θ0)，则周长为 2πsin(θ0)
                因此生成两个随机数 θ/天顶角 和周长位置 a
                θ 的取值范围为 [0, π]
                a 的取值范围为 [0, 2π]
                若 a < 2πsin(θ0)，则 (θ, a) 是圆面上的点，输出
            """
            while True:
                zenith = random.uniform(0, math.pi)  # 天顶角
                a = random.uniform(0, 2*math.pi)
                if a < 2.0*math.pi*math.sin(zenith):
                    azimuth = a / math.sin(zenith)  # 方位角
                    return P3(
                        x=math.sin(zenith)*math.cos(azimuth),
                        y=math.sin(zenith)*math.sin(azimuth),
                        z=math.cos(zenith)
                    )

        @staticmethod
        def uniformly_distributed_in_sphere() -> P3:
            """
            单位球内均匀分布
            原理：产生三个随机数 x y z，为 [-1,1] 上均匀分布
                若这三个数对应的点在单位球内部，则输出
            """
            while True:
                x = random.uniform(-1, 1)
                y = random.uniform(-1, 1)
                z = random.uniform(-1, 1)
                if x**2+y**2+z**2 <= 1:
                    return P3(x, y, z)

        @staticmethod
        def uniformly_distributed_along_elliptic_circumference(a: float, b: float) -> P2:
            """
            椭圆的圆周均匀分布点
            原理：求椭圆周长 c0，生成 [0, c) 的均与分布点 c，求得方位角度 azimuth，再转为二维点

            椭圆必须是正椭圆
            a 为 x 轴方向轴长度
            b 为 y 轴方向轴长度
            """
            e = BaseUtils.Ellipse(A=1/(a**2), C=1/(b**2), B=0.0, D=1.0)
            c0 = e.circumference  # 椭圆周长

            c = random.uniform(0, c0)

            return e.point_after(c)

        @staticmethod
        def uniformly_distributed_in_ellipse(a: float, b: float) -> P2:
            """
            椭圆内均匀分布
            原理：生成两个 [-a, a] 和 [-b, b] 分布的点 x y，
                若 (x,y) 在椭圆内则返回，否则重试

            椭圆必须是正椭圆
            a 为 x 轴方向轴长度
            b 为 y 轴方向轴长度
            """
            while True:
                x = random.uniform(-a, a)
                y = random.uniform(-b, b)
                if (x**2)/(a**2)+(y**2)/(b**2) <= 1:
                    return P2(x, y)

        @staticmethod
        def uniformly_distributed_at_ellipsoidal_surface(a: float, b: float, c: float) -> P3:
            """
            椭球球面均匀分布
            原理：天顶角 zenith / θ 的取值范围为 [0, π]，
                当具体取值为 θ0 时，对应的椭圆周长可以计算，设为 c0，最大值为 c_max
                因此生成两个随机数 θ/天顶角 和周长位置 a
                θ 的取值范围为 [0, π]
                a 的取值范围为 [0, c_max]
                若 a < c0，则 (θ, a) 是椭圆面上的点，输出

            椭球必须是正椭球
            a 为 x 轴方向轴长度
            b 为 y 轴方向轴长度
            c 为 z 轴方向轴长度
            """
            # 椭球在 z=0 平面，即 xy 平面上的 椭圆
            e_xy = BaseUtils.Ellipse(A=1/(a**2), C=1/(b**2), B=0.0, D=1.0)

            # 椭球在 zy 平面上的椭圆，且横坐标为 z，纵坐标为 y
            e_zy = BaseUtils.Ellipse(A=1/(c**2), C=1/(b**2), B=0.0, D=1.0)

            # 椭球在 zx 平面上的椭圆，且横坐标为 z，纵坐标为 x
            e_zx = BaseUtils.Ellipse(A=1/(c**2), C=1/(a**2), B=0.0, D=1.0)

            # e_xy 椭圆的周长
            c_max = e_xy.circumference

            while True:
                # 天顶角 / 高度角
                zenith = random.uniform(0, math.pi)  # 天顶角

                # 可能的周长
                a = random.uniform(0, c_max)

                # 由 天顶角 和椭球交点产生的小椭圆 e_cur
                a_cur = e_zx.point_at(zenith).y  # ax
                b_cur = e_zy.point_at(zenith).y  # by

                # 小椭圆 e_cur
                e_cur = BaseUtils.Ellipse(
                    A=1/(a_cur**2), C=1/(b_cur**2), B=0.0, D=1.0)
                # 小椭圆周长
                c0_cur = e_cur.circumference

                # 如果 a 小于小椭圆周长，则 (zenith,a) 在椭球面上
                if a <= c0_cur:
                    # x y 坐标
                    p_xy = e_cur.point_after(a)
                    # z 坐标
                    pz = e_zx.point_at(zenith).x
                    return P3(p_xy.x, p_xy.y, pz)

        @staticmethod
        def uniformly_distributed_in_ellipsoid(a: float, b: float, c: float) -> P3:
            """
            椭球球内均匀分布
            原理：产生三个随机数 x y z，为 [-a,a] [-b,b] [-c,c]上均匀分布
                若这三个数对应的点在单位球内部，则输出

            椭球必须是正椭球
            a 为 x 轴方向轴长度
            b 为 y 轴方向轴长度
            c 为 z 轴方向轴长度

            since
            """
            while True:
                x = random.uniform(-a, a)
                y = random.uniform(-b, b)
                z = random.uniform(-c, c)
                if (x**2)/(a**2)+(y**2)/(b**2)+(z**2)/(c**2) <= 1:
                    return P3(x, y, z)

        @classmethod
        def hypersphere_volume(cls, d: int, r: float = 1.0) -> float:
            """
            超球体的体积
            https://baike.baidu.com/item/%E8%B6%85%E7%90%83%E9%9D%A2/4907511?fr=aladdin#2

            d 维度
            r 超球体半径
            """
            if isinstance(d, int):
                if d % 2 == 1:
                    # 维度为奇数
                    k = (d-1)//2
                    c = (2**d)*(math.factorial(k)) * \
                        (math.pi**k)/(math.factorial(d))
                    return c*(r**d)
                else:
                    # 维度为偶数
                    k = d//2
                    c = (math.pi**k)/(math.factorial(k))
                    return c*(r**d)
            else:
                raise ValueError(f"维度{d}必须是整数")

        @classmethod
        def hypersphere_area(cls, d: int, r: float = 1.0) -> float:
            """
            超球体的表面积
            https://baike.baidu.com/item/%E8%B6%85%E7%90%83%E9%9D%A2/4907511?fr=aladdin#2

            d 维度
            r 超球体半径
            """
            if isinstance(d, int):
                return cls.hypersphere_volume(d, r)*d/r
            else:
                raise ValueError(f"维度{d}必须是整数")

        @classmethod
        def uniformly_distributed_at_hyperespherical_surface(cls, d: int, r: float = 1.0) -> List[float]:
            """
            超球体面均匀分布
            递归计算

            d 维度
            r 超球体半径

            注意 2 维球体表面分布，指的是圆周分布
            """
            if isinstance(d, int):
                if d == 1:
                    # 一维直线
                    raise ValueError("一维球无表面")
                elif d == 2:
                    # 二维圆
                    p = cls.uniformly_distributed_along_circumference().change_length(r)
                    return [p.x, p.y]
                else:
                    # 高维
                    while True:
                        # 第一个维度分布
                        fisrt_dim = random.uniform(-r, r)
                        # 剩余维度，是一个 d-1 维的超球体，表面积最大为
                        area_max = cls.hypersphere_area(d-1, r)
                        # 表面积均匀分布
                        area_pick = random.uniform(-area_max, area_max)

                        # 实际上 fisrt_dim 确定后，d-1 维球体的半径为
                        r_sub = math.sqrt(r**2-fisrt_dim**2)  # bug fixed
                        # 这样实际的表面积是
                        area_real = cls.hypersphere_area(d-1, r_sub)
                        if area_pick <= area_real:
                            return [fisrt_dim] + cls.uniformly_distributed_at_hyperespherical_surface(d-1, r_sub)
            else:
                raise ValueError(f"维度{d}必须是整数")

        @classmethod
        def uniformly_distributed_at_hypereellipsoidal_surface(cls, axes: List[float]) -> List[float]:
            """
            超椭球球面均匀分布
            axes 为各个轴的半轴长，例如[3.5, 7.5, 3.5, 7.5, 0.08]
            !!! 注意：因为超椭球的表面积很难计算（椭圆积分），所以先当作超球体处理，然后各轴拉伸为超椭球
                这样的分布，不再是标准的均匀分布
            """
            dim = len(axes)
            p = cls.uniformly_distributed_at_hyperespherical_surface(dim)
            for i in range(dim):
                p[i] *= axes[i]
            return p

        @classmethod
        def uniformly_distributed_in_hypereellipsoid(cls, axes: List[float]) -> List[float]:
            """
            超椭球球内均匀分布
            axes 为各个轴的半轴长，例如[3.5, 7.5, 3.5, 7.5, 0.08]
            """
            dim = len(axes)
            while True:
                p = [random.uniform(-axes[i], axes[i]) for i in range(dim)]
                r = 0.0
                for i in range(dim):
                    r += (p[i]**2)/(axes[i]**2)
                if r <= 1:
                    return p

        @classmethod
        def uniformly_distributed_in_hyperesphere(cls, d: int, r: float = 1.0) -> List[float]:
            """
            超球体内均匀分布

            d 维度
            r 半径
            """
            return cls.uniformly_distributed_in_hypereellipsoid([r]*d)

        @classmethod
        def gauss(cls, mu: float = 0.0, sigma: float = 1.0) -> float:
            """
            高斯分布

            since v0.1.4
            """
            return random.gauss(mu, sigma)

        @classmethod
        def gauss_multi_dimension(cls, mu_list: List[float], sigma_list: List[float]) -> List[float]:
            """
            多维无关高斯分布

            since v0.1.4
            """
            len_mu = len(mu_list)
            len_sigma = len(sigma_list)

            if len_mu != len_sigma:
                raise ValueError(
                    "gauss_multi_dimension mu_list 和 sigma_list 维度不同一")

            return [cls.gauss(mu_list[i], sigma_list[i]) for i in range(len_mu)]
