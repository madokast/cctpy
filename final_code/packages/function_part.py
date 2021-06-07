
from packages.point import *
from packages.plot import *
from typing import TypeVar, Callable
_T = TypeVar("_T")


class Function_Part:
    """
    函数段
    即包含一个函数 func，和自变量的起始值和终止值


    since 2021年1月9日
    """

    def __init__(self, func: Callable[[float], _T], start: float, end: float, scale: float = 1.0) -> None:
        self.func = func
        self.start = start
        self.end = end
        self.scale = scale

        self.length = abs(start-end) * self.scale

        # forward 为正数，说明自变量向着增大的方向，即 end > start
        self.forward = start < end

    def valve_at(self, x: float, err=1e-6) -> _T:
        """
        注意，此时函数的起点变成了 0
        取值范围为 [0, self.length]
        """
        x = x/self.scale
        if x > self.length+err or x < -err:
            print(f"Function_Part：自变量超出范围, x={x}, length={self.length}")
        return self.func(self.start + (
            x if self.forward else (-x)
        ))

    def append(self, func: Callable[[float], _T], start: float, end: float, scale: float = 1.0) -> 'Function_Part':
        """
        原函数段尾加新的函数
        """
        appended = Function_Part(func, start, end, scale)

        def fun_linked(t):
            if t < self.length:
                return self.valve_at(t)
            else:
                return appended.valve_at(t-self.length)

        return Function_Part(fun_linked, 0, self.length+appended.length)

if __name__ == "__main__":
    if True:  # test Function_Part
        fp = Function_Part(lambda x: x, 5, 2)
        print(fp.valve_at(0))
        print(fp.valve_at(1))

        fp = fp.append(lambda x: x**2, 0, 5)
        fp = fp.append(lambda x: -x**2, 0, 5, 20)
        fp = fp.append(lambda x: x**2, 0, 5)
        fp = fp.append(lambda x: -x**2, 0, 5, 20)
        Plot2.plot([P2(x, fp.valve_at(x))
                    for x in BaseUtils.linspace(0, fp.length, 100)])
        Plot2.show()