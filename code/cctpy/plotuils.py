"""
绘图工具类
"""
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


class Plot3:
    INIT: bool = False
    ax = None

    @staticmethod
    def __init():
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        fig = plt.figure()
        Plot3.ax = fig.gca(projection='3d')
        Plot3.ax.grid(False)

        Plot3.INIT = True

    @staticmethod
    def plot3d(lines: List[Tuple[np.ndarray, str]]) -> None:
        if not Plot3.INIT:
            Plot3.__init()

        for lc in lines:
            x = lc[0][:, 0]
            y = lc[0][:, 1]
            z = lc[0][:, 2]
            Plot3.ax.plot(x, y, z, lc[1])

    @staticmethod
    def show():
        if not Plot3.INIT:
            raise RuntimeError("Plot3::请在show前调用plot3d")

        plt.show()


class Plot2:
    INIT = False

    @staticmethod
    def __init():
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        Plot2.INIT = True

    @staticmethod
    def plot2d(lines: List[Tuple[np.ndarray, str]]) -> None:
        if not Plot2.INIT:
            Plot2.__init()

        for lc in lines:
            x = lc[0][:, 0]
            y = lc[0][:, 1]
            plt.plot(x, y, lc[1])

    @staticmethod
    def plot2d_xy(x: np.ndarray, y: np.ndarray, describe='r') -> None:
        if not Plot2.INIT:
            Plot2.__init()

        plt.plot(x, y, describe)

    @staticmethod
    def show():
        if not Plot2.INIT:
            raise RuntimeError("Plot3::请在show前调用plot3d")

        plt.show()
