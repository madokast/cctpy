"""
CCT 建模优化代码
绘图工具 Plot2

作者：赵润晓
日期：2021年5月9日
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *

# Plot2 用于绘制二维图形，如粒子 x/y 方向轨迹、设计轨道等
# 绘图工具的使用方法很简单
# 第一步，Plot2.plot_xxx() 绘制图形，底层都是 plt.plot()
# 第二步，Plot2.show() 显示图片，底层即 plt.show()

# 函数 plot_xy(x,y,describe) 绘制一个点
# 其中点的横坐标为 x，纵坐标为 y，describe 描述点的颜色/样式
# describe 的具体用法见 https://www.biaodianfu.com/matplotlib-plot.html 
# 这里给出最常用的颜色/样式
# k 黑色
# r 红色
# g 绿色
# b 蓝色
# y 黄色
# . 点，即多个数据点绘制时，每个数据点绘制为点
# - 实线，即多个数据点绘制时，按顺序用实线连接
# -- 虚线，即多个数据点绘制时，按顺序用虚线连接

# 下面绘制点 (2,3)，可以看到默认情况下 describe="r."，即红色点
Plot2.plot_xy(2,3)
Plot2.show()
