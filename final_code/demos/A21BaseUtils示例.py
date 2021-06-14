"""
CCT 建模优化代码
A11 Baseutils 示例


作者：赵润晓
日期：2021年5月2日
"""

from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *


# Baseutils 工具类，提供了很多实用的方法

# 函数 equal()
# 可以用来判断两个数是否相等
print(BaseUtils.equal(1, 1))  # True
print(BaseUtils.equal(1, 2))  # False

# 还可以判断两个 P2/P3 对象是否相等
print(BaseUtils.equal(P2.zeros(), P2.origin()))  # True
print(BaseUtils.equal(P3(x=1), P3(y=1)))  # False

# err 设置允许的绝对误差
# 允许误差 0.2 时，1.0 和 1.1 相等
print(BaseUtils.equal(1.0, 1.1, err=0.2))  # True
# 对象 P2/P3 的判断，也会考虑绝对误差，只有当每个分量的差小于绝对误差，对象才相等
print(BaseUtils.equal(P2(1, 1), P2(0.9, 1.05), err=0.2))  # True

# msg ：当 msg 不为空时，且a不等于b，则抛出错误，错误信息为msg
# BaseUtils.equal(1,2,msg="1和2不相等")
# 出现以下异常信息
# Traceback (most recent call last):
#   File "c:/Users/madoka_9900/Documents/github/cctpy/final_code/demos/A21BaseUtils示例.py", line 34, in <module>
#     BaseUtils.equal(1,2,msg="1和2不相等")
#   File "c:\Users\madoka_9900\Documents\github\cctpy\final_code\packages\base_utils.py", line 60, in equal
#     raise AssertionError(msg)
# AssertionError: 1和2不相等

# 函数 linspace()
print(BaseUtils.linspace(1, 2, 2))
print(BaseUtils.linspace(1, 2, 3))
print(BaseUtils.linspace(1, 2, 4))
print(BaseUtils.linspace(1, 2, 5))
# [1.0, 2.0]
# [1.0, 1.5, 2.0]
# [1.0, 1.3333333333333333, 1.6666666666666665, 2.0]
# [1.0, 1.25, 1.5, 1.75, 2.0]

points = BaseUtils.linspace(P2.origin(), P2(3, 4), 20)
# Plot2.plot(points,describe='r.')
# Plot2.show()

print(BaseUtils.linspace(3, 1, 5))
# [3.0, 2.5, 2.0, 1.5, 1.0]
print(BaseUtils.linspace(1, -1, 5))
# [1.0, 0.5, 0.0, -0.5, -1.0]

a1 = math.pi
a2 = math.pi/2
a3 = math.pi/3
a4 = math.pi/4
b1 = BaseUtils.radian_to_angle(a1)
b2 = BaseUtils.radian_to_angle(a2)
b34 = BaseUtils.radian_to_angle([a3, a4])
print(b1, b2, b34)  # 180.0 90.0 [59.99999999999999, 45.0]
print(BaseUtils.angle_to_radian(b1))  # 3.141592653589793
print(BaseUtils.angle_to_radian(b2))  # 1.5707963267948966
print(BaseUtils.angle_to_radian(b34))
# [1.0471975511965976, 0.7853981633974483]

p1 = P2(0, 0)
p2 = P2(1, 1)
p3 = P2(1, 0)
center, r = BaseUtils.circle_center_and_radius(p1, p2, p3)
# 绘制这三个点
# Plot2.plot_p2s([p1,p2,p3],describe='k.')
# # 绘制圆心
# Plot2.plot_p2(center,describe='ro')
# Plot2.equal()
# Plot2.show()

xs = [1, 2, 3, 4, 5]
ys = [-2, 0, 5, 9, 20]
fit1 = BaseUtils.polynomial_fitting(xs, ys, 1)
fit2 = BaseUtils.polynomial_fitting(xs, ys, 2)
fit3 = BaseUtils.polynomial_fitting(xs, ys, 3)
print(fit1)
print(fit2)
print(fit3)
# [-0.7000000000000017, 2.6999999999999997]
# [-0.1999999999999946, 2.271428571428568, 0.07142857142857197]
# [-1.599999999999992, 4.238095238095229, -0.6785714285714245, 0.08333333333333283]

xs = [1, 2, 3, 4, 5]
ys = [-2, 0, 5, 9, 20]
fit1 = BaseUtils.polynomial_fitting(xs, ys, 1)
fit2 = BaseUtils.polynomial_fitting(xs, ys, 2)
fit3 = BaseUtils.polynomial_fitting(xs, ys, 3)
fun1 = BaseUtils.polynomial_fitted_function(fit1)
fun2 = BaseUtils.polynomial_fitted_function(fit2)
fun3 = BaseUtils.polynomial_fitted_function(fit3)
# 返回值是一个函数
print(type(fun1))  # <class 'function'>
print(fun1(1))  # -4.2
# 绘制拟合点
# Plot2.plot_xy_array(xs, ys, 'rx')
# # 绘制拟合函数
# Plot2.plot_function(fun1, start=0, end=6, describe='k-')
# Plot2.plot_function(fun2, start=0, end=6, describe='g-')
# Plot2.plot_function(fun3, start=0, end=6, describe='y-')
# Plot2.info()
# Plot2.legend("point",'linear','order2','order3')
# Plot2.show()


arr1 = [1,2,3,4,5]
arr2 = [P2(1,2),P2(3,4),P2(5,6)]
print(BaseUtils.list_multiply(arr1,2))
# [2, 4, 6, 8, 10]
print(BaseUtils.list_multiply(arr2,3))
# [(3.0, 6.0), (9.0, 12.0), (15.0, 18.0)]

y = lambda x:x*x
yd = BaseUtils.derivative(y)
# Plot2.plot_function(y,start=-4,end=4,describe='r-')
# Plot2.plot_function(yd,start=-4,end=4,describe='k-')
# Plot2.equal()
# Plot2.info()
# Plot2.show()

# interpolate_lagrange
y = BaseUtils.interpolate_lagrange(
    x = 2.2,
    x0 = 1, y0 = 2,
    x1 = 2, y1 = 4,
    x2 = 3, y2 = 6,
    x3 = 4, y3 = 8
)
print(y) # 4.4

# is_sorted
print(BaseUtils.is_sorted([1,2,3])) # True
print(BaseUtils.is_sorted([1,2,1])) # False

BaseUtils.print_traceback()
# <frame at 0x0000020DFFC8BBA0, file 'c:\\Users\\madoka_9900\\Documents\\github\\cctpy\\final_code\\packages\\base_utils.py', line 260, code print_traceback>
# <frame at 0x0000020DE8FDC440, file 'c:/Users/madoka_9900/Documents/github/cctpy/final_code/demos/A21BaseUtils示例.py', line 149, code <module>>

fun_1 = lambda :BaseUtils.print_traceback()
fun_2 = lambda :fun_1()
fun_3 = lambda :fun_2()
fun_3()
# <frame at 0x000001F07F47D9A0, file 'c:/Users/madoka_9900/Documents/github/cctpy/final_code/demos/A21BaseUtils示例.py', line 153, code <lambda>>
# <frame at 0x000001F07F47D810, file 'c:/Users/madoka_9900/Documents/github/cctpy/final_code/demos/A21BaseUtils示例.py', line 154, code <lambda>>
# <frame at 0x000001F07F47D680, file 'c:/Users/madoka_9900/Documents/github/cctpy/final_code/demos/A21BaseUtils示例.py', line 155, code <lambda>>
# <frame at 0x000001F06E3EC440, file 'c:/Users/madoka_9900/Documents/github/cctpy/final_code/demos/A21BaseUtils示例.py', line 156, code <module>>

yd = lambda t,Y:2*t
y3 = BaseUtils.runge_kutta4(0,3,0,yd,dt=0.01)
print(y3) # 8.999999999999984

yd = lambda t,Y:2*t
ts,Ys = BaseUtils.runge_kutta4(0,3,0,yd,dt=0.01,record=True)
# Plot2.plot_xy_array(ts,Ys)
# Plot2.equal()
# Plot2.info()
# Plot2.show()

yd = lambda t,Y:2*t
y3 = BaseUtils.solve_ode(0,3,0,yd,dt=0.1)
print(y3) # 9.0