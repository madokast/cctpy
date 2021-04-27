"""
CCT 建模优化代码
P3 使用示例

作者：赵润晓
日期：2021年4月26日
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *


# P3 代表一个三维点，或者三维向量

# 构造一个三维点，或者三维向量
# 使用类直接构造点 (2,3,4)
point234 = P3(2, 3, 4)
point234 = P3(x=2, y=3, z=4)
print(point234)  # (2.0, 3.0, 4.0)
# 如果不指定，则返回零矢量
print("P3() =",P3()) # (0.0, 0.0, 0.0)
# 使用参数名传值时，没有传值的参数默认为 0
print("P3(y=2) =",P3(y=2)) # (0.0, 2.0, 0.0)

# 函数 length() 返回三维矢量的长度，或者三维点到原点的距离
print(point234.length())  # 5.385164807134504，即 √29

# 函数 normalize() 矢量长度归一化，返回新矢量
print("point234 归一化 =", point234.normalize()) # (0.3713906763541037, 0.5570860145311556, 0.7427813527082074)
print("point234 归一化后长度 =", point234.normalize().length()) # 1.0
print("point234 自身不变", point234) # (2.0, 3.0, 4.0)

# 函数 change_length(len) 改变矢量的长度，返回新矢量
print("point234 长度变为10 =", point234.change_length(10)) # (3.713906763541037, 5.570860145311556, 7.427813527082074)
print("point234 长度变为10后长度为 ", point234.change_length(10).length()) # 10.0
print("point234 自身不变", point234) # (2.0, 3.0, 4.0)

# 函数 copy()，点/矢量复制一份，两者独立
point234_copied = point234.copy()
print("point234 自身和复制", point234, point234_copied) # (2.0, 3.0, 4.0) (2.0, 3.0, 4.0)
point234_copied.x = 1000  # 修改复制
print("point234 自身和复制(复制的有修改)", point234, point234_copied) # (2.0, 3.0, 4.0) (1000, 3.0, 4.0)

# 函数 __add__()，矢量加法，通过 + 运算符使用
point111 = P3(1, 1, 1)
print("point234 + point111 =", point234+point111) # (3.0, 4.0, 5.0)

# 函数 __neg__()，矢量取反，通过一元运算符 - 使用
print("-point234 =", -point234) # (-2.0, -3.0, -4.0)

# 函数 __sub__()，矢量加法，通过 - 运算符使用
print("point234 - point111 =", point234-point111) # (1.0, 2.0, 3.0)

# 函数 __iadd__()，矢量原地加法，通过 += 运算符使用，矢量自身变化
point234_copied_for_iadd = point234.copy() # 因为矢量自身变化，所以先复制一份
print("point234_copied_for_iadd =",point234_copied_for_iadd) # (2.0, 3.0, 4.0)
point234_copied_for_iadd+=point111
print("point234_copied_for_iadd += point111 后",point234_copied_for_iadd) # (3.0, 4.0, 5.0)

# 函数 __isub__()，矢量原地加法，通过 -= 运算符使用，矢量自身变化
point234_copied_for_isub = point234.copy() # 因为矢量自身变化，所以先复制一份
print("point234_copied_for_isub =",point234_copied_for_isub) # (2.0, 3.0, 4.0)
point234_copied_for_isub-=point111
print("point234_copied_for_isub -= point111 后",point234_copied_for_isub) # (1.0, 2.0, 3.0)


# 函数 __mul__() 和 __rmul__()，矢量乘法，使用 * 运算符
# 根据因数的不同，分为 标量乘法 和 数量级 两种
# 1. 当其中一个因数为标量时，进行标量乘法，结果为矢量
print("point234 * 2 =",point234 * 2) # (4.0, 6.0, 8.0)
print("2 * point234 =",2 * point234) # (4.0, 6.0, 8.0)
# 2. 两个因数都是矢量时，进行数量级，结果为一个标量
print("point111 * point234 =",point111 * point234) # 9.0
# 其他情况不支持
try:
    [] * point111
except ValueError as ve:
    print("出现异常：",ve)

# 函数 __truediv__()，矢量除法，只支持矢量 v / 标量 a，实际运算为 v * (1/a)
print("point234 / 2 =",point234 / 2) # (1.0, 1.5, 2.0)
# 除 0 会报错
try:
    point234/0
except ValueError as ve:
    print("出现异常：",ve)

# 函数 __matmul__()，矢量外积/叉乘，使用 @ 符号使用
point100 = P3(x=1)
point010 = P3(y=1)
print("point100 @ point010 =",point100 @ point010) # (0.0, 0.0, 1.0)

# 函数 __str__() 和 __repr__()，将矢量转为字符粗，使用 print() 打印时自动调用
print("point234,point234.__str__(),point234.__repr__() 打印结果一致",point234,point234.__str__(),point234.__repr__())

# __eq__()，判断两个三维点/矢量是否相等，使用 == 符号时自动调用
# 每个元素默认容忍绝对误差 1e-6，即 (a,b,c)==(d,e,f) 表示 |a-d|<=1e-6 且 |b-e|<=1e-6 且 |c-f|<=1e-6
# 当使用 __eq__() 时，可以指定 err 绝对误差，msg 报错信息
# msg 的意义如下：
# 无论 msg 是否指定（传值），当判断结果为相等时，返回 True
# 若 msg 指定，则当判断结果为相等时，返回 True，判断结果为不相等时，爆出异常信息
point100_add_1e6 = P3(x=1+1e-6)
point100_add_2e6 = P3(x=1+2e-6)
print("point100 == point100_add_1e6 is",point100 == point100_add_1e6) # True
print("point100 == point100_add_2e6 is",point100 == point100_add_2e6) # False
try:
    point100.__eq__(3)
except ValueError as ve:
    print("遇到异常：",ve) # 3 不是 P3 不能进行相等判断
else:
    raise AssertionError("居然没有异常")
try:
    point100.__eq__(point100_add_2e6,msg=f"{point100}和{point100_add_2e6}不相等")
except AssertionError as ae:
    print("遇到异常：",ae) # (1.0, 0.0, 0.0)和(1.000002, 0.0, 0.0)不相等
else:
    raise AssertionError("居然没有异常")

# 类函数 x_direct(x) y_direct(y) z_direct(z)，分别获得 x y z 轴上的点，或者说平行于 x y z 轴的矢量
# 如果不传值，得到单位矢量
print("P3.x_direct() =",P3.x_direct()) # (1.0, 0.0, 0.0)
print("P3.y_direct() =",P3.y_direct()) # (0.0, 1.0, 0.0)
print("P3.z_direct() =",P3.z_direct()) # (0.0, 0.0, 1.0)
# 传值，则获得对应长度的矢量 / 对应位置的点
print("P3.x_direct(4) =",P3.x_direct(4)) # (4.0, 0.0, 0.0)
print("P3.y_direct(-5) =",P3.y_direct(-5)) # (0.0, -5.0, 0.0)
print("P3.z_direct(6) =",P3.z_direct(6)) # (0.0, 0.0, 6.0)

# 类函数 origin() 和 zeros() 都是获得 (0,0,0)，仅仅是理解是的区别
# origin() 一般看作原点，zeros() 一般看作零矢量
print("P3.origin() =",P3.origin()) # (0.0, 0.0, 0.0)
print("P3.zeros() =",P3.zeros()) # (0.0, 0.0, 0.0)
print("P3.origin()==P3.zeros() is",P3.origin()==P3.zeros()) # True

# 函数 to_list()，点/矢量 (x,y,z) 转为数组 [x,y,z]
print("point234.to_list() is",point234.to_list())

# 类函数 from_numpy_ndarry(array) 将 numpy 数组转为 P3 点或 P3 数组,根据 numpy 数组形状有不同的返回值
array123 = numpy.array([1,2,3])
array1_2_3 = numpy.array([[1],[2],[3]])
array123_456 = numpy.array([[1,2,3],[4,5,6]])
print(f"P3.from_numpy_ndarry({array123})={P3.from_numpy_ndarry(array123)}") # P3.from_numpy_ndarry([1 2 3])=(1.0, 2.0, 3.0)
print(f"P3.from_numpy_ndarry({array1_2_3})={P3.from_numpy_ndarry(array1_2_3)}") # P3.from_numpy_ndarry([[1] [2] [3]])=(1.0, 2.0, 3.0)
print(f"P3.from_numpy_ndarry({array123_456})={P3.from_numpy_ndarry(array123_456)}") # P3.from_numpy_ndarry([[1 2 3] [4 5 6]])=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]

# 函数 to_numpy_ndarry3()，将三维点/矢量转为 numpy 数组
# 可以使用 numpy_dtype 指定数据类型，如 numpy.float64 和 numpy.float32
print("point234.to_numpy_ndarry3() =",point234.to_numpy_ndarry3()) # [2. 3. 4.]

# 函数 to_p2() 将三维点/矢量 P3 转为二维点/矢量 P2，默认去除 z 方向即 (x,y,z) -> (x,y)
# 可以使用 transformation 指定转换方式
print("point234.to_p2() =",point234.to_p2()) # (2.0, 3.0)
print("point234.to_p2(指定转换方式) =",point234.to_p2(transformation= lambda p:P2(x=p.x+p.y+p.z))) # (9.0, 0.0)

# 函数 populate(another) 将 another 的值赋到自身中，自身发生变化
point234_for_populate = point234.copy() # 先复制一下，避免影响 point234
print("populate 前 point234_for_populate =",point234_for_populate) # (2.0, 3.0, 4.0)
point234_for_populate.populate(point111)
print("populate 后 point234_for_populate =",point234_for_populate) # (1.0, 1.0, 1.0)

# 类函数 random()，随机产生一个 P3，其中 x y z ∊ [0,1)
print("P3.random() =",P3.random()) # 每次运行值不同，例如 (0.5861806498869068, 0.04421660486896695, 0.8156823583814132)

# 类函数 as_p3(anything)，无实际意义的“类型转换”，用于代码提示
mixed_list = [1,2,3,P3()] # 创建一个混合数组
p = mixed_list[-1] # 拿出最后一个元素，实际上是 P3()
# 但是 p 没有代码提示，于是可以进行“类型转换”
p = P3.as_p3(p)
print("出现代码提示",p.to_list()) # [0.0, 0.0, 0.0]

