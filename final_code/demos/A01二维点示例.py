"""
P2 使用示例
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *


# P2 代表一个二维点，或者二维向量

# 构造一个二维点，或者二维向量
# 使用类直接构造点 (2,3)
point23 = P2(2, 3)
point23 = P2(x=2, y=3)
print("点point23 =", point23)
# xxx

# length()，获得二维向量的长度，以 (3,4) 为例
point34 = P2(x=3, y=4)
length_of_34 = point34.length()
print("length_of_34 =", length_of_34)

# normalize()，将二维矢量归一化，以 (3,0) 和 (1,1) 为例
point30 = P2(3, 0)
point11 = P2(1, 1)
point30_normalized = point30.normalize()
point11_normalized = point11.normalize()
print("point30_normalized =", point30_normalized)
print("point11_normalized =", point11_normalized)

# change_length()，改变二维矢量的长度，以 (3,0) 和 (1,1) 为例，长度分别变为 5 和 10√2
point30_length_5 = point30.change_length(5)
point11_length_10sqrt2 = point11.change_length(10*math.sqrt(2))
print("point30_length_5 =", point30_length_5)
print("point11_length_10sqrt2 =", point11_length_10sqrt2)

# 复制一个二维点，或者二维向量，以 (1,1) 为例
point11_copid = point11.copy()
print("point11_copid =", point11_copid)

# 矢量加法，使用加号或者 __add__() 函数，以 (3,0)+(1,1)为例
point30_add_11 = point30+point11
point30_add_11_another_way = point30.__add__(point11)
print("point30_add_11 =", point30_add_11)
print("point30_add_11_another_way =", point30_add_11_another_way)

# 矢量取反，使用符号 - 或者 __neg__() 函数，以 (3,0) 为例
point30_neg = - point30
point30_neg_another_way = point30.__neg__()
print("point30_neg =", point30_neg)
print("point30_neg_another_way =", point30_neg_another_way)

# 矢量相减，使用符号 - 或者 __sub__() 函数以 (3,0)-(1,1) 为例
point30_sub_11 = point30 - point11
point30_sub_11_another_way = point30.__sub__(point11)
print("point30_sub_11 =", point30_sub_11)
print("point30_sub_11_another_way =", point30_sub_11_another_way)

# 矢量原地相加，使用符号 += 或者 __iadd__() 函数，操作后矢量本身值发生变化，以 (3,0)+=(1,1) 为例
point30_for_iadd1 = point30.copy()  # 复制 (3,0)
point30_for_iadd2 = point30.copy()  # 复制 (3,0)
print("原地相加前 point30_for_iadd1=", point30_for_iadd1)
print("原地相加前 point30_for_iadd2=", point30_for_iadd2)
point30_for_iadd1 += point11
point30_for_iadd2.__iadd__(point11)
print("原地相加后 point30_for_iadd1=", point30_for_iadd1)
print("原地相加后 point30_for_iadd2=", point30_for_iadd2)

# 矢量原地相减，使用符号 -= 或者 __isub__() 函数，操作后矢量本身值发生变化，以 (3,0)-=(1,1) 为例
point30_for_isub1 = point30.copy()  # 复制 (3,0)
point30_for_isub2 = point30.copy()  # 复制 (3,0)
print("原地相减前 point30_for_isub1=", point30_for_isub1)
print("原地相减前 point30_for_isub2=", point30_for_isub2)
point30_for_isub1 -= point11
point30_for_isub2.__isub__(point11)
print("原地相减后 point30_for_isub1=", point30_for_isub1)
print("原地相减后 point30_for_isub2=", point30_for_isub2)

# 矢量旋转相关
# _rotation_matrix(phi) 获取旋转矩阵，这是一个 2*2 的矩阵。注意 phi 单位为弧度，以 30 度为例，使用 BaseUtils.angle_to_radian(30) 将角度值转为弧度制
rotation_matrix30 = P2._rotation_matrix(BaseUtils.angle_to_radian(30))  # 注意
rotation_matrix45 = P2._rotation_matrix(BaseUtils.angle_to_radian(45))  # 注意
print("旋转矩阵 30 度", rotation_matrix30)
print("旋转矩阵 45 度", rotation_matrix45)

# _matmul(matrix) ，2*2的矩阵和矢量相乘，得到一个新的矢量。以 (3,4) 为例
matrix1010 = [[1, 0], [0, 1]]  # 单位矩阵
matrix1000 = [[1, 0], [0, 0]]
matrix0001 = [[0, 0], [0, 1]]
point34_matrix1010 = point34._matmul(matrix1010)
point34_matrix1000 = point34._matmul(matrix1000)
point34_matrix0010 = point34._matmul(matrix0001)
print("matrix1010 × (3,4)", point34_matrix1010)
print("matrix1000 × (3,4)", point34_matrix1000)
print("matrix0001 × (3,4)", point34_matrix0010)

# rotate(phi)，矢量旋转，逆时针，返回新矢量，以 (1,0) 为例
point10 = P2(1, 0)
point10_rotate30 = point10.rotate(BaseUtils.angle_to_radian(30))
point10_rotate45 = point10.rotate(BaseUtils.angle_to_radian(45))
print("(1,0) 逆时针旋转 30 度", point10_rotate30)
print("(1,0) 逆时针旋转 45 度", point10_rotate45)

# angle_to_x_axis(), 获取矢量到 x 轴的夹角，弧度制
point10_rotate30_angle_to_x_axis = point10_rotate30.angle_to_x_axis()
point10_rotate45_angle_to_x_axis = point10_rotate45.angle_to_x_axis()
print("point10_rotate30_angle_to_x_axis =", BaseUtils.radian_to_angle(
    point10_rotate30_angle_to_x_axis))  # BaseUtils.radian_to_angle() 将弧度制转为角度值
print("point10_rotate45_angle_to_x_axis =",
      BaseUtils.radian_to_angle(point10_rotate45_angle_to_x_axis))

# __mul__() 矢量乘法，随乘法对象的不同有 点积 和 标量乘法 两种
# 使用符号 * 进行乘法运算
print("(3,4)*2 =", P2(3, 4)*2)  # 标量乘法 (3,4)*2 = (6.0, 8.0)
print("2*(3,4) =", 2*P2(3, 4))  # 标量乘法 2*(3,4) = (6.0, 8.0)
print("(3,4)*(1,2) =", P2(3, 4)*P2(1, 2))  # 点积 (3,4)*(1,2) = 11.0

# __truediv__() 除法运算，只能是 矢量/变量 这样的形式，实际计算为 矢量*(1/变量)
print("(3,4)/2 =", P2(3, 4)/2)  # (3,4)/2 = (1.5, 2.0)

# angle_to(another) 计算矢量到另一个矢量 another 的夹角，以逆时针计算，返回弧度
point10, point01 = P2(1, 0), P2(0, 1)
point10_to_point01 = BaseUtils.radian_to_angle(point10.angle_to(point01))
point01_to_point10 = BaseUtils.radian_to_angle(point01.angle_to(point10))
print("(1,0)到(0,1)的角度 =", point10_to_point01) # 90
print("(0,1)到(1,0)的角度 =", point01_to_point10) # 270

# to_p3(lambda) 将二维点/矢量转为三维点/矢量。默认 (x,y) 转为 (x,y,0)，可以使用 lambda 函数指定转换方式
print("(3,4).to_p3() =",point34.to_p3())
print("(3,4).to_p3(lambda) =",point34.to_p3(lambda p2:P3(p2.x,p2.y,p2.x+p2.y)))

# __str__() 和 __repr__()，将自身转为字符串，使用 print() 时自动调用。
print(point10) # (1.0, 0.0)
print(point10.__str__()) # (1.0, 0.0)
print(point10.__repr__()) # (1.0, 0.0)

# __eq__()，判断两个二维点/矢量是否相等，使用 == 符号时自动调用
# 每个元素默认容忍绝对误差 1e-6，即 (a,b)==(c,d) 表示 |a-c|<=1e-6 且 |b-d|<=1e-6
# 当使用 __eq__() 时，可以指定 err 绝对误差，msg 报错信息
# msg 的意义如下：
# 无论 msg 是否指定（传值），当判断结果为相等时，返回 True
# 若 msg 指定，则当判断结果为相等时，返回 True，判断结果为不相等时，爆出异常信息
point10_add_1e6 = P2(1+1e-6,0)
point10_add_2e6 = P2(1+2e-6,0)
print(point10==point10_add_1e6)
print(point10==point10_add_2e6)
try:
    point10.__eq__(3)
except ValueError as ve:
    print("遇到异常：",ve)
else:
    raise AssertionError("居然没有异常")
try:
    point10.__eq__(point10_add_2e6,msg=f"{point10}和{point10_add_2e6}不相等")
except AssertionError as ae:
    print("遇到异常：",ae)
else:
    raise AssertionError("居然没有异常")

# 静态函数 x_direct(x) 获得 x 轴上的一点/或者获得平行于 x 轴的矢量，即 y=0
# 如果不指定 x，则 x=1
print(P2.x_direct()) # (1.0, 0.0)
print(P2.x_direct(-10)) # (-10.0, 0.0)

# 静态函数 y_direct(y) 获得 y 轴上的一点/或者获得平行于 y 轴的矢量，即 x=0
# 如果不指定 y，则 y=1
print(P2.y_direct()) # (0.0, 1.0)
print(P2.y_direct(20)) # (0.0, 20.0)

# 静态函数 origin() 和 zeros()，返回值都是 (0,0)
# 仅仅 origin() 表示原点，zeros() 表示零矢量，意义上的区别
print(P2.origin()) # (0.0, 0.0)
print(P2.zeros()) # (0.0, 0.0)
print(P2.origin()==P2.zeros()) # True

# 函数 to_list()，把点/矢量 (x,y) 转为数组 [x,y]，以 (3,4) 为例
list34 = point34.to_list()
print(list34[0],list34[1]) # 3.0 4.0

# 静态函数 from_numpy_ndarry(ndarray) 将 numpy 的数组 ndarray 转为 P2 或者 P2 数组
# 当数组为 1*2 或 2*1 时，转为单个 P2 点
# 当数组为 n*2 转为 P2 数组
ndarray12 = numpy.array([1,2])
ndarray_1_2 = numpy.array([[1],[2]])
ndarray12_34 = numpy.array([[1,2],[3,4]])
ndarray123 = numpy.array([1,2,3])
print(P2.from_numpy_ndarry(ndarray12))
print(P2.from_numpy_ndarry(ndarray_1_2))
print(P2.from_numpy_ndarry(ndarray12_34))
try:
    print(P2.from_numpy_ndarry(ndarray123))
except Exception as e:
    print("遇到异常：",e)

# 静态函数 from_list(list) 将 list 数组转为 P2
# 如果 list 中元素为数字，则取前两个元素转为 P2
# 如果 list 中元素也是 list，则迭代进行
list12 = [1,2]
list123 = [1,2,3]
list12_34 = [[1,2],[3,4]]
list12_345 = [[1,2],[3,4,5]]
print(P2.from_list(list12)) # (1.0, 2.0)
print(P2.from_list(list123)) # (1.0, 2.0)
print(P2.from_list(list12_34)) # [(1.0, 2.0), (3.0, 4.0)]
print(P2.from_list(list12_345)) # [(1.0, 2.0), (3.0, 4.0)]

# 静态方法 extract(p2_list) 从 P2 数组 p2_list 中抽取 x 坐标和 y 坐标，返回两个数组
# 例如 p2_list = [(1,2),(3,4),(5,6)]
# 则返回两个数组 [1,3,5] 和 [2,4,6]
p2_list = [P2(1,2),P2(3,4),P2(5,6)]
x_list,y_list = P2.extract(p2_list)
print(x_list,y_list) # [1.0, 3.0, 5.0] [2.0, 4.0, 6.0]

# 静态方法 extract_x(p2_list) 和 extract_y(p2_list)从 P2 数组 p2_list 中抽取 x 坐标和 y 坐标
print(P2.extract_x(p2_list)) # [1.0, 3.0, 5.0]
print(P2.extract_y(p2_list)) # [2.0, 4.0, 6.0]

