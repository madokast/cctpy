"""
测试 numpy 中对象、函数的是否使用正确
"""

from baseutils import Equal
import unittest
import numpy as np


class NdArrayTest(unittest.TestCase):
    """
    测试对多维数组 ndarray 的理解是否正确
    """

    def test_array_len(self):
        """
        理解数组长度
        使用 arr.shape[0] 获得数组第一维的长度
        """
        arr = np.array([
            [1, 1, 1],
            [1, 2, 3],
            [3, 4, 5],
            [4, 3, 2]
        ])
        self.assertEqual(arr.shape[0], 4)

    def test_array_assign(self):
        """
        ndarry 数组用[:]赋值，会改变数组本身吗
        测试一下
        测试结果：不会
        """
        arr = np.empty((2, 3))
        arr_0 = arr
        arr[0, :] = np.array([1, 2, 3])
        arr_1 = arr
        arr[1, :] = np.array([1, 2, 3])
        arr_2 = arr
        self.assertTrue(arr_0 is arr_1)
        self.assertTrue(arr_0 is arr_2)

    def test_array_assign_2(self):
        """
        以下测试，说明 [0, :] 返回的是 ndarry 的一个视图
        所以修改 arr ，s 也会发生变化，以为是同一块内存空间
        """
        arr = np.empty((2, 3))
        arr_1 = arr[0, :]  # 获取切片
        arr[0, 0] = 1.  # 修改主体
        arr_2 = arr[0, :]  # 再次获取切片
        self.assertTrue(arr_1[0] == arr_2[0])

    def test_numpy_inner(self):
        a1 = np.array([1, 2, 3])
        a2 = np.array([4, 5, 6])
        self.assertTrue(Equal.equal_float(np.inner(a1, a2), 4 + 10 + 18))


if __name__ == "__main__":
    unittest.main(verbosity=1)
