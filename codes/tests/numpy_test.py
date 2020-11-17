"""
测试 numpy 中对象、函数的是否使用正确
"""

from cctpy.baseutils import Equal, Vectors
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
        """
        内积测试 矢量叉乘
        Returns
        -------

        """
        a1 = np.array([1, 2, 3])
        a2 = np.array([4, 5, 6])
        self.assertTrue(Equal.equal_float(np.inner(a1, a2), 4 + 10 + 18))

    def test_broadcast(self):
        """
        广播测试
        Returns
        -------

        """
        a1 = np.array([
            [1, 1, 0],
            [2, 2, 0]
        ])

        a2 = np.array([
            [0, 0, 5]
        ])

        a3 = np.array([
            [1, 1, 5],
            [2, 2, 5]
        ])

        self.assertTrue((a1 + a2 == a3).all())

    def test_add_update(self):
        """
        += 测试
        Returns
        -------

        """
        arr = np.array([1, 2, 3], dtype=np.float64)

        id_1 = id(arr)

        arr += np.array([1, 1, 1], dtype=np.float64)

        id_2 = id(arr)

        self.assertEqual(id_1, id_2)

        Equal.equal_vector(
            arr,
            np.array([2, 3, 4], dtype=np.float64)
        )

    def test_minus(self):
        """
        负号测试
        Returns
        -------

        """
        for ignore in range(10):
            a = np.random.randn(3)
            b = np.empty(3)
            b[0] = -a[0]
            b[1] = -a[1]
            b[2] = -a[2]
            self.assertTrue(Equal.equal_vector(-a, b))

    def test_power(self):
        self.assertTrue(Equal.equal_float(1.1 ** 50, 117.39085287969579))

    def test_each_diff(self):
        w = np.array([5, 6, 8, 2])

        w_diff = w[1:] - w[:-1]

        self.assertTrue(Equal.equal_vector(w_diff, np.array([1, 2, -6])))

    def test_each_diff_1(self):
        w = np.array([
            [1, 1, 1],
            [2, 3, 4],
            [4, 3, 2]
        ])

        w_diff = w[1:] - w[:-1]

        self.assertTrue(Equal.equal_vector(w_diff, np.array([
            [1, 2, 3],
            [2, 0, -2]
        ])))

    def test_each_cross(self):
        xyz = np.array([
            [1, 0, 0],  # x
            [0, 1, 0],  # y
            [0, 0, 1]  # z
        ])
        yzx = np.array([
            [0, 1, 0],  # y
            [0, 0, 1],  # z
            [1, 0, 0]  # x
        ])

        self.assertTrue(Equal.equal_vector(np.cross(xyz, yzx), np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])))

    def test_each_cross_1(self):
        for i in range(10):
            vs1 = np.random.rand(10, 3)
            vs2 = np.random.rand(10, 3)

            vs12_cross = np.cross(vs1, vs2)

            vs12_cross2 = np.empty((10, 3))

            for j in range(vs1.shape[0]):
                v1 = vs1[j, :]
                v2 = vs2[j, :]
                vs12_cross2[j, :] = np.cross(v1, v2)

            self.assertTrue(Equal.equal_vector(vs12_cross, vs12_cross2))

    def test_each_diff_2(self):
        w = np.array([
            [1, 1, 1],
            [2, 3, 4],
            [4, 3, 2]
        ])

        w_diff_3 = 3 * (w[1:] - w[:-1])

        self.assertTrue(Equal.equal_vector(w_diff_3, np.array([
            [3, 6, 9],
            [6, 0, -6]
        ])))

    def test_each_diff_3(self):
        w = np.array([
            [1, 1, 1],
            [2, 3, 4],
            [4, 3, 2]
        ], dtype=np.float64)

        w_mid = 0.5 * (w[1:] + w[:-1])

        self.assertTrue(Equal.equal_vector(w_mid, np.array([
            [1.5, 2, 2.5],
            [3, 3, 3]
        ])))

    def test_broadcast_1(self):
        w = np.array([
            [1, 1, 1],
            [2, 3, 4],
            [4, 3, 2]
        ], dtype=np.float64)

        v = Vectors.create(10, 20, 30)

        self.assertTrue(Equal.equal_vector(v - w, np.array([
            [10 - 1, 20 - 1, 30 - 1],
            [10 - 2, 20 - 3, 30 - 4],
            [10 - 4, 20 - 3, 30 - 2]
        ], dtype=np.float64)))

    def test_broadcast_len(self):
        w = np.array([
            [1, 1, 1],  # 1.732
            [2, 3, 4],  # 5.38
            [4, 3, 2]
        ], dtype=np.float64)

        w_len = np.linalg.norm(w, ord=2, axis=1)

        self.assertTrue(Equal.equal_vector(w_len, np.array([1.73205081, 5.38516481, 5.38516481]), err=1e-5))

        w_len = w_len ** 3

        self.assertTrue(
            Equal.equal_vector(w_len, np.array([1.73205081 ** 3, 5.38516481 ** 3, 5.38516481 ** 3]), err=1e-5))

    def test_sum(self):
        w = np.array([
            [1, 1, 1],
            [2, 3, 4],
            [0.1, 0.2, 0.3]
        ], dtype=np.float64)

        w_sum = np.sum(w, axis=0)

        self.assertTrue(Equal.equal_vector(w_sum, np.array([3.1, 4.2, 5.3])))

    def test_multi(self):
        w = np.array([
            [1, 1, 1],
            [2, 3, 4],
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3]
        ], dtype=np.float64)

        v = np.array([1, 2, 3, 4], dtype=np.float64).reshape((4, 1))

        wv = np.multiply(w, v)

        self.assertTrue(Equal.equal_vector(
            wv,
            np.array(
                [[1., 1., 1.],
                 [4., 6., 8.],
                 [0.3, 0.6, 0.9],
                 [0.4, 0.8, 1.2]]
            )
        ))


    def test_cut(self):
        w = np.array([
            [1, 1, 1],
            [2, 3, 4],
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3]
        ], dtype=np.float64)

        wc = w[:,0]

        self.assertTrue(Equal.equal_vector(
            wc,
            np.array([1,2,0.1,0.1])
        ))

    def test_vstack(self):
        a = Vectors.create(1,2,3)
        b = Vectors.create(2,3,4)

        ab = np.column_stack((a,b))

        print(ab)

    def test_copy(self):
        a = Vectors.create_float32(1, 2, 3)
        b = Vectors.create_float32(2, 3, 4)

        data = np.empty((6,),dtype = np.float32)

        data[0:3] = a
        data[3:6] = b

        print(data)

        self.assertTrue(Equal.equal_vector(data,np.array([1., 2., 3. ,2. ,3. ,4.],dtype=np.float32)))

if __name__ == "__main__":
    unittest.main(verbosity=2)
