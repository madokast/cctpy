import unittest

import numpy as np

from cctpy.baseutils import Vectors, Equal, Scalar
from cctpy.cuda.cuda_cross import cross
from cctpy.cuda.cuda_dot_a_v import dot_a_v
from cctpy.cuda.cuda_dot_v_v import dot_v_v
from cctpy.cuda.cuda_add3d_local import add3d_local
from cctpy.cuda.cuda_add3d import add3d
from cctpy.cuda.cuda_neg3d import neg3d
from cctpy.cuda.cuda_copy3d import copy3d
from cctpy.cuda.cuda_length3d import len3d
from cctpy.cuda.cuda_global import use_global


class CudaTest(unittest.TestCase):
    def test_cross(self):
        for ig in range(10):
            a = Vectors.random_float32()
            b = Vectors.random_float32()
            ans_d = Vectors.empty_float32()
            ans_h = Vectors.cross(a, b)
            cross(a, b, ans_d)
            self.assertTrue(Equal.equal_vector(ans_h, ans_d, err=1e-5))

    def test_dot_a_v(self):
        for ig in range(10):
            a = Scalar.random_float32()
            v = Vectors.random_float32()
            ans_h = a * v
            dot_a_v(a, v)
            self.assertTrue(Equal.equal_vector(ans_h, v, err=1e-5))

    def test_dot_v_v(self):
        for ig in range(10):
            v1 = Vectors.random_float32()
            v2 = Vectors.random_float32()
            a = Scalar.empty_float32()
            a_h = np.inner(v1, v2)
            dot_v_v(v1, v2, a)
            self.assertTrue(Equal.equal_vector(a, a_h, err=1e-5))

    def test_add3d_local(self):
        for ig in range(10):
            a_local = Vectors.random_float32()
            a_local_h = a_local.copy()
            for j in range(10):
                b = Vectors.random_float32()
                a_local_h += b
                add3d_local(a_local, b)
                self.assertTrue(Equal.equal_vector(a_local, a_local_h, err=1e-6))

    def test_add3d(self):
        for ig in range(10):
            a = Vectors.random_float32()
            b = Vectors.random_float32()
            ret_d = Vectors.empty_float32()
            add3d(a, b, ret_d)
            ret_h = a + b
            self.assertTrue(Equal.equal_vector(ret_d, ret_h, err=1e-6))

    def test_neg3d(self):
        for ig in range(10):
            a = Vectors.random_float32()
            ret_h = -a
            neg3d(a)
            self.assertTrue(Equal.equal_vector(a, ret_h, err=1e-6))

    def test_copy3d(self):
        for ig in range(10):
            src = Vectors.random_float32()
            des = Vectors.empty_float32()
            copy3d(src, des)
            self.assertTrue(Equal.equal_vector(src, des, err=1e-6))

    def test_len3d(self):
        for ig in range(10):
            v = Vectors.random_float32()
            length = Scalar.empty_float32()
            len3d(v, length)
            self.assertTrue(Equal.equal_float(length, Vectors.length(v), err=1e-7))

    def test_use_global(self):
        for ig in range(10):
            src = Scalar.random_float32()
            des = Scalar.empty_float32()
            use_global(src, des)
            self.assertTrue(Equal.equal_vector(src + 1, des, err=1e-7))


if __name__ == '__main__':
    unittest.main()
