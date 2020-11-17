import time
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
from cctpy.cuda.cuda_dB import dB
from cctpy.cuda.cuda_magnet_at_point import magnet_at_point
from cctpy.cuda.cuda_run_one_line import particle_run_len_one_line
from cctpy.cuda.cuda_add3d_local_float_and_double import add3d_local_double, add3d_local_foult
from cctpy.cuda.cuda_compute_and_table import sin_sum_tb, sin_sum_compute
from cctpy.cct import SoleLayerCct
from cctpy.abstract_classes import LocalCoordinateSystem


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
            self.assertTrue(Equal.equal_float(length, Vectors.length(v), err=1e-5))

    def test_use_global(self):
        for ig in range(10):
            src = Scalar.random_float32()
            des = Scalar.empty_float32()
            use_global(src, des)
            self.assertTrue(Equal.equal_vector(src + 1, des, err=1e-7))

    def test_tB(self):
        p0 = Vectors.create_float32(1, 1, 1)
        p1 = Vectors.create_float32(1.1, 1.2, 1.3)
        p = Vectors.create_float32(0, 0, 0)
        ret = Vectors.empty_float32()

        dB(p0, p1, p, ret)

        self.assertTrue(
            Equal.equal_vector(ret, np.array([0.014429237347383132, -0.028858474694766215, 0.01442923734738309]),
                               err=1e-5))

    def test_magnet_at_point_part1(self):
        r: float = 0.01
        length: float = 0.1
        n: int = 20
        pas: int = 360
        ts = np.linspace(0, n * 2 * np.pi, n * pas)
        line = np.array([
            [r * np.cos(t), r * np.sin(t), t / (n * 2 * np.pi) * length] for t in ts
        ], dtype=np.float32)
        p = Vectors.create_float32(0, 0, 0)

        ret = Vectors.empty_float32()

        pas_len = Scalar.of_int32(ts.shape[0])
        print(f"len={ts.shape[0]} -- python")

        s = time.time()
        magnet_at_point(line, pas_len, Scalar.of_float32(10000), p, ret)
        e = time.time()
        print(f"time={e - s}")  # 1.0495378971099854s
        # print(ret) # [1.6659981e-02 9.6419232e-04 1.2504296e+00]

        self.assertTrue(Equal.equal_vector(
            ret, Vectors.create(0.016659991558021645, 9.641212628421269E-4, 1.2504328150541577), err=1e-5))

    def test_magnet_at_point_part2(self):
        r: float = 0.01
        length: float = 0.1
        n: int = 20
        pas: int = 360
        ts = np.linspace(0, n * 2 * np.pi, n * pas)
        line = np.array([
            [r * np.cos(t), r * np.sin(t), t / (n * 2 * np.pi) * length] for t in ts
        ], dtype=np.float32)
        p = Vectors.create_float32(0, 0, 0)

        cct = SoleLayerCct(line, 10000, LocalCoordinateSystem.global_coordinate_system())

        # [1.66599891e-02 9.64121190e-04 1.25043283e+00]
        s = time.time()
        m = cct.magnetic_field_at(p)
        e = time.time()
        print(f"time={e - s}")

        self.assertTrue(Equal.equal_vector(m, Vectors.create(
            0.016659991558021645, 9.641212628421269E-4, 1.2504328150541577), err=1e-6))

    def test_particle_run_len_one_line(self):
        r: float = 0.01
        length: float = 0.1
        n: int = 20
        pas: int = 360
        ts = np.linspace(0, n * 2 * np.pi, n * pas)
        line = np.array([
            [r * np.cos(t), r * np.sin(t), t / (n * 2 * np.pi) * length] for t in ts
        ], dtype=np.float32)

        p = Vectors.create_float32(0, 0, 0)
        v = Vectors.create_float32(0.0, 0.0, 1.839551780274753E8)
        run_mass = Scalar.of_float32(2.1182873748205775E-27)
        speed = Scalar.of_float32(1.839551780274753E8)
        len = Scalar.of_float32(0.1)

        pas_len = Scalar.of_int32(ts.shape[0])
        print(f"p={p},v={v}")

        s = time.time()
        particle_run_len_one_line(
            line, pas_len, Scalar.of_float32(10000), len, p, v, run_mass, speed
        )
        e = time.time()

        print(f"\n\n------------\n\ntime={e - s}\n\n---------------\n\n")

        print(f"p={p},v={v}")

        self.assertTrue(True)

    def test_float_double(self):
        f = Vectors.create_float32(0, 0, 0)
        d = Vectors.create(0, 0, 0)
        dd = Vectors.create(0, 0, 0)

        for i in range(2):
            t: float = np.sin(i) * (10 ** (i % 10 - 5))
            add3d_local_foult(f, Vectors.create_float32(t, t, t))
            add3d_local_double(d, Vectors.create_float32(t, t, t).astype(np.float64))
            add3d_local_double(dd, Vectors.create(t, t, t))
            print(f, d, dd)

        print('\n\n----------------------\n')
        print(f, d, dd)
        print('\n\n----------------------\n')

        self.assertTrue(True)

    def test_compute_and_table(self):
        sin_table = np.array([np.sin(i / 180 * np.pi) for i in range(0, 360)], dtype=np.float32)

        number = Scalar.of_int32(1000*1000)

        result_tb = Scalar.empty_float32()

        result_cm = Scalar.empty_float32()

        s = time.time()
        sin_sum_tb(sin_table, result_tb, number)
        print(f"\n\ntime_tb={time.time() - s}\n\n")

        s = time.time()
        sin_sum_compute(result_cm, number)
        print(f"\n\ntime_cm={time.time() - s}\n\n")

        print(f"table_sum={result_tb}, compute_sum={result_cm}")

        self.assertTrue(Equal.equal_vector(result_cm, result_tb,err=1))


if __name__ == '__main__':
    unittest.main()
