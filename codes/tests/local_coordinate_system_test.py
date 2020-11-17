from cctpy.baseutils import Vectors, Equal
import unittest
import numpy as np
from cctpy.abstract_classes import LocalCoordinateSystem


class LocalCoordinateSystemTest(unittest.TestCase):

    def test_point_to_local_coordinate(self):
        for i in range(10):
            o = np.random.rand(3) * (i + 1)
            main = np.random.rand(3) * (i + 1)
            temp = np.random.rand(3) * (i + 1)
            second = np.cross(main, temp)
            lc = LocalCoordinateSystem(o, main, second)

            p = np.random.rand(3) * (i + 1)
            op = p - o
            zi = Vectors.normalize_self(main.copy())
            xi = Vectors.normalize_self(second.copy())
            yi = np.cross(zi, xi)

            x = np.inner(op, xi)
            y = np.inner(op, yi)
            z = np.inner(op, zi)

            self.assertTrue(Equal.equal_vector(
                lc.point_to_local_coordinate(p),
                np.array([x, y, z])
            ))

    def test_global_coordinate_system(self):
        """
        测试全局坐标系
        Returns None
        -------

        """
        gcs = LocalCoordinateSystem.global_coordinate_system()
        for i in range(10):
            v0 = np.random.rand(3)
            v = gcs.point_to_local_coordinate(v0)

            self.assertTrue(Equal.equal_vector(v0, v))

    def test_local_to_global_point(self):
        i = 0
        while i < 10:
            ol = np.random.randn(3)  # 任意坐标系的原点

            xl = np.random.randn(3)  # 任意坐标系的 x 方向
            yl = np.random.randn(3)  # 任意坐标系的 y 方向

            xl = Vectors.normalize_self(xl)
            yl = Vectors.normalize_self(yl)

            if Equal.equal_vector(xl, yl) or Equal.equal_vector(xl + yl, np.zeros(3)):
                # 如果 xl 和 yl 平行，则重新生成
                i -= 1
                continue

            zl = np.cross(xl, yl)
            zl = Vectors.normalize_self(zl)

            yl = -np.cross(xl, zl)  # 正交化

            lcs = LocalCoordinateSystem(ol, zl, xl)

            self.assertTrue(Equal.equal_vector(yl, lcs.YI))

            for ignore in range(10):
                pl = np.random.randn(3)  # 局部坐标系中任意一点

                pg = lcs.point_to_global_coordinate(pl)  # 转全局

                pl_1 = lcs.point_to_local_coordinate(pg)  # 转回来

                self.assertTrue(Equal.equal_vector(pl, pl_1))

            i += 1

    def test_local_to_global_line(self):
        i = 0
        while i < 10:
            ol = np.random.randn(3)  # 任意坐标系的原点

            xl = np.random.randn(3)  # 任意坐标系的 x 方向
            yl = np.random.randn(3)  # 任意坐标系的 y 方向

            xl = Vectors.normalize_self(xl)
            yl = Vectors.normalize_self(yl)

            if Equal.equal_vector(xl, yl) or Equal.equal_vector(xl + yl, np.zeros(3)):
                # 如果 xl 和 yl 平行，则重新生成
                i -= 1
                continue

            zl = np.cross(xl, yl)
            zl = Vectors.normalize_self(zl)

            yl = -np.cross(xl, zl)  # 正交化

            lcs = LocalCoordinateSystem(ol, zl, xl)

            self.assertTrue(Equal.equal_vector(yl, lcs.YI))

            for ignore in range(10):
                pl_start = np.random.randn(3)  # 局部坐标系中任意起点
                pl_end = np.random.randn(3)  # 局部坐标系中任意终点

                line_l = np.linspace(pl_start, pl_end, 50)

                line_g = lcs.line_to_global_coordinate(line_l)  # 转全局

                line_l_1 = lcs.line_to_local_coordinate(line_g)  # 转回来

                self.assertTrue(Equal.equal_vector(line_l, line_l_1))

            i += 1


if __name__ == '__main__':
    unittest.main(verbosity=2)
