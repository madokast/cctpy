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


if __name__ == '__main__':
    unittest.main(verbosity=2)
