import unittest

import numpy as np

from cctpy.baseutils import Ellipse, Equal, Vectors


class EllipseTest(unittest.TestCase):
    def test_point_at(self):
        e = Ellipse(5, 6, 7, 8)
        p = e.point_at(9)
        self.assertTrue(Equal.equal_vector(p, np.array([-1.4668197123719513, 0.6634655254837646])))

    def test_circumference(self):
        e = Ellipse(5, 6, 7, 8)
        self.assertTrue(Equal.equal_float(e.circumference, 8.37790879394205))

    def test_point_after_1(self):
        e = Ellipse(51, 61, 71, 81)
        p1 = e.point_after(10)

        self.assertTrue(Equal.equal_vector(p1, np.array([0.0303850368280807, 1.054820048197822])))

    def test_point_after_2(self):
        e = Ellipse(51, 61, 71, 81)
        p2 = e.point_after(20)

        self.assertTrue(Equal.equal_vector(p2, np.array([-1.4268778700102562, 0.8455347803217155])))

    def test_point_after_3(self):
        e = Ellipse(51, 61, 71, 81)
        p3 = e.point_after(30)

        self.assertTrue(Equal.equal_vector(p3, np.array([-0.7745589880016186, -0.5731435014508517])))

    def test_1(self):
        e = Ellipse(1, 2, 3, 4)
        c = e.circumference

        c1 = 0.0

        ts = np.linspace(0, 2 * np.pi, 3600 * 4).tolist()
        # 12.113380275260914 12.118307671053898
        # 12.113380275260914 12.113874246192136
        # 12.113380275260914 12.113430052414605
        for t in ts[:-1]:
            c1 += Vectors.length(
                e.point_at(t) - e.point_at(t + (ts[1] - ts[0]))
            )

        # print(c, c1)
        self.assertTrue(Equal.equal_float(c, c1))


if __name__ == '__main__':
    unittest.main()
