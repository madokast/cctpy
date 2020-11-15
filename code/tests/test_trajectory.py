import unittest

from cctpy.abstract_classes import *
from cctpy.plotuils import *
from cctpy.baseutils import *


class TrajectoryTestCase(unittest.TestCase):
    def test_straight(self):
        s = StraightLine2(1, np.array([1., 1.]), np.array([0., 0.]))
        self.assertTrue(Equal.equal_float(s.get_length(), 1))

    def test_tr_01(self):
        t = Trajectory(StraightLine2(2.0, Vectors.create(1, 0), Vectors.create(0, 0))) \
            .add_arc_line(0.95, False, 22.5) \
            .add_strait_line(1.5) \
            .add_arc_line(0.95, False, 22.5) \
            .add_strait_line(2.0 + 2.2) \
            .add_arc_line(0.95, True, 67.5) \
            .add_strait_line(1.5) \
            .add_arc_line(0.95, True, 67.5) \
            .add_strait_line(2.2)

        self.assertTrue(Equal.equal_vector(t.direct_at_end(), Vectors.create(0, -1)))

        # Plot3.plot3d(t.line_and_color())
        # Plot3.show()

    def test_tr_02(self):
        """
        彩蛋，把绘图代码注释取消即可
        Returns
        -------

        """
        c1 = Trajectory(
            StraightLine2(0.01, Vectors.create(0, 1), Vectors.create(0, 0))
        ).add_arc_line(
            1, True, 135
        ).add_arc_line(
            0.01, True, 90
        ).add_strait_line(
            0.1
        ).add_arc_line(
            0.01, True, 90
        ).add_arc_line(
            0.9, False, 360 - 90
        ).add_arc_line(
            0.01, True, 90
        ).add_strait_line(
            0.1
        ).add_arc_line(
            0.01, True, 90
        ).add_arc_line(
            1, True, 135
        )

        c2 = c1 + Vectors.create(3, 0)

        t = Trajectory(StraightLine2(
            0.8, Vectors.create(1, 0), Vectors.create(6, 1)
        )).add_arc_line(
            0.01, True, 90
        ).add_strait_line(
            0.2
        ).add_arc_line(
            0.01, True, 90
        ).add_strait_line(
            0.7
        ).add_arc_line(
            0.01, False, 90
        ).add_strait_line(
            1.7
        ).add_arc_line(
            0.01, True, 90
        ).add_strait_line(
            0.2
        ).add_arc_line(
            0.01, True, 90
        ).add_strait_line(
            1.7
        ).add_arc_line(
            0.01, False, 90
        ).add_strait_line(
            0.7
        ).add_arc_line(
            0.01, True, 90
        ).add_strait_line(
            0.2
        ).add_arc_line(
            0.01, True, 90
        ).add_strait_line(
            0.8
        ) + Vectors.create(0.5, 0)

        # Plot3.plot3d(c1.line_and_color('r'))
        # Plot3.plot3d(c2.line_and_color('b'))
        # Plot3.plot3d(t.line_and_color('g'))

        # Plot3.show()

        print(f"CCT={c1.get_length() + c2.get_length() + t.get_length()}")

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
