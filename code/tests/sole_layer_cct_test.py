import unittest

import numpy as np

from cctpy.abstract_classes import LocalCoordinateSystem
from cctpy.baseutils import Vectors, Equal
from cctpy.cct import SoleLayerCct
from cctpy.constant import MM, ORIGIN3, XI, ZI, YI
from cctpy.plotuils import Plot3, Plot2
import time


class SoleLayerCctCase(unittest.TestCase):
    def test_solenoid(self):
        """
        测试螺线管磁场
        Returns
        -------

        """
        winding_num = 20

        r = 10 * MM

        length = 0.1

        path = np.array([
            [r * np.sin(t), r * np.cos(t), t / (winding_num * 2 * np.pi) * length]
            for t in np.linspace(0, winding_num * 2 * np.pi, winding_num * 360)
        ])

        cct = SoleLayerCct(path, 1000, LocalCoordinateSystem.global_coordinate_system())

        p0 = cct.magnetic_field_at(Vectors.create(0, 0, 0))
        p1 = cct.magnetic_field_at(Vectors.create(0, 0, 0.05))
        p2 = cct.magnetic_field_at(Vectors.create(0, 0.001, 0.05))

        self.assertTrue(
            Equal.equal_vector(p0, np.array([-9.641213222579932E-5, -0.0016659990498348225, -0.1250432833225887])))
        self.assertTrue(
            Equal.equal_vector(p1, np.array([-7.496745957474563E-4, -3.7777669447541795E-19, -0.24645342280018406])))
        self.assertTrue(
            Equal.equal_vector(p2, np.array([-7.128547526957852E-4, -6.352747104407253E-20, -0.2464630712060889])))

    def test_speed(self):
        passed = True
        if not passed:
            start = time.time()

            winding_num = 1000

            r = 10 * MM

            length = 0.1

            path = np.array([
                [r * np.sin(t), r * np.cos(t), t / (winding_num * 2 * np.pi) * length]
                for t in np.linspace(0, winding_num * 2 * np.pi, winding_num * 360)
            ])

            cct = SoleLayerCct(path, 1000, LocalCoordinateSystem.global_coordinate_system())

            line = np.linspace(
                Vectors.create(0, 0, -0.1),
                Vectors.create(0, 0, 0.2),
                7105
            )

            bz = cct.magnetic_field_along(line)[:, 2]

            Plot2.plot2d_xy(line[:, 2], bz)

            end = time.time()  # 417.59825801849365

            print("\n-----------------------------------\n", end - start, "\n----------------")

            Plot2.show()


if __name__ == '__main__':
    unittest.main()
