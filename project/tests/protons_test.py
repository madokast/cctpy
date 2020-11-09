import unittest

from cctpy.baseutils import Equal
from cctpy.constant import Protons


class ProtonsTest(unittest.TestCase):
    def test_static_mass_MeV(self):
        # print(Protons.STATIC_ENERGY_MeV) 938.2719367522984
        self.assertTrue(Equal.equal_float(Protons.STATIC_ENERGY_MeV, 938.2720813))


if __name__ == '__main__':
    unittest.main()
