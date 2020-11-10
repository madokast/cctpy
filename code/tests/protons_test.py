import unittest

from cctpy.baseutils import Equal
from cctpy.constant import Protons


class ProtonsTest(unittest.TestCase):
    def test_static_mass_MeV(self):
        # print(Protons.STATIC_ENERGY_MeV) 938.2719367522984
        self.assertTrue(Equal.equal_float(Protons.STATIC_ENERGY_MeV, 938.2720813))

    def test_STATIC_ENERGY_J(self):
        self.assertTrue(Equal.equal_float(Protons.STATIC_ENERGY_J, 1.503277361017269E-10))

    def test_STATIC_ENERGY_eV(self):
        self.assertTrue(Equal.equal_float(Protons.STATIC_ENERGY_eV, 9.382719665019624E8, err=0.0001 * 1e8))

    def test_STATIC_ENERGY_MeV(self):
        self.assertTrue(Equal.equal_float(Protons.STATIC_ENERGY_MeV, 938.2720813, err=1e-6))

    def test_get_total_energy_MeV(self):
        self.assertTrue(Equal.equal_float(Protons.get_total_energy_MeV(250), 250 + Protons.STATIC_ENERGY_MeV))

    def test_get_kinetic_energy_MeV_after_momentum_dispersion(self):
        self.assertTrue(Equal.equal_float(Protons.get_kinetic_energy_MeV_after_momentum_dispersion(1000, 0.01),
                                          1014.8580212819448, err=1))

    def test_(self):
        self.assertTrue(Equal.equal_float(Protons.convert_momentum_dispersion_to_energy_dispersion(0.01, 250),
                                          0.017896104536270847, err=0.001))


if __name__ == '__main__':
    unittest.main()
