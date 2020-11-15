from cctpy.baseutils import Vectors, Equal, Debug
import unittest
import numpy as np


class BaseUtilsTest(unittest.TestCase):
    def test_equal_float(self):
        self.assertTrue(Equal.equal_float(1., 1., 1e-5))
        self.assertTrue(Equal.equal_float(np.sqrt(2), np.sqrt(2), 1e-5))
        self.assertTrue(Equal.equal_float(np.sqrt(2), np.sqrt(2) + 1e-6, 1e-5))

    def test_length(self):
        self.assertEqual(Vectors.length(np.array([1., 1.])), np.sqrt(2))
        self.assertEqual(Vectors.length(np.array([1., 1., 1.])), np.sqrt(3))

    def test_update_length(self):
        arr1 = np.array([1.])
        Vectors.update_length(arr1, 2)
        arr2 = np.array([2.])
        self.assertTrue(Equal.equal_vector(arr1, arr2, 1e-10))

        arr1 = np.array([1., 1.])
        Vectors.update_length(arr1, 2)
        arr2 = np.array([np.sqrt(2), np.sqrt(2)])
        self.assertTrue(Equal.equal_vector(arr1, arr2, 1e-10))

    def test_normalize_locally(self):
        arr = np.array([1., 1.])
        Vectors.normalize_self(arr)
        self.assertTrue(Equal.equal_float(Vectors.length(arr), 1.))

        arr = np.array([23., 12.])
        Vectors.normalize_self(arr)
        self.assertTrue(Equal.equal_float(Vectors.length(arr), 1.))

        arr = np.array([0, -1.])
        Vectors.normalize_self(arr)
        self.assertTrue(Equal.equal_float(Vectors.length(arr), 1.))

    def test_rotate_self_z_axis(self):
        v2 = np.array([2.0, 3.0])
        r1 = Vectors.rotate_self_z_axis(v2.copy(), 0.1)
        r2 = Vectors.rotate_self_z_axis(v2.copy(), 0.2)
        r3 = Vectors.rotate_self_z_axis(v2.copy(), -0.1)
        r4 = Vectors.rotate_self_z_axis(v2.copy(), 1.0)

        # print(r1, r2, r3,r4)

        self.assertTrue(Equal.equal_vector(r1, np.array([1.6905080806155672, 3.184679329127734])))
        self.assertTrue(Equal.equal_vector(r2, np.array([1.3641251632972997, 3.3375383951138478])))
        self.assertTrue(Equal.equal_vector(r3, np.array([2.2895085804965363, 2.785345662540421])))
        self.assertTrue(Equal.equal_vector(r4, np.array([-1.4438083426874098, 3.3038488872202123])))

    def test_debug(self):
        Debug.print_traceback()


if __name__ == '__main__':
    unittest.main(verbosity=1)
