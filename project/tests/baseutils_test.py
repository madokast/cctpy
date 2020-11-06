from project.baseutils import *
import unittest
import numpy as np


class BaseUtilsTest(unittest.TestCase):
    def test_equal_float(self):
        self.assertTrue(equal_float(1., 1., 1e-5))
        self.assertTrue(equal_float(np.sqrt(2), np.sqrt(2), 1e-5))
        self.assertTrue(equal_float(np.sqrt(2), np.sqrt(2) + 1e-6, 1e-5))

    def test_length(self):
        self.assertEqual(length(np.array([1., 1.])), np.sqrt(2))
        self.assertEqual(length(np.array([1., 1., 1.])), np.sqrt(3))

    def test_update_length(self):
        arr1 = np.array([1.])
        update_length(arr1, 2)
        arr2 = np.array([2.])
        self.assertTrue(equal_vector(arr1, arr2, 1e-10))

        arr1 = np.array([1., 1.])
        update_length(arr1, 2)
        arr2 = np.array([np.sqrt(2), np.sqrt(2)])
        self.assertTrue(equal_vector(arr1, arr2, 1e-10))

    def test_normalize_locally(self):
        arr = np.array([1., 1.])
        normalize_locally(arr)
        self.assertTrue(equal_float(length(arr),1.))

        arr = np.array([23., 12.])
        normalize_locally(arr)
        self.assertTrue(equal_float(length(arr), 1.))

        arr = np.array([0, -1.])
        normalize_locally(arr)
        self.assertTrue(equal_float(length(arr), 1.))


if __name__ == '__main__':
    unittest.main(verbosity=1)
