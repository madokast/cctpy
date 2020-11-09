import unittest


class BulidinTest(unittest.TestCase):
    def test_int(self):
        self.assertEqual(5, int(5.0))
        self.assertEqual(5, int(5.1))
        self.assertEqual(5, int(5.5))
        self.assertEqual(5, int(5.9))


if __name__ == '__main__':
    unittest.main()
