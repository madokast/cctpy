import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        arr = [1,2,3]
        b = [3,4]
        c = [0,-1]

        arr.extend(b)
        self.assertEqual(len(arr),5)

        arr.extend(c)
        self.assertEqual(arr[5],0)



if __name__ == '__main__':
    unittest.main()
