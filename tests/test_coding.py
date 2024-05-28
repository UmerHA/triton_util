
import unittest

from triton_util.coding import cdiv

class TestTritonUtil(unittest.TestCase):
    
    def test_cdiv(self):
        self.assertEqual(cdiv(10, 2), 5)
        self.assertEqual(cdiv(10, 3), 4)

if __name__ == '__main__':
    unittest.main()
