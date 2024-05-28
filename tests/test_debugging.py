
import unittest

from triton_util.debugging import _test_pid_conds

class TestTritonUtil(unittest.TestCase):
    def test_test_pid_conds(self):
        self.assertTrue(_test_pid_conds(''))
        self.assertTrue(_test_pid_conds('>0', 1, 1))
        self.assertFalse(_test_pid_conds('>0', 0, 1))
        self.assertTrue(_test_pid_conds('=0,=1', 0, 1, 0))

if __name__ == '__main__':
    unittest.main()
