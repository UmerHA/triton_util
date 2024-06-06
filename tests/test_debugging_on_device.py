import os
import importlib

import pytest
import unittest

import triton
import triton.language as tl

from triton_util.debugging import _test_pid_conds

@pytest.fixture(scope="class", autouse=True)
def set_env_and_reload(request):
    os.environ['TRITON_INTERPRET'] = '0'
    importlib.reload(triton)
    importlib.reload(tl)

class TestTritonUtil(unittest.TestCase):
    def test_test_pid_conds(self):
        self.assertTrue(_test_pid_conds(''))
        self.assertTrue(_test_pid_conds('>0', 1, 1))
        self.assertFalse(_test_pid_conds('>0', 0, 1))
        self.assertTrue(_test_pid_conds('=0,=1', 0, 1, 0))

if __name__ == '__main__':
    unittest.main()
