import os
import importlib

import pytest

import triton
import triton.language as tl

from triton_util.debugging import _test_pid_conds

class TestDebuggingUtils:
    def test_test_pid_conds(self):
        assert _test_pid_conds('')
        assert _test_pid_conds('>0', 1, 1)
        assert not _test_pid_conds('>0', 0, 1)
        assert _test_pid_conds('=0,=1', 0, 1, 0)

if __name__ == '__main__':
    pytest.main()
