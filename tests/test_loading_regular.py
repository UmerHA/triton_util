import os
import importlib

import pytest
import unittest

import triton
import triton.language as tl

@pytest.fixture(scope="class", autouse=True)
def set_env_and_reload(request):
    os.environ['TRITON_INTERPRET'] = '0'
    importlib.reload(triton)
    importlib.reload(tl)

class TestCase(unittest.TestCase):
    '''Test if reloading triton with current TRITON_INTERPRET works, which is needed for testing both interpreter and regular mode.'''
    def test_interpreter_mode(self):
        self.assertEqual(os.environ.get('TRITON_INTERPRET', '0'), '0', 'TRITON_INTERPRET is not 0')

        @triton.jit
        def some_kernel(): pass

        self.assertIsInstance(some_kernel, triton.runtime.jit.JITFunction, 'kernel was not jitted into JITFunction, so triton loaded in interpreter mode.')

if __name__ == '__main__':
    unittest.main()
