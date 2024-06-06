import os
import importlib

import pytest

import triton
import triton.language as tl

class TestLoading:
    '''Test if reloading triton with current TRITON_INTERPRET works, which is needed for testing both interpreter and regular mode.'''

    def test_interpreter_mode(self, triton_interpret):
        assert os.environ.get('TRITON_INTERPRET', '0') == triton_interpret, f'TRITON_INTERPRET is not {triton_interpret}'

        @triton.jit
        def some_kernel(): pass
        
        jit_fn_type = triton.runtime.interpreter.InterpretedFunction if triton_interpret == '1' else triton.runtime.jit.JITFunction

        assert isinstance(some_kernel, jit_fn_type), f'kernel was not jitted into expected type ({type(some_kernel)} instead of {jit_fn_type}), so triton loaded in wrong mode.'

if __name__ == '__main__':
    pytest.main()
