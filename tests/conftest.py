import os
import importlib

import pytest

import triton
import triton.language as tl

@pytest.fixture(scope='class', params=['0', '1']) # Run tests in regular mode (TRITON_INTERPRET=0) and in intepreter mode (TRITON_INTERPRET=1)
def triton_interpret(request):
    '''Set env var TRITON_INTERPRET and reload triton'''
    os.environ['TRITON_INTERPRET'] = request.param
    importlib.reload(triton)
    importlib.reload(tl)
    yield request.param
    os.environ.pop('TRITON_INTERPRET', None)
