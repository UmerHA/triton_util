import pytest

import torch

import triton
import triton.language as tl

import triton_util as tu
from triton_util.debugging import _test_pid_conds, offsets_from_base

class TestDebuggingUtils:
    def test_test_pid_conds(self):
        assert _test_pid_conds('')
        assert _test_pid_conds('>0', 1, 1)
        assert not _test_pid_conds('>0', 0, 1)
        assert _test_pid_conds('=0,=1', 0, 1, 0)

    def test_assert_tensors_gpu_ready(self, triton_interpret):
        t1 = torch.ones(4, device='cuda')      # gpu, contiguous
        t2 = torch.ones(4)                     # cpu, contiguous 
        t3 = torch.ones(4, device='cuda')[::2] # gpu, non-contiguous
        t4 = torch.ones(4)[::2]                # cpu, non-contiguous

        tu.assert_tensors_gpu_ready(t1)
        
        if triton_interpret == '1':
            tu.assert_tensors_gpu_ready(t2)
        else:
            with pytest.raises(AssertionError): tu.assert_tensors_gpu_ready(t2)

        with pytest.raises(AssertionError): tu.assert_tensors_gpu_ready(t3)

        with pytest.raises(AssertionError): tu.assert_tensors_gpu_ready(t4)

    def test_offsets_from_base(self, triton_interpret):

        t = torch.zeros(4, device='cuda')
        out = torch.empty(4, device='cuda')

        @triton.jit
        def some_kernel(t_ptr, out_ptr):
            offs = offsets_from_base(t_ptr + tl.arange(4), t_ptr)
            tl.store(out_ptr, offs)

        some_kernel[(1,)](t, out)

        assert list(out) == list(range(4))

if __name__ == '__main__':
    pytest.main()
