import inspect

import pytest

import torch

import triton
import triton.language as tl

import triton_util as tu

class TestCodingUtils:
    def test_cdiv(self):
        assert tu.cdiv(10, 2)==5
        assert tu.cdiv(10, 3)==4

    ## offsets

    def test_offset_1d(self, triton_interpret):
        i = torch.ones(4, device='cuda')
        o = torch.zeros(4, device='cuda')

        @triton.jit
        def partial_copy(i_ptr, o_ptr, n):
            offs = tu.offset_1d(sz=2, n_prev_chunks=n)
            mask = offs < 4
            vals = tl.load(i_ptr + offs, mask)
            tl.store(o_ptr + offs, vals, mask)            

        partial_copy[(1,)](i, o, n=0)
        assert list(o) == [1,1,0,0]

        partial_copy[(1,)](i, o, n=1)
        assert list(o) == [1,1,1,1]

    def test_offset_2d(self, triton_interpret):
        i = torch.ones(4,4, device='cuda')
        o = torch.zeros(4,4, device='cuda')

        @triton.jit
        def partial_copy(i_ptr, o_ptr, n1, n2):
            offs1 = n1*2 + tl.arange(0,2)
            offs2 = n2*2 + tl.arange(0,2)
            offs = tu.offset_2d(offs1, offs2, stride0=4)
            mask = (offs1 < 4) & (offs2 < 4)
            vals = tl.load(i_ptr + offs, mask)
            tl.store(o_ptr + offs, vals, mask)            

        partial_copy[(1,)](i, o, n1=0, n2=0)
        assert list(o) == [[1,1,0,0], [1,1,0,0], [0,0,0,0], [0,0,0,0]]

        partial_copy[(1,)](i, o, n1=1, n2=0)
        assert list(o) == [[1,1,0,0], [1,1,0,0], [1,1,0,0], [1,1,0,0]]
    
    ## masks

    def test_mask_1d(self, triton_interpret):
        i = torch.ones(4, device='cuda')
        o = torch.zeros(4, device='cuda')

        @triton.jit
        def partial_copy(i_ptr, o_ptr, n):
            offs = n*2 + tl.arage(0,2)
            mask = mask_1d(offs, 4)
            vals = tl.load(i_ptr + offs, mask)
            tl.store(o_ptr + offs, vals, mask)            

        partial_copy[(1,)](i, o, n=0)
        assert list(o) == [1,1,0,0]

        partial_copy[(1,)](i, o, n=1)
        assert list(o) == [1,1,1,1]
    
    def test_mask_2d(self, triton_interpret):
        i = torch.ones(4,4, device='cuda')
        o = torch.zeros(4,4, device='cuda')

        @triton.jit
        def partial_copy(i_ptr, o_ptr, n1, n2):
            offs1 = n1*2 + tl.arange(0,2)
            offs2 = n2*2 + tl.arange(0,2)
            offs = tl.expand_dims(offs1, 1)*4 + tl.expand_dims(offs2, 0)*1
            mask = tu.mask_2d(offs1, offs2, 4, 4)
            vals = tl.load(i_ptr + offs, mask)
            tl.store(o_ptr + offs, vals, mask)            

        partial_copy[(1,)](i, o, n1=0, n2=0)
        assert list(o) == [[1,1,0,0], [1,1,0,0], [0,0,0,0], [0,0,0,0]]

        partial_copy[(1,)](i, o, n1=1, n2=0)
        assert list(o) == [[1,1,0,0], [1,1,0,0], [1,1,0,0], [1,1,0,0]]

    # # load

    def test_load_1d(self, triton_interpret):
        i = torch.ones(4, device='cuda')
        o = torch.zeros(4, device='cuda')

        @triton.jit
        def partial_copy(i_ptr, o_ptr, n):
            offs = n*2 + tl.arange(0,2)
            mask = offs < 4
            vals = tu.load_1d(i_ptr, 2, n, 4)
            tl.store(o_ptr + offs, vals, mask)            

        partial_copy[(1,)](i, o, n=0)
        assert list(o) == [1,1,0,0]

        partial_copy[(1,)](i, o, n=1)
        assert list(o) == [1,1,1,1]

    def test_load_full_1d(self, triton_interpret):
        i = torch.ones(4, device='cuda')
        o = torch.zeros(4, device='cuda')

        @triton.jit
        def copy(i_ptr, o_ptr):
            offs = tl.arage(0,4)
            mask = offs < 4
            vals = tu.load_full_1d(i_ptr, 4)
            tl.store(o_ptr + offs, vals, mask)            

        copy[(1,)](i, o)
        assert list(o) == list(i)

    def test_load_2d(self, triton_interpret):
        i = torch.ones(4,4, device='cuda')
        o = torch.zeros(4,4, device='cuda')

        @triton.jit
        def partial_copy(i_ptr, o_ptr, n1, n2):
            offs1 = n1*2 + tl.arange(0,2)
            offs2 = n2*2 + tl.arange(0,2)
            offs = tl.expand_dims(offs1, 1)*4 + tl.expand_dims(offs2, 0)*1
            mask = (offs1 < 4) & (offs2 < 4)
            vals = tu.load_2d(i_ptr, 2, 2, n1, n2, 4, 4, 4, 1)
            tl.store(o_ptr + offs, vals, mask)            

        partial_copy[(1,)](i, o, n1=0, n2=0)
        assert list(o) == [[1,1,0,0], [1,1,0,0], [0,0,0,0], [0,0,0,0]]

        partial_copy[(1,)](i, o, n1=1, n2=0)
        assert list(o) == [[1,1,0,0], [1,1,0,0], [1,1,0,0], [1,1,0,0]]
    
    def test_load_full_2d(self, triton_interpret):
        i = torch.ones(4,4, device='cuda')
        o = torch.zeros(4,4, device='cuda')

        @triton.jit
        def copy(i_ptr, o_ptr):
            offs1 = tl.arange(0,4)
            offs2 = tl.arange(0,4)
            offs = tl.expand_dims(offs1, 1)*4 + tl.expand_dims(offs2, 0)*1
            mask = (offs1 < 4) & (offs2 < 4)
            vals = tu.load_full_2d(i_ptr, 4, 4, 4, 1)
            tl.store(o_ptr + offs, vals, mask)            

        copy[(1,)](i, o)
        assert list(o) == list(i)

    # # store

    def test_store_1d(self, triton_interpret):
        i = torch.ones(4, device='cuda')
        o = torch.zeros(4, device='cuda')

        @triton.jit
        def partial_copy(i_ptr, o_ptr, n):
            offs = n*2 + tl.arange(0,2)
            mask = offs < 4
            vals = tl.load(i_ptr + offs, mask)
            tu.store_1d(vals, o_ptr, 2, n, 4)

        partial_copy[(1,)](i, o, n=0)
        assert list(o) == [1,1,0,0]

        partial_copy[(1,)](i, o, n=1)
        assert list(o) == [1,1,1,1]

    def test_store_full_1d(self, triton_interpret):
        i = torch.ones(4, device='cuda')
        o = torch.zeros(4, device='cuda')

        @triton.jit
        def copy(i_ptr, o_ptr):
            offs = tl.arange(0,4)
            mask = offs < 4
            vals = tl.load(i_ptr + offs, mask)
            tu.store_full_1d(vals, o_ptr, 4)

        copy[(1,)](i, o)
        assert list(o) == list(i)

    def test_store_2d(self, triton_interpret):
        i = torch.ones(4,4, device='cuda')
        o = torch.zeros(4,4, device='cuda')

        @triton.jit
        def partial_copy(i_ptr, o_ptr, n1, n2):
            offs1 = n1*2 + tl.arange(0,2)
            offs2 = n2*2 + tl.arange(0,2)
            offs = tl.expand_dims(offs1, 1)*4 + tl.expand_dims(offs2, 0)*1
            mask = (offs1 < 4) & (offs2 < 4)
            vals = tl.load(i_ptr + offs, mask)
            tu.store_2d(vals, o_ptr, 2, 2, n1, n2, 4, 4, 4, 1)

        partial_copy[(1,)](i, o, n1=0, n2=0)
        assert list(o) == [[1,1,0,0], [1,1,0,0], [0,0,0,0], [0,0,0,0]]

        partial_copy[(1,)](i, o, n1=1, n2=0)
        assert list(o) == [[1,1,0,0], [1,1,0,0], [1,1,0,0], [1,1,0,0]]

    def test_store_full_2d(self, triton_interpret):
        i = torch.ones(4,4, device='cuda')
        o = torch.zeros(4,4, device='cuda')

        @triton.jit
        def copy(i_ptr, o_ptr):
            offs1 = tl.arange(0,4)
            offs2 = tl.arange(0,4)
            offs = tl.expand_dims(offs1, 1)*4 + tl.expand_dims(offs2, 0)*1
            mask = (offs1 < 4) & (offs2 < 4)
            vals = tl.load(i_ptr + offs, mask)
            tu.store_full_2d(vals, o_ptr, 4, 4, 4, 1)

        copy[(1,)](i, o, n1=0, n2=0)
        assert list(o) == list(i)

if __name__ == '__main__':
    pytest.main()
