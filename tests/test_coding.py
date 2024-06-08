import inspect

import pytest

import torch

import triton
import triton.language as tl

from triton_util.coding import *

class TestCodingUtils:
    def test_cdiv(self):
        assert cdiv(10, 2)==5
        assert cdiv(10, 3)==4

    def test_constify(self, triton_interpret):
        @constify(const='p1 p2 p3')
        def fn1(p1, p2, p3, p4, p5_ptr, p6): pass

        @constify(but='p4 p5')
        def fn2(p1, p2, p3, p4, p5): pass

        @constify(but='*_ptr')
        def fn3(p1, p2, p3, p4_ptr, p5, p6_ptr): pass

        @constify(but='p4 *_ptr')
        def fn4(p1, p2, p3, p4, p5_ptr, p6): pass

        @constify
        def fn5(p1, p2, p3): pass

        for fn, const_params in [
            (fn1, ['p1', 'p2', 'p3']),
            (fn2, ['p1', 'p2', 'p3']),
            (fn3, ['p1', 'p2', 'p3', 'p5']),
            (fn4, ['p1', 'p2', 'p3', 'p6']),
            (fn5, []),
        ]:
            sig = inspect.signature(fn)
            for name, param in sig.parameters.items(): 
                assert (param.annotation==tl.constexpr)==(name in const_params), f'Failed for {fn.__name__} with signature {sig}'

    def test_tjit(self, triton_interpret):
        @tjit(const='p1 p2 p3')
        def fn1(p1, p2, p3, p4, p5_ptr, p6):
            tl.arange(0,p1) # tl.arange expects constexpr
            tl.arange(0,p2)
            tl.arange(0,p3)

        @tjit(non_const='p4 p5')
        def fn2(p1, p2, p3, p4, p5):
            tl.arange(0,p1)
            tl.arange(0,p2)
            tl.arange(0,p3)

        @tjit(non_const='*_ptr')
        def fn3(p1, p2, p3, p4_ptr, p5, p6_ptr):
            tl.arange(0,p1)
            tl.arange(0,p2)
            tl.arange(0,p3)
            tl.arange(0,p5)

        @tjit(non_const='p4 *_ptr')
        def fn4(p1, p2, p3, p4, p5_ptr, p6):
            tl.arange(0,p1)
            tl.arange(0,p2)
            tl.arange(0,p3)
            tl.arange(0,p6)

        @tjit
        def fn5(p1, p2, p3): pass

        @tjit
        def fn6(p1, p2, p3: tl.constexpr): tl.arange(0,p3)

        # tl.arange needs multiple of 2, so 8 is valid, but 1 is not
        fn1[(1,)](8,8,8,1,1,1)
        fn2[(1,)](8,8,8,1,1)
        fn3[(1,)](8,8,8,1,8,1)
        fn4[(1,)](8,8,8,1,1,8)
        fn5[(1,)](1,1,1)
        fn6[(1,)](1,1,8)

    ## offsets

    def test_get_1d_offset(self, triton_interpret):
        i = torch.ones(4, device='cuda')
        o = torch.zeros(4, device='cuda')

        @triton.jit
        def partial_copy(i_ptr, o_ptr, n):
            offs = get_1d_offset(sz=2, n_prev_chunks=n)
            mask = offs < 4
            vals = tl.load(i_ptr + offs, mask)
            tl.store(o_ptr + offs, vals, mask)            

        partial_copy[(1,)](i, o, n=0)
        assert list(o) == [1,1,0,0]

        partial_copy[(1,)](i, o, n=1)
        assert list(o) == [1,1,1,1]

    def test_get_2d_offset(self, triton_interpret):
        i = torch.ones(4,4, device='cuda')
        o = torch.zeros(4,4, device='cuda')

        @triton.jit
        def partial_copy(i_ptr, o_ptr, n1, n2):
            offs1 = n1*2 + tl.arange(0,2)
            offs2 = n2*2 + tl.arange(0,2)
            offs = get_2d_offset(offs1, offs2, stride0=4)
            mask = (offs1 < 4) & (offs2 < 4)
            vals = tl.load(i_ptr + offs, mask)
            tl.store(o_ptr + offs, vals, mask)            

        partial_copy[(1,)](i, o, n1=0, n2=0)
        assert list(o) == [[1,1,0,0], [1,1,0,0], [0,0,0,0], [0,0,0,0]]

        partial_copy[(1,)](i, o, n1=1, n2=0)
        assert list(o) == [[1,1,0,0], [1,1,0,0], [1,1,0,0], [1,1,0,0]]
    
    ## masks

    def test_get_1d_mask(self, triton_interpret):
        i = torch.ones(4, device='cuda')
        o = torch.zeros(4, device='cuda')

        @triton.jit
        def partial_copy(i_ptr, o_ptr, n):
            offs = n*2 + tl.arange(0,2)
            mask = get_1d_mask(offs, 4)
            vals = tl.load(i_ptr + offs, mask)
            tl.store(o_ptr + offs, vals, mask)            

        partial_copy[(1,)](i, o, n=0)
        assert list(o) == [1,1,0,0]

        partial_copy[(1,)](i, o, n=1)
        assert list(o) == [1,1,1,1]
    
    def test_get_2d_mask(self, triton_interpret):
        i = torch.ones(4,4, device='cuda')
        o = torch.zeros(4,4, device='cuda')

        @triton.jit
        def partial_copy(i_ptr, o_ptr, n1, n2):
            offs1 = n1*2 + tl.arange(0,2)
            offs2 = n2*2 + tl.arange(0,2)
            offs = tl.expand_dims(offs1, 1)*4 + tl.expand_dims(offs2, 0)*1
            mask = get_2d_mask(offs1, offs2, 4, 4)
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
            vals = load_1d(i_ptr, 2, n, 4)
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
            offs = tl.arange(0,4)
            mask = offs < 4
            vals = load_full_1d(i_ptr, 4)
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
            vals = load_2d(i_ptr, 2, 2, n1, n2, 4, 4, 4, 1)
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
            vals = load_full_2d(i_ptr, 4, 4, 4, 1)
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
            store_1d(vals, o_ptr, 2, n, 4)

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
            store_full_1d(vals, o_ptr, 4)

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
            store_2d(vals, o_ptr, 2, 2, n1, n2, 4, 4, 4, 1)

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
            store_full_2d(vals, o_ptr, 4, 4, 4, 1)

        copy[(1,)](i, o, n1=0, n2=0)
        assert list(o) == list(i)

if __name__ == '__main__':
    pytest.main()
