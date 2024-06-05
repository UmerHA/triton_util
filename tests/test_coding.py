
import unittest
import inspect

import triton.language as tl

from triton_util.coding import cdiv, constify, tjit

class TestTritonUtil(unittest.TestCase):
    
    def test_cdiv(self):
        self.assertEqual(cdiv(10, 2), 5)
        self.assertEqual(cdiv(10, 3), 4)

    def test_tjit(self):
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
        def fn6(p1, p2, p3: tl.constexpr):
            tl.arange(0,p3)

        # tl.arange needs multiple of 2, so 8 is valid, but 1 is not
        fn1[(1,)](8,8,8,1,1,1)
        fn2[(1,)](8,8,8,1,1)
        fn3[(1,)](8,8,8,1,8,1)
        fn4[(1,)](8,8,8,1,1,8)
        fn5[(1,)](1,1,1)
        fn6[(1,)](1,1,8)

    def test_constify(self):
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
                self.assertTrue((param.annotation==tl.constexpr)==(name in const_params), f'Failed for {fn.__name__} with signature {sig}')

if __name__ == '__main__':
    unittest.main()
