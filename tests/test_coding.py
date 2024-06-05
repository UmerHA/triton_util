
import unittest
import inspect

import triton.language as tl

from triton_util.coding import cdiv, constify

class TestTritonUtil(unittest.TestCase):
    
    def test_cdiv(self):
        self.assertEqual(cdiv(10, 2), 5)
        self.assertEqual(cdiv(10, 3), 4)

    def test_constify(self):
        @constify(const='p1 p2 p3')
        def fn1(p1, p2, p3, p4, p5_ptr, p6): pass

        @constify(but='p4 p5')
        def fn2(p1, p2, p3, p4, p5): pass

        @constify(but='*_ptr')
        def fn3(p1, p2, p3, p4_ptr, p5, p6_ptr): pass

        @constify(but='p4 *_ptr')
        def fn4(p1, p2, p3, p4, p5_ptr, p6): pass

        @constify()
        def fn5(p1, p2, p3): pass

        for fn, const_params in [
            (fn1, ['p1', 'p2', 'p3']),
            (fn2, ['p1', 'p2', 'p3']),
            (fn3, ['p1', 'p2', 'p3', 'p5']),
            (fn4, ['p1', 'p2', 'p3', 'p6']),
            (fn5, ['p1', 'p2', 'p3']),
        ]:
            sig = inspect.signature(fn)
            for name, param in sig.parameters.items(): self.assertTrue(isinstance(param.annotation, tl.const) == (name in const_params))

if __name__ == '__main__':
    unittest.main()
