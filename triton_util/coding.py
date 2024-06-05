import triton
import triton.language as tl
import inspect
import re

def cdiv(a,b): return (a + b - 1) // b

def const(func, const='', *, but=''):
    assert const == '' or but == '', 'Provide either const or but, not both'
    const, but = const.split(' '), but.split(' ')
    sig = inspect.signature(func)

    def match_pattern(param_name, pattern):
        regex = re.sub(r'\*', r'.*', pattern) # Convert wildcard pattern to regex pattern
        return re.match(f"^{regex}$", param_name)

    def should_be_const(param_name):
        if const != '': return any(match_pattern(param_name, pattern) for pattern in const)
        if but != '':   return not any(match_pattern(param_name, pattern) for pattern in but)
        raise False
    
    new_params = [
        param.replace(annotation=tl.constexpr) if should_be_const(name) else param
        for name, param in sig.parameters.items()
    ]

    new_sig = sig.replace(parameters=new_params)
    def wrapper(*args, **kwargs): return func(*args, **kwargs)
    wrapper.__signature__ = new_sig
    return wrapper

def tjit(fn = None, *, const='', non_const='', version=None, do_not_specialize = None, debug = None, noinline = None):
    '''Apply const and triton.jit to fn.'''
    fn = const(fn, const=const, but=non_const)
    return triton.jit(fn, version=version, do_not_specialize=do_not_specialize, debug=debug, noinline=noinline)

@tjit(const='sz')
def get_1d_offset(sz, n_prev_chunks=0): return n_prev_chunks * sz + tl.arange(0, sz)

@tjit
def get_2d_offset(offs0, offs1, stride0, stride1=1):  return tl.expand_dims(offs0, 1)*stride0 + tl.expand_dims(offs1, 0)*stride1

@tjit
def get_1d_mask(offs, max): return offs < max

@tjit
def get_2d_mask(offs0, offs1, max0, max1): return (tl.expand_dims(offs0, 1) < max0) & (tl.expand_dims(offs1, 0) < max1)

@tjit(const='sz0 sz1')
def load_2d(ptr, sz0, sz1, n0, n1, max0, max1, stride0, stride1=1):
    '''Chunk 2d matrix (defined by ptr) into 2d grid, where each chunk has size (sz0,sz1). Load the (n0,n1)th chunk. Ie, load [n0*sz0,...,(n0+1)*sz0-1] x [n1*sz1,...,(n1+1)*sz1-1].'''
    offs0 = get_1d_offset(sz0, n0)
    offs1 = get_1d_offset(sz1, n1)        
    offs = get_2d_offset(offs0, offs1, stride0, stride1)
    mask = get_2d_mask(offs0, offs1, max0, max1)
    return tl.load(ptr + offs, mask) 

@tjit(const='sz0 sz1')
def load_full_2d(ptr, sz0, sz1, stride0, stride1=1):
    '''Load 2d block [0,...,sz0-1] x [0,...,sz1-1] '''
    offs = get_2d_offset(tl.arange(0, sz0), tl.arange(0, sz1), stride0, stride1)
    mask = get_2d_mask(  tl.arange(0, sz0), tl.arange(0, sz1), sz0, sz1)
    return tl.load(ptr + offs, mask) 

@tjit(const='sz')
def load_full_1d(ptr, sz, stride=1):
    '''Load 1d block [0,...,sz-1]'''
    offs = get_1d_offset(sz)
    mask = get_1d_mask(offs, sz)
    return tl.load(ptr + offs, mask) 

@tjit(const='sz0 sz1')
def store_2d(vals, ptr, sz0, sz1, n0, n1, max0, max1, stride0, stride1=1):
    '''Store 2d block into (n0,n1)th chunk of matrix (defined by ptr), where each chunk has size (sz0, sz1)'''
    offs0 = get_1d_offset(sz0, n0)
    offs1 = get_1d_offset(sz1, n1)        
    offs = get_2d_offset(offs0, offs1, stride0, stride1)
    mask = get_2d_mask(offs0, offs1, max0, max1)
    tl.store(ptr + offs, vals, mask)

@tjit(const='sz0 sz1')
def store_full_2d(vals, ptr, sz0, sz1, stride0, stride1=1):
    '''Store 2d block into matrix (defined by ptr)'''
    offs = get_2d_offset(tl.arange(0, sz0), tl.arange(0, sz1), stride0, stride1)
    mask = get_2d_mask(  tl.arange(0, sz0), tl.arange(0, sz1), sz0, sz1)
    return tl.store(ptr + offs, vals, mask)

@tjit(const='sz')
def store_full_1d(vals, ptr, sz, stride=1):
    '''Store 1d block into vector (defined by ptr)'''
    offs = get_1d_offset(sz)
    mask = get_1d_mask(offs, sz)
    return tl.store(ptr + offs, vals, mask)
