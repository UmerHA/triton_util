import triton
import triton.language as tl

def cdiv(a,b): return (a + b - 1) // b

@triton.jit
def get_1d_offset(size, n_prev_chunks=0): return n_prev_chunks * size + tl.arange(0, size)

@triton.jit
def get_2d_offset(offs_0, offs_1, stride_0, stride_1=1):  return tl.expand_dims(offs_0, 1)*stride_0 + tl.expand_dims(offs_1, 0)*stride_1

@triton.jit
def get_1d_mask(offs, max): return offs < max

@triton.jit
def get_2d_mask(offs_0, offs_1, max_0, max_1): return (tl.expand_dims(offs_0, 1) < max_0) & (tl.expand_dims(offs_1, 0) < max_1)

@triton.jit
def load_2d(ptr, sz0, sz1, n0, n1, max0, max1, stride0, stride1=1):
    '''Chunk 2d matrix (defined by ptr) into 2d grid, where each chunk has size (sz0,sz1). Load the (n0,n1)th chunk. Ie, load [n0*sz0,...,(n0+1)*sz0-1] x [n1*sz1,...,(n1+1)*sz1-1].'''
    offs0 = get_1d_offset(sz0, n0)
    offs1 = get_1d_offset(sz1, n1)        
    offs = get_2d_offset(offs0, offs1, stride0, stride1)
    mask = get_2d_mask(offs0, offs1, max0, max1)
    return tl.load(ptr + offs, mask) 

@triton.jit
def load_full_2d(ptr, sz0, sz1, stride0, stride1=1):
    '''Load 2d block [0,...,sz0-1] x [0,...,sz1-1] '''
    offs = get_2d_offset(tl.arange(0, sz0), tl.arange(0, sz1), stride0, stride1)
    mask = get_2d_mask(  tl.arange(0, sz0), tl.arange(0, sz1), sz0, sz1)
    return tl.load(ptr + offs, mask) 

@triton.jit
def load_full_1d(ptr, sz, stride=1):
    '''Load 1d block [0,...,sz-1]'''
    offs = get_1d_offset(sz)
    mask = get_1d_mask(offs, sz)
    return tl.load(ptr + offs, mask) 
