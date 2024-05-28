# Make Triton easier 🔱 😊
> Utility functions for OpenAI Triton

Writing fast GPU kernels is easier with Triton than with CUDA, but there is still a lot of tedious indices juggling. That is not necessary.

**Triton-util provides simple higher level abstractions for frequent but repetetive tasks.** This allows you to write code that is more like your actual ideas.

Example: Say you have a 2d matrix which you have chunked along both axes, and you want to get the next chunk. With triton-util, you write
```py
# ptr = pointer to matrix
# sz0/sz1 = sizes of chunk
# n0/n1 = index of chunk
# max0/max1 = bound of matrix
# stride0/stride1 = strides of matrix (stride1 is not set, so will use the sensible default 1)
load_2d(ptr, sz0, sz1, n0, n1, max0, max1, stride0)
```
instead of
```py
offs0 = n0 * sz0 + tl.arange(0, sz0)
offs1 = n1 * sz1 + tl.arange(0, sz1)
offs = offs0[:,None] * stride0 + offs1[None,:] * stride1
mask = (offs0[:,None] < max0) & (offs1[None,:] < max1)
return tl.load(ptr + offs, mask) 
```

Additionally, triton-util provides utility functions to make debugging easier. Want to print `txt` only on the 1st kernel? Write `print_once('txt')` - that's it!


<br/>

## Debugging utils

`print_once(txt)`
- Print txt, only on 1st kernel (ie all pids = 0)

`breakpoint_once()`
- Enter breakpoint, only on 1st kernel (ie all pids = 0)

`print_if(txt, conds)`
- Print txt, if condition on pids is fulfilled
- Eg `breakpoint_if('=0,>1')` stops if `pid_0 = 0`, `pid_1 > 1` and `pid_2` is abitrary

`breakpoint_if(conds)`
- Enter breakpoint, if condition on pids is fulfilled
- Eg `breakpoint_if('=0,>1')` stops if `pid_0 = 0`, `pid_1 > 1` and `pid_2` is abitrary

`assert_tensors_gpu_ready(*tensors)`
- assert all tensors are contiguous, and (unless `'TRITON_INTERPRET'=='1'`) on GPU

<br/>

## Coding utils

`cdiv(a,b)`:
- ceiling division

`get_1d_offset(sz, n_prev_chunks=0)`
- Return 1d offsets to `(n_prev_chunks+1)`th chunk of size `sz`

`get_2d_offset(offs_0, offs_1, stride_0, stride_1=1)`
- Create a 2d offets from two 1d offets

`get_1d_mask(offs, max)`
- Create a 1d mask from a 1d offset and a max value

`get_2d_mask(offs_0, offs_1, max_0, max_1)`
- Create a 2d mask from two 1d offsets and max values

`load_2d(ptr, sz0, sz1, n0, n1, max0, max1, stride0, stride1=1)`
- Chunk 2d matrix (defined by ptr) into 2d grid, where each chunk has size `(sz0,sz1)`, and load the `(n0,n1)`th chunk.

`load_full_2d(ptr, sz0, sz1, stride0, stride1=1)`
- Load 2d block of size `sz0 x sz1`

`load_full_1d(ptr, sz, stride=1)`
- Load 1d block of size `sz`

<br/>

___

Brought to you by [Umer](https://x.com/UmerHAdil) ❤️
