import os
import triton
import triton.language as tl

@triton.jit
def test_pid_conds(conds):
    '''Test if condition on pids are fulfilled
    E.g.:
        '=0'    checks that pid_0 == 0
        ',>1'   checks that pid_1 > 1
        '>1,=0' checks that pid_0 > 1 and pid_1 == 0
    '''
    pids = tl.program_id(0)[0], tl.program_id(1)[0], tl.program_id(2)[0]
    conds = conds.replace(' ','').split(',')
    for i, (cond, pid) in enumerate(zip(conds, pids)):
        if cond=='': continue
        if   cond[:2] in ['<=', '>=', '!=']: op, threshold = cond[:2], int(cond[2:])
        elif cond[:1] in ['<',  '>',  '=' ]: op, threshold = cond[:1], int(cond[1:])        
        else: raise ValueError(f"Rules may only use these ops: '<','>','>=','<=','=', '!='. Invalid rule in '{cond}'.")
        op = '==' if op == '=' else op
        if not eval(f'{pid} {op} {threshold}'): return False
    return True

@triton.jit
def breakpoint_if(conds):
    '''Stop kernel, if any condition of pids is fulfilled'''
    from IPython.core.debugger import set_trace
    if test_pid_conds(conds): set_trace()

@triton.jit
def print_if(txt, conds):
    '''Print txt, if any condition of pids is fulfilled'''
    if test_pid_conds(conds): print(txt)

@triton.jit
def breakpoint_once(): breakpoint_if('=0,=0,=0')

@triton.jit
def print_once(txt): print_if(txt,'=0,=0,=0')

def assert_tensors_gpu_ready(*tensors):
    for t in tensors:
        assert t.is_contiguous(), "A tensor is not contiguous"
        if not os.environ.get('TRITON_INTERPRET') == '1': assert t.is_cuda, "A tensor is not on cuda"
