"""Training experiment: cache column-wise FP8 activations for weight gradients.

Packed tensors have shape [features, padded_tokens], with one E8M0 scale
per 32 tokens. Quantization matches the existing online FP8 wgrad path.
The normal FP8SwiGLU implementation remains unchanged.
"""
import torch
import triton
import triton.language as tl

import mxtest
from fp8_swiglu import FP8SwiGLU, _fp8_swiglu_fwd
from experiments.cached_fp8_wgrad_fused import saved_hidden_forward, saved_grads_backward


@triton.jit
def _pack_columns(x, y, xq, xs, yq, ys, M: tl.constexpr, F: tl.constexpr,
                  STRIDE: tl.constexpr, MP: tl.constexpr, DUAL: tl.constexpr,
                  BM: tl.constexpr, BF: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BF + tl.arange(0, BF)
    offsets = rows[:, None] * STRIDE + cols[None, :]
    valid = (rows[:, None] < M) & (cols[None, :] < F)
    values = tl.load(x + offsets, mask=valid, other=0)
    q, s = mxtest.online_scale(tl.trans(values))
    packed = cols[:, None] * MP + rows[None, :]
    packed_mask = (cols[:, None] < F) & (rows[None, :] < MP)
    groups = tl.program_id(0) * (BM // 32) + tl.arange(0, BM // 32)
    scale_offsets = cols[:, None] * (MP // 32) + groups[None, :]
    scale_mask = (cols[:, None] < F) & (groups[None, :] < MP // 32)
    tl.store(xq + packed, q, mask=packed_mask)
    tl.store(xs + scale_offsets, s, mask=scale_mask)
    if DUAL:
        values1 = tl.load(y + offsets, mask=valid, other=0)
        q1, s1 = mxtest.online_scale(tl.trans(values1))
        tl.store(yq + packed, q1, mask=packed_mask)
        tl.store(ys + scale_offsets, s1, mask=scale_mask)


def pack_columns(x, y=None):
    assert x.ndim == 2 and x.dtype == torch.bfloat16 and x.is_cuda
    assert x.stride(1) == 1
    m, f = x.shape
    mp = triton.cdiv(m, 32) * 32
    xq = torch.empty((f, mp), device=x.device, dtype=torch.float8_e4m3fn)
    xs = torch.empty((f, mp // 32), device=x.device, dtype=torch.uint8)
    yq = ys = None
    if y is not None:
        assert y.shape == x.shape and y.stride() == x.stride() and y.dtype == x.dtype
        yq, ys = torch.empty_like(xq), torch.empty_like(xs)
    _pack_columns[(triton.cdiv(m, 64), triton.cdiv(f, 64))](
        x, y, xq, xs, yq, ys, m, f, x.stride(0), mp, y is not None,
        BM=64, BF=64, num_warps=4,
    )
    return (xq, xs) if y is None else (xq, xs, yq, ys)


PACKED_WGRAD_CONFIGS = [
    triton.Config({'BM': bm, 'BN': bn, 'BK': bk}, num_warps=warps, num_stages=stages)
    for bm,bn,bk,warps,stages in (
        (64,64,64,4,2), (128,64,64,4,2), (256,64,64,4,2),
        (128,64,128,4,2), (128,64,128,8,2), (128,128,64,4,2),
        (128,128,64,8,2), (128,64,64,4,3), (256,64,64,4,3),
        (256,64,128,8,2), (256,128,64,8,2), (128,128,128,8,2),
    )
]


@triton.autotune(configs=PACKED_WGRAD_CONFIGS, key=['MP','N','K','DUAL'], cache_results=True)
@triton.jit
def _packed_wgrad(dq, ds, uq, us, xq, xs, dw, uw,
                   MP: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   DUAL: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) // tl.cdiv(K, BK)
    col = tl.program_id(0) % tl.cdiv(K, BK)
    dp = tl.make_block_ptr(dq, (N, MP), (MP, 1), (row*BN, 0), (BN, BM), (1,0))
    sp = tl.make_block_ptr(ds, (N, MP//32), (MP//32, 1), (row*BN, 0), (BN, BM//32), (1,0))
    xp = tl.make_block_ptr(xq, (K, MP), (MP, 1), (col*BK, 0), (BK, BM), (1,0))
    xsp = tl.make_block_ptr(xs, (K, MP//32), (MP//32, 1), (col*BK, 0), (BK, BM//32), (1,0))
    if DUAL:
        up = tl.make_block_ptr(uq, (N, MP), (MP, 1), (row*BN, 0), (BN, BM), (1,0))
        usp = tl.make_block_ptr(us, (N, MP//32), (MP//32, 1), (row*BN, 0), (BN, BM//32), (1,0))
        acc1 = tl.zeros((BN, BK), tl.float32)
    acc = tl.zeros((BN, BK), tl.float32)
    for _ in range(tl.cdiv(MP, BM)):
        d = tl.load(dp, boundary_check=(0,1), padding_option='zero')
        s = tl.load(sp, boundary_check=(0,1), padding_option='zero')
        x = tl.load(xp, boundary_check=(0,1), padding_option='zero')
        sx = tl.load(xsp, boundary_check=(0,1), padding_option='zero')
        acc = tl.dot_scaled(d, s, 'e4m3', tl.trans(x), sx, 'e4m3', acc)
        if DUAL:
            u = tl.load(up, boundary_check=(0,1), padding_option='zero')
            su = tl.load(usp, boundary_check=(0,1), padding_option='zero')
            acc1 = tl.dot_scaled(u, su, 'e4m3', tl.trans(x), sx, 'e4m3', acc1)
            up = tl.advance(up, (0, BM))
            usp = tl.advance(usp, (0, BM//32))
        dp = tl.advance(dp, (0, BM))
        sp = tl.advance(sp, (0, BM//32))
        xp = tl.advance(xp, (0, BM))
        xsp = tl.advance(xsp, (0, BM//32))
    op = tl.make_block_ptr(dw, (N, K), (K, 1), (row*BN,col*BK), (BN,BK), (1,0))
    tl.store(op, acc, boundary_check=(0,1))
    if DUAL:
        op1 = tl.make_block_ptr(uw, (N,K), (K,1), (row*BN,col*BK), (BN,BK), (1,0))
        tl.store(op1, acc1, boundary_check=(0,1))


def packed_wgrad(dq, ds, xq, xs, uq=None, us=None):
    n, mp = dq.shape
    k, mp_x = xq.shape
    assert mp == mp_x and mp % 32 == 0
    assert dq.is_contiguous() and xq.is_contiguous()
    assert ds.shape == (n, mp//32) and xs.shape == (k, mp//32)
    dw = torch.empty((n,k),device=dq.device,dtype=torch.float32)
    uw = None if uq is None else torch.empty_like(dw)
    _packed_wgrad[lambda c:(triton.cdiv(n,c['BN'])*triton.cdiv(k,c['BK']),)](
        dq,ds,uq,us,xq,xs,dw,uw,mp,n,k,uq is not None)
    return dw if uq is None else (dw,uw)


def cached_backward(dy, gate, up, weights, xq, xs, hq, hs, fuse_grad_pack=False):
    dw, ds, gw, gs, uw, us = weights
    m, o = dy.shape
    h, k = gw.shape
    if fuse_grad_pack:
        dg,du,dq,dqs,uq,uqs = saved_grads_backward(dy,gate,up,dw,ds)
    else:
        dg, du = torch.empty_like(gate), torch.empty_like(up)
        mxtest.fp8gemm_bwd_swiglu_act[
            lambda c:(triton.cdiv(m,c['BLOCK_M'])*triton.cdiv(h,c['BLOCK_K']),)
        ](dy,dw,ds,gate,up,dg,du,m,o,h,dy.stride(0),dw.stride(0),ds.stride(0),
          gate.stride(0),up.stride(0),dg.stride(0),du.stride(0))
    dx = torch.empty((m,k),device=dy.device,dtype=torch.bfloat16)
    mxtest.fp8_swiglu_bwd_dx[
        lambda c:(triton.cdiv(m,c['BLOCK_M'])*triton.cdiv(k,c['BLOCK_K']),)
    ](dg,du,gw,gs,uw,us,dx,m,h,k,dg.stride(0),du.stride(0),gw.stride(0),gs.stride(0),
      uw.stride(0),us.stride(0),dx.stride(0))
    if not fuse_grad_pack:
        dq, dqs, uq, uqs = pack_columns(dg,du)
    dyq, dys = pack_columns(dy)
    dgw, duw = packed_wgrad(dq,dqs,xq,xs,uq,uqs)
    ddw = packed_wgrad(dyq,dys,hq,hs)
    return dx,dgw,duw,ddw


class _CachedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,gm,um,dm,gw,gs,uw,us,dw,ds,bias,fuse_hidden_pack,fuse_grad_pack,fuse_input_pack):
        if fuse_hidden_pack:
            y,gate,up,hq,hs,xq,xs = saved_hidden_forward(x,gw,gs,uw,us,dw,ds,bias,fuse_input_pack)
        else:
            y,gate,up,hidden = _fp8_swiglu_fwd(x,gm,um,dm,gw,gs,uw,us,dw,ds,bias,True)
            hq,hs = pack_columns(hidden)
        if not fuse_input_pack:
            xq,xs = pack_columns(x)
        ctx.fuse_grad_pack = fuse_grad_pack
        ctx.save_for_backward(gate,up,dw,ds,gw,gs,uw,us,xq,xs,hq,hs)
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx,dy):
        gate,up,dw,ds,gw,gs,uw,us,xq,xs,hq,hs = ctx.saved_tensors
        dy = dy.contiguous().to(torch.bfloat16)
        dx,dgw,duw,ddw = cached_backward(dy,gate,up,(dw,ds,gw,gs,uw,us),xq,xs,hq,hs,ctx.fuse_grad_pack)
        db = dy.sum(0,dtype=torch.float32) if ctx.has_bias else None
        return dx,dgw,duw,ddw,None,None,None,None,None,None,db,None,None,None


class CachedWgradSwiGLU(FP8SwiGLU):
    """Experimental FP8 dW; training forward includes both activation packs."""
    def __init__(self,in_features,hidden_features,out_features,bias=True,device=None,
                 fuse_hidden_pack=False,fuse_grad_pack=False,fuse_input_pack=False):
        super().__init__(in_features,hidden_features,out_features,bias=bias,fp8_wgrad=True,device=device)
        self.fuse_hidden_pack = fuse_hidden_pack
        self.fuse_grad_pack = fuse_grad_pack
        self.fuse_input_pack = fuse_input_pack
        if fuse_input_pack and not fuse_hidden_pack:
            raise ValueError('fuse_input_pack requires fuse_hidden_pack')

    def forward(self,x):
        if not torch.is_grad_enabled():
            return super().forward(x)
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(f'expected [M, {self.in_features}]')
        if x.dtype != torch.bfloat16 or not x.is_cuda:
            raise TypeError('expected CUDA BF16 input')
        x = x.contiguous()
        gw,gs = self.quantized_gate_weight
        uw,us = self.quantized_up_weight
        dw,ds = self.quantized_down_weight
        return _CachedSwiGLU.apply(x,self.gate_weight,self.up_weight,self.down_proj.master_weight,
                                  gw,gs,uw,us,dw,ds,self.down_proj.bias,self.fuse_hidden_pack,self.fuse_grad_pack,self.fuse_input_pack)
