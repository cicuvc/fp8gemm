"""Numerical and cache-lifecycle checks for the column-packed FP8 experiment."""
import pytest
import torch

import mxtest
from fp8_swiglu import FP8SwiGLU
from experiments.cached_fp8_wgrad import CachedWgradSwiGLU, pack_columns, packed_wgrad

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')


@pytest.mark.parametrize('shape',[(1,33),(17,65),(65,97),(256,512)])
def test_column_pack_matches_online_transpose(shape):
    m,f=shape
    torch.manual_seed(39)
    # Non-contiguous rows are allowed; the feature dimension remains contiguous.
    x=torch.randn(m,2*f,device='cuda',dtype=torch.bfloat16)[:,:f]
    # Use matching explicit strides to exercise the dual packer's row stride.
    y_storage=torch.randn(m,2*f,device='cuda',dtype=torch.bfloat16)[:,:f]
    q,s,uq,us=pack_columns(x,y_storage)
    for original,actual,scales in ((x,q,s),(y_storage,uq,us)):
        # contiguous() may retain an arbitrary stride on a singleton axis.
        transposed=torch.empty((f,m),device=x.device,dtype=x.dtype)
        transposed.copy_(original.T)
        ref_q,ref_s=mxtest.launch_row_quantize(transposed)
        torch.testing.assert_close(actual[:,:m].view(torch.uint8),ref_q.view(torch.uint8),rtol=0,atol=0)
        torch.testing.assert_close(scales,ref_s,rtol=0,atol=0)
        assert torch.count_nonzero(actual[:,m:].view(torch.uint8)).item()==0
        q_single,s_single=pack_columns(original)
        torch.testing.assert_close(q_single.view(torch.uint8),actual.view(torch.uint8),rtol=0,atol=0)
        torch.testing.assert_close(s_single,scales,rtol=0,atol=0)


def test_packed_weight_grad_matches_online_with_padding():
    torch.manual_seed(40)
    x=torch.randn(33,65,device='cuda',dtype=torch.bfloat16)
    d=torch.randn(33,97,device='cuda',dtype=torch.bfloat16)
    u=torch.randn_like(d)
    xq,xs=pack_columns(x)
    dq,ds,uq,us=pack_columns(d,u)
    expected=mxtest.launch_fp8gemm_weight_grad_dual(d,u,x,out_dtype=torch.float32)
    actual=packed_wgrad(dq,ds,xq,xs,uq,us)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(packed_wgrad(dq,ds,xq,xs),expected[0],rtol=1e-5,atol=1e-5)


@pytest.mark.parametrize('fuse_hidden,fuse_grads,fuse_input',[(False,False,False),(True,False,False),(False,True,False),(True,True,False),(True,False,True)])
@pytest.mark.parametrize('bias',[False,True])
def test_cached_module_matches_online_and_refreshes_weights(fuse_hidden,fuse_grads,fuse_input,bias):
    torch.manual_seed(41)
    reference=FP8SwiGLU(65,97,33,bias=bias,fp8_wgrad=True,device='cuda')
    cached=CachedWgradSwiGLU(65,97,33,bias=bias,device='cuda',
                            fuse_hidden_pack=fuse_hidden,fuse_grad_pack=fuse_grads,fuse_input_pack=fuse_input)
    cached.load_state_dict(reference.state_dict())
    reference_optimizer=torch.optim.SGD(reference.parameters(),lr=0.001)
    cached_optimizer=torch.optim.SGD(cached.parameters(),lr=0.001)
    source=torch.randn(17,130,device='cuda',dtype=torch.bfloat16)[:,:65]
    xs=[source.detach().requires_grad_(),source.detach().requires_grad_()]
    dy=torch.randn(33,17,device='cuda',dtype=torch.bfloat16).T
    for _ in range(2):
        outputs=[]
        for layer,x,optimizer in zip((reference,cached),xs,(reference_optimizer,cached_optimizer)):
            optimizer.zero_grad(set_to_none=True)
            x.grad=None
            y=layer(x)
            y.backward(dy)
            outputs.append((y.detach(),x.grad.clone(),*[p.grad.clone() for p in layer.parameters()]))
        for a,b in zip(*outputs):
            torch.testing.assert_close(a,b,rtol=1e-3,atol=1e-4)
        reference_optimizer.step()
        cached_optimizer.step()
    # The no_grad path should retain the ordinary inference behavior.
    with torch.no_grad():
        torch.testing.assert_close(reference(xs[0]),cached(xs[1]),rtol=1e-3,atol=1e-4)
