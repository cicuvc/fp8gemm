"""Global topK Gumbel-max distribution, independent Philox reference, and graphs."""

import numpy as np
import pytest
import torch

from experiments.mxfp8_core import quantize_weight
from experiments.mxfp8_topk import MergePlan, create_sampling_state
from experiments.tk_mxfp8_sampler import MXFP8LMHeadSampler
from tests.test_tk_mxfp8_topk import compiled_topk, inputs, quantized, plain_logits, reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


def pack_local(z, k, block=16):
    """Independent host-side partition/topK construction for the merge tests."""
    m, n = z.shape
    bucket = 1 << (k - 1).bit_length()
    result = torch.zeros((m, (n + block - 1) // block, bucket), device=z.device, dtype=torch.int64)
    for part, start in enumerate(range(0, n, block)):
        values, ids = reference(z[:, start:start + block], min(bucket, n - start))
        ids += start
        bits = values.contiguous().view(torch.int16).long() & 0xffff
        bits = torch.where((bits & 0x7fff) == 0, 0, bits)
        order = torch.where((bits & 0x8000) != 0, (~bits) & 0xffff, bits ^ 0x8000)
        keys = (order << 32) | (0xffffffff - ids)
        result[:, part, :keys.shape[1]] = keys
    return result


def cpu_uniform(state, k):
    """NumPy Philox4x32-10 reference; no reuse of Triton sampling code."""
    state = state.cpu().numpy().view(np.uint64)
    mask = np.uint64(0xffffffff)
    offsets = state[:, 1, None] + np.arange(k, dtype=np.uint64)[None, :]
    c0, c1 = offsets & mask, offsets >> np.uint64(32)
    c2, c3 = np.zeros_like(c0), np.zeros_like(c0)
    key0, key1 = state[:, 0, None] & mask, state[:, 0, None] >> np.uint64(32)
    for _ in range(10):
        p0 = c0 * np.uint64(0xD2511F53)
        p2 = c2 * np.uint64(0xCD9E8D57)
        c0, c1, c2, c3 = ((p2 >> np.uint64(32)) ^ c1 ^ key0, p2 & mask,
                          (p0 >> np.uint64(32)) ^ c3 ^ key1, p0 & mask)
        key0 = (key0 + np.uint64(0x9E3779B9)) & mask
        key1 = (key1 + np.uint64(0xBB67AE85)) & mask
    return ((c0 >> np.uint64(9)).astype(np.float32) + np.float32(.5)) * np.float32(2. ** -23)


def cpu_sample(z, k, temperature, state):
    values, ids = reference(z, k)
    if temperature == 0 or k == 1:
        return ids[:, 0]
    values = values.float().cpu().numpy().astype(np.float64)
    uniform = cpu_uniform(state, k).astype(np.float64)
    scores = (values - values.max(axis=1, keepdims=True)) / np.float32(temperature)
    scores -= np.log(-np.log(uniform))
    # Exact floating ties choose the lower original token ID.
    indices = ids.cpu().numpy()
    selected = np.where(scores == scores.max(axis=1, keepdims=True), indices, 2**63 - 1).min(axis=1)
    return torch.from_numpy(selected).to(z.device)


@pytest.mark.parametrize("k", range(1, 17))
@pytest.mark.parametrize("temperature", [0., .2, 1., 4.])
def test_merge_sample_matches_independent_reference(k, temperature):
    torch.manual_seed(29)
    # Multiple merge levels, including a final vocabulary tail.
    z = (torch.randn(37, 157, device="cuda") * 2).bfloat16()
    keys = pack_local(z, k)
    plan = MergePlan(len(z), keys.shape[1], k, z.device)
    plan.values.fill_(123); plan.indices.fill_(-7)
    state = create_sampling_state(len(z), z.device, seed=2**45 + 17)
    state[:, 1] = torch.arange(len(z), device=z.device) * 2**35 + 9
    for _ in range(2):
        before = state.clone()
        expected = cpu_sample(z, k, temperature, before)
        actual = plan.sample(keys, temperature=temperature, rng_state=state)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(state[:, 0], before[:, 0], atol=0, rtol=0)
        advance = 16 if temperature > 0 and k > 1 else 0
        torch.testing.assert_close(state[:, 1], before[:, 1] + advance, atol=0, rtol=0)
    # Sampling did not materialize the global topK list in either output buffer.
    assert (plan.values == 123).all() and (plan.indices == -7).all()


@pytest.mark.parametrize("temperature", [.25, 1., 4.])
def test_empirical_topk_temperature_distribution(temperature):
    n = 50000
    # Global top 5 all live in one partition, with repeated cutoff scores in
    # another. This detects sampling local candidates before global selection.
    row = torch.tensor([2., 1., .5, 0., -.5, -1., -2., -3.,
                        -.5, -1., -2., -3., -4., -5., -6., -7.], device="cuda", dtype=torch.bfloat16)
    z = row[None, :].expand(n, -1).contiguous()
    keys = pack_local(z, 5, block=8)
    plan = MergePlan(n, 2, 5, z.device)
    state = create_sampling_state(n, z.device, seed=739)
    sampled = plan.sample(keys, temperature=temperature, rng_state=state)
    observed = torch.bincount(sampled, minlength=len(row)).double() / n
    expected = torch.zeros_like(observed)
    expected[:5] = (row[:5].double() / temperature).softmax(0)
    tolerance = 6 * torch.sqrt(expected * (1 - expected) / n) + 2 / n
    assert ((observed - expected).abs() <= tolerance).all()
    assert torch.count_nonzero(observed[5:]).item() == 0


def test_uniform_ties_and_state_reordering():
    n = 32768
    z = torch.zeros((n, 33), device="cuda", dtype=torch.bfloat16)
    keys = pack_local(z, 16)
    plan = MergePlan(n, keys.shape[1], 16, z.device)
    state = create_sampling_state(n, z.device, seed=1234)
    initial = state.clone()
    saved = plan.sample(keys, rng_state=state)
    copies = saved.clone()
    counts = torch.bincount(saved, minlength=33)
    assert torch.count_nonzero(counts[16:]).item() == 0
    expected = n / 16
    assert ((counts[:16] - expected).abs() < 6 * (n * (1/16) * (15/16)) ** .5).all()
    order = torch.arange(n - 1, -1, -1, device="cuda")
    moved = initial[order].contiguous()
    replay = plan.sample(keys[order].contiguous(), rng_state=moved)
    torch.testing.assert_close(replay, saved[order], atol=0, rtol=0)
    torch.testing.assert_close(moved, state[order], atol=0, rtol=0)
    # A further call advances state without modifying previous output storage.
    again = plan.sample(keys, rng_state=state)
    assert (again != saved).any()
    torch.testing.assert_close(saved, copies, atol=0, rtol=0)


@pytest.mark.parametrize("temperature", [0., 1e-30, 1e30])
def test_extreme_temperature(temperature):
    z = torch.tensor([[100., 0., -100., -200.]], device="cuda", dtype=torch.bfloat16).expand(1024, -1).contiguous()
    keys = pack_local(z, 4)
    state = create_sampling_state(len(z), z.device, seed=19)
    plan = MergePlan(len(z), 1, 4, z.device)
    output = plan.sample(keys, temperature=temperature, rng_state=state)
    if temperature < 1:
        assert (output == 0).all()
    else:
        expected = cpu_sample(z, 4, temperature, create_sampling_state(len(z), z.device, seed=19))
        torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.parametrize("row,allowed", [([float('inf'), 0., float('inf'), -1.], [0, 2]),
                                           ([float('-inf')] * 4, [0, 1, 2, 3])])
def test_infinite_logits_convention(row, allowed):
    n = 4096
    z = torch.tensor([row], device="cuda", dtype=torch.bfloat16).expand(n, -1).contiguous()
    plan = MergePlan(n, 1, 4, z.device)
    out = plan.sample(pack_local(z, 4), rng_state=create_sampling_state(n, z.device, seed=5))
    counts = torch.bincount(out, minlength=4)
    for i in range(4):
        if i in allowed:
            assert abs(counts[i].item() - n / len(allowed)) < 6 * (n / len(allowed)) ** .5
        else:
            assert counts[i].item() == 0


def test_finite_bf16_extremes_with_large_temperature():
    maximum = torch.finfo(torch.bfloat16).max
    z = torch.tensor([[maximum, 0., -maximum, -maximum]], device="cuda", dtype=torch.bfloat16)
    z = z.expand(8192, -1).contiguous()
    state = create_sampling_state(len(z), z.device, seed=509)
    expected = cpu_sample(z, 4, 3e38, state)
    plan = MergePlan(len(z), 1, 4, z.device)
    actual = plan.sample(pack_local(z, 4), temperature=3e38, rng_state=state)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert (actual >= 2).any()


@pytest.mark.parametrize("config", [0, 8, 70, 71])
@pytest.mark.parametrize("k", [1, 7, 16])
def test_fused_sampler_matches_plain_gemm(config, k):
    x, w = inputs(17, 128, 513)
    x.requires_grad_(); w.requires_grad_()
    xq, args = quantized(x.detach(), w.detach())
    z = plain_logits(args, config)
    state = create_sampling_state(len(x), x.device, seed=591)
    expected = cpu_sample(z, k, .7, state)
    initial = state.clone()
    for permute in (False, True):
        state.copy_(initial)
        sampler = MXFP8LMHeadSampler(top_k=k, temperature=.7, config=config, permute_n=permute)
        result = sampler(x, w, quantized_weight=(args[1], args[3]), rng_state=state)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        assert result.shape == (len(x),) and result.dtype == torch.int64
        assert not result.requires_grad
        assert all(xq.x is None and not p.refs for xq, p in sampler._states.values())


@pytest.mark.parametrize("k", [1, 7, 16])
def test_graph_replay_advances_gpu_rng(k):
    x, w = inputs(17, 128, 513)
    wq, ws = quantize_weight(w)
    state = create_sampling_state(len(x), x.device, seed=281)
    sampler = MXFP8LMHeadSampler(top_k=k, temperature=1.3)
    def step(rng=state):
        return sampler(x, w, quantized_weight=(wq, ws), rng_state=rng)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for seed in (11, 12, 13):
        nx, nw = inputs(17, 128, 513, seed=seed)
        x.copy_(nx); w.copy_(nw)
        nq, ns = quantize_weight(w)
        wq.copy_(nq); ws.copy_(ns)
        before = state.clone()
        reference_state = state.clone()
        expected = step(reference_state)
        graph.replay()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)
        torch.testing.assert_close(state, reference_state, atol=0, rtol=0)
        torch.testing.assert_close(state[:, 1], before[:, 1] + (16 if k > 1 else 0), atol=0, rtol=0)


def test_automatic_state_reset_and_workspace_lifetime():
    x, w = inputs(33, 128, 257)
    sampler = MXFP8LMHeadSampler(top_k=16, temperature=.8, seed=877)
    first = sampler(x, w)
    old = first.clone()
    state = next(iter(sampler._rng_states.values()))
    snapshot = state.clone()
    pointer = state.data_ptr()
    sampler.clear_workspace()
    second = sampler(x, w)
    assert (second != first).any()
    torch.testing.assert_close(first, old, atol=0, rtol=0)
    torch.testing.assert_close(state[:, 1], snapshot[:, 1] + 16, atol=0, rtol=0)
    sampler.reset_rng()
    assert state.data_ptr() == pointer and (state[:, 1] == 0).all()
    torch.testing.assert_close(sampler(x, w), first, atol=0, rtol=0)
    sampler.reset_rng(seed=123)
    torch.testing.assert_close(sampler(x, w), MXFP8LMHeadSampler(top_k=16, temperature=.8, seed=123)(x, w), atol=0, rtol=0)


def test_reset_automatic_state_used_by_graph():
    x, w = inputs(17, 128, 257)
    sampler = MXFP8LMHeadSampler(top_k=16, seed=291)
    first = sampler(x, w)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        sampler(x, w)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = sampler(x, w)
    sampler.reset_rng()
    graph.replay()
    torch.testing.assert_close(captured, first, atol=0, rtol=0)
    graph.replay()
    assert (captured != first).any()


def test_counter_wrap_and_greedy_without_state():
    z = torch.zeros(7, 16, device="cuda", dtype=torch.bfloat16)
    keys = pack_local(z, 16)
    state = create_sampling_state(7, z.device, seed=2**63 - 1)
    state[:, 1] = -16  # uint64 2**64-16; reserves the final 16 offsets.
    plan = MergePlan(7, 1, 16, z.device)
    expected = cpu_sample(z, 16, 1., state)
    actual = plan.sample(keys, rng_state=state)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert (state[:, 1] == 0).all()
    assert (plan.sample(keys, temperature=0) == 0).all()


def test_sampling_validation_and_aliases():
    for t in (-1, float('nan'), float('inf'), True, '1', 1e100, 1e-100):
        with pytest.raises(ValueError, match="temperature"):
            MXFP8LMHeadSampler(temperature=t)
    for seed in (-1, 2**63, True, 1.5):
        with pytest.raises(ValueError, match="seed"):
            create_sampling_state(1, 'cuda', seed=seed)
    z = torch.zeros(3, 2, device="cuda", dtype=torch.bfloat16)
    keys = pack_local(z, 2)
    plan = MergePlan(3, 1, 2, z.device)
    with pytest.raises(ValueError, match="requires rng_state"):
        plan.sample(keys)
    state = create_sampling_state(3, z.device)
    for invalid in (state.float(), state[:1], state.cpu(), state.T.contiguous()):
        with pytest.raises(ValueError, match="rng_state"):
            plan.sample(keys, rng_state=invalid)
    with pytest.raises(ValueError, match="overlap"):
        plan.sample(keys, rng_state=keys.view(3, 2))
    with pytest.raises(ValueError, match="overlap"):
        plan.sample(keys, rng_state=state, out=state.view(-1)[:3])
    assert (state[:, 1] == 0).all()
