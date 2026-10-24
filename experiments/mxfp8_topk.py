"""Bounded topK merges and final-merge Gumbel-max sampling.

Sampling uses BF16 logits and FP32 arithmetic. Each row owns a Philox seed and
64-bit draw offset in an int64 [M,2] CUDA tensor. A stochastic call reserves 16
offsets, independent of K and merge layout. Greedy calls do not advance state.
"""

import math
import struct

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_key(key):
    order = (key >> 32).to(tl.uint32)
    bits = tl.where((order & 0x8000) != 0, order ^ 0x8000, (~order) & 0xffff)
    value = (bits << 16).to(tl.float32, bitcast=True)
    index = (0xffffffff - (key & 0xffffffff)).to(tl.int64)
    return value, index


@triton.jit
def _uniform_open(seed, offsets):
    # Midpoints of 2**23 equal bins: exactly representable FP32 in (0,1).
    # Neither logarithm in -log(-log(u)) can encounter an endpoint.
    bits = tl.randint(seed, offsets).to(tl.uint32)
    return ((bits >> 9).to(tl.float32) + 0.5) * (2. ** -23)


@triton.jit
def _sample_topk(key, RNG, Samples, row, K: tl.constexpr, WANT: tl.constexpr,
                 TEMPERATURE: tl.constexpr):
    # Only the globally selected top K gets noise. Perturbing local candidates
    # before global selection would sample a different distribution.
    rank = tl.arange(0, K)
    key = tl.gather(key, rank, 0)
    value, index = _decode_key(key)
    if TEMPERATURE == 0. or WANT == 1:
        selected = tl.sum(tl.where(rank == 0, index, 0), 0)
    else:
        seed = tl.load(RNG + row * 2).to(tl.uint64)
        offset = tl.load(RNG + row * 2 + 1).to(tl.uint64)
        offsets = tl.add(offset, rank.to(tl.uint64), sanitize_overflow=False)
        u = _uniform_open(seed, offsets)
        gumbel = -tl.log(-tl.log(u))
        maximum = tl.max(tl.where(rank < WANT, value, float('-inf')), 0)
        # Finite logits use the usual stabilized difference. For +inf choose
        # uniformly among +inf candidates; for all -inf choose uniformly in K.
        delta = tl.where(maximum == float('inf'),
                         tl.where(value == float('inf'), 0., float('-inf')),
                         tl.where(maximum == float('-inf'), 0., value - maximum))
        scaled = tl.div_rn(delta, TEMPERATURE)
        if TEMPERATURE > 1.:
            # Opposite finite BF16 extremes can overflow value-maximum, even
            # when division by a large T would yield a small finite difference.
            half_delta = value * .5 - maximum * .5
            scaled = tl.where(delta == float('-inf'),
                              tl.div_rn(half_delta, TEMPERATURE * .5), scaled)
        score = scaled + gumbel
        score = tl.where(rank < WANT, score, float('-inf'))
        best = tl.max(score, 0)
        selected = tl.min(tl.where((rank < WANT) & (score == best), index, 0x7fffffffffffffff), 0)
        tl.store(RNG + row * 2 + 1, tl.add(offset, 16, sanitize_overflow=False))
    tl.store(Samples + row, selected)


@triton.jit
def _merge(Keys, Out, Values, Indices, P: tl.constexpr, Q: tl.constexpr,
           K: tl.constexpr, WANT: tl.constexpr, FAN: tl.constexpr,
           FINAL: tl.constexpr, RNG=None, SAMPLE: tl.constexpr = False,
           TEMPERATURE: tl.constexpr = 1.):
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    i = tl.arange(0, FAN * K)
    src = group * FAN * K + i
    key = tl.load(Keys + row * P * K + src, src < P * K, 0).to(tl.uint64)
    # Bitonic selection; every program sees at most FAN*K <= 128 keys.
    key = tl.sort(key, descending=True)
    if FINAL:
        if SAMPLE:
            _sample_topk(key, RNG, Indices, row, K, WANT, TEMPERATURE)
        else:
            value, index = _decode_key(key)
            tl.store(Values + row * WANT + i, value.to(tl.bfloat16), i < WANT)
            tl.store(Indices + row * WANT + i, index, i < WANT)
    else:
        tl.store(Out + (row * Q + group) * K + i, key, i < K)


def validate_k(k, v=None):
    if type(k) is not int or not 1 <= k <= 16 or (v is not None and k > v):
        raise ValueError("top_k must be an integer in [1, min(16, V)]")


def validate_temperature(temperature):
    if type(temperature) not in (int, float):
        raise ValueError("temperature must be a nonnegative finite FP32 constant")
    try:
        value = struct.unpack("f", struct.pack("f", temperature))[0]
    except (OverflowError, struct.error):
        value = float('inf')
    if not math.isfinite(value) or value < 0 or (temperature != 0 and value == 0):
        raise ValueError("temperature must be a nonnegative finite FP32 constant")
    return value


def validate_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")


def create_sampling_state(m, device, *, seed=0):
    """Create CUDA int64 [M,2]: per-row seed (seed+row modulo 2**64), offset=0.

    Move this tensor's rows with sequences when batching/reordering. The sampler
    updates offsets in place. Clone it to fork a stream or save/restore it for
    replay. Caller-owned states must have one independent row per sequence.
    """
    validate_seed(seed)
    if type(m) is not int or m <= 0:
        raise ValueError("Positive integer row count required")
    if torch.device(device).type != "cuda":
        raise ValueError("sampling state requires a CUDA device")
    seeds = torch.arange(m, device=device, dtype=torch.int64) + seed
    return torch.stack((seeds, torch.zeros_like(seeds)), dim=1)


class MergePlan:
    """No [M,V] workspace; sorted output uses ascending token IDs for ties."""
    def __init__(self, m, parts, k, device):
        validate_k(k)
        if type(m) is not int or type(parts) is not int or min(m, parts) <= 0:
            raise ValueError("Positive integer row and partition counts required")
        self.m, self.parts, self.k, self.device = m, parts, k, torch.device(device)
        self.bucket = triton.next_power_of_2(k)
        self.levels = []
        p = parts
        while p > 8:
            q = triton.cdiv(p, 8)
            self.levels.append(torch.empty((m, q, self.bucket), device=device, dtype=torch.int64))
            p = q
        self.values = torch.empty((m, k), device=device, dtype=torch.bfloat16)
        self.indices = torch.empty((m, k), device=device, dtype=torch.int64)

    def __call__(self, keys, *, out=None):
        self._validate_keys(keys)
        values, indices = (self.values, self.indices) if out is None else out
        for tensor, dtype in ((values, torch.bfloat16), (indices, torch.int64)):
            if (tensor.shape != (self.m, self.k) or tensor.dtype != dtype
                    or tensor.device != keys.device or not tensor.is_contiguous()):
                raise ValueError("Invalid topK outputs")
        self._validate_overlap(keys, (values, indices))
        with torch.cuda.device(keys.device):
            current = self._reduce(keys)
            p = current.shape[1]
            _merge[(self.m, 1)](current, None, values, indices, p, 1, self.bucket,
                                self.k, triton.next_power_of_2(p), True, num_warps=4)
        return values, indices

    def _validate_keys(self, keys):
        if (keys.shape != (self.m, self.parts, self.bucket) or keys.dtype != torch.int64
                or keys.device != self.values.device or not keys.is_contiguous()):
            raise ValueError("Invalid packed local topK keys")

    def _validate_overlap(self, keys, outputs):
        for i, tensor in enumerate(outputs):
            lo, hi = tensor.data_ptr(), tensor.data_ptr() + tensor.nbytes
            if any(lo < t.data_ptr() + t.nbytes and t.data_ptr() < hi
                   for t in (keys, *self.levels, *outputs[:i])):
                raise ValueError("topK outputs/state must not overlap workspace or each other")

    def _reduce(self, keys):
        current = keys
        for buffer in self.levels:
            p, q = current.shape[1], buffer.shape[1]
            _merge[(self.m, q)](current, buffer, None, None, p, q, self.bucket,
                                self.k, 8, False, num_warps=4)
            current = buffer
        return current

    def sample(self, keys, *, temperature=1., rng_state=None, out=None):
        """Merge to global topK and draw one int64 token per row, in the final
        merge kernel. Positive temperature and K>1 require a CUDA RNG state;
        temperature=0 or K=1 is greedy and leaves the state unchanged.
        Default output storage is independent across calls. No topK list is
        materialized by this path. Concurrent use of scratch/state is invalid.
        """
        self._validate_keys(keys)
        temperature = validate_temperature(temperature)
        if rng_state is None:
            if temperature > 0 and self.k > 1:
                raise ValueError("Stochastic sampling requires rng_state")
        elif (rng_state.shape != (self.m, 2) or rng_state.dtype != torch.int64
              or rng_state.device != keys.device or not rng_state.is_contiguous()):
            raise ValueError("Expected contiguous CUDA int64 rng_state[M,2] on keys device")
        if out is None:
            out = torch.empty(self.m, device=keys.device, dtype=torch.int64)
        if (out.shape != (self.m,) or out.dtype != torch.int64
                or out.device != keys.device or not out.is_contiguous()):
            raise ValueError("Expected contiguous int64 sampling output[M] on keys device")
        self._validate_overlap(keys, (out,) if rng_state is None else (out, rng_state))
        with torch.cuda.device(keys.device):
            current = self._reduce(keys)
            p = current.shape[1]
            _merge[(self.m, 1)](current, None, None, out, p, 1, self.bucket,
                                self.k, triton.next_power_of_2(p), True,
                                rng_state, True, temperature, num_warps=4,
                                enable_fp_fusion=False)
        return out
