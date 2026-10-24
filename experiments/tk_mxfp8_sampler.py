"""MXFP8 LM-head topK + temperature sampling, without full logits or topP.

Final Triton merge selects the global topK BEFORE adding Gumbel noise. For T>0
it returns argmax((logit-max)/T - log(-log(U))); no softmax tensor is needed.
T=0 or K=1 is greedy and does not consume RNG state. Inference only; no autograd.

Example::

    sampler = MXFP8LMHeadSampler(top_k=16, temperature=.8, seed=123)
    token_ids = sampler(x_bf16, weight, quantized_weight=(wq, ws))

Default RNG streams are per (device, batch size), with row positions identifying
sequences. For dynamic batches, supply create_sampling_state(M, device, seed=...)
and move its rows with sequences. Each stochastic call advances offsets by 16.
The RNG is independent of PyTorch's global generator / torch.manual_seed.

Warm shapes before CUDA Graph capture. Warmup consumes RNG state; each graph
replay advances the same GPU state. reset_rng() resets existing buffers in place,
including those captured by graphs. clear_workspace() does not reset RNG.
Do not use the same sampler/state concurrently on multiple streams.
"""

import torch

from experiments.mxfp8_topk import create_sampling_state, validate_seed, validate_temperature
from experiments.tk_mxfp8_topk import MXFP8TopKLinear


class MXFP8LMHeadSampler(MXFP8TopKLinear):
    def __init__(self, *, top_k=16, temperature=1., seed=0, config=8,
                 raster_group=13, sms=0, permute_n=True):
        super().__init__(top_k=top_k, config=config, raster_group=raster_group,
                         sms=sms, permute_n=permute_n)
        self.temperature = validate_temperature(temperature)
        validate_seed(seed)
        self.seed = seed
        self._rng_states = {}

    def forward(self, x, weight, *, quantized_weight=None, rng_state=None):
        temperature = validate_temperature(self.temperature)

        def finish(plan):
            state = rng_state
            if state is None and temperature > 0 and self.top_k > 1:
                key = (plan.device, plan.m)
                if key not in self._rng_states:
                    self._rng_states[key] = create_sampling_state(plan.m, plan.device, seed=self.seed)
                state = self._rng_states[key]
            return plan.merge.sample(plan.local(), temperature=temperature, rng_state=state)

        return self._run(x, weight, quantized_weight=quantized_weight, finish=finish)

    @torch.no_grad()
    def reset_rng(self, seed=None):
        """Reset automatic streams, preserving pointers used by captured graphs.
        Caller-owned rng_state tensors are unaffected. Call on the same stream
        as subsequent sampling, outside graph capture.
        """
        if seed is not None:
            validate_seed(seed)
            self.seed = seed
        for (device, m), state in self._rng_states.items():
            state.copy_(create_sampling_state(m, device, seed=self.seed))
