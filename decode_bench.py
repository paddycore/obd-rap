"""
Decode-step latency of one LLaMA decoder layer under KV-cache layouts that store
the same number of values per token.

pruned   K, V: [B, H, T, w]. w = D is the dense model; w = 2s is OBD-RAP / RAP with
         s retained RoPE pairs per head. Only the new token is rotated, and
         attention runs on the narrow heads with the original 1/sqrt(D) scale.
lowrank  K, V: [B, G, T, r/G] latents, as in the OBD-LLM K/V baseline (Palu-style).
         RoPE needs the full key, so every step rebuilds and rotates the keys of
         all T positions. V's up-projection is applied after the softmax, so V costs
         no rebuild. G = 1 is one latent shared by all heads (the variant whose
         quality obd_rap_pipeline.py measures); G = H is one latent per head.

Weights and caches are random: timing does not depend on their values.
"""
import math
import time

import torch
import torch.nn.functional as F

LLAMA2_7B = dict(d=4096, H=32, D=128, ffn=11008, n_layers=32)


def rope_tables(n_pos, D, base=10000.0, device=None, dtype=torch.float32):
    """HF LLaMA cos/sin, [n_pos, D] (each frequency repeated in both halves)."""
    inv = 1.0 / base ** (torch.arange(0, D, 2, device=device, dtype=torch.float32) / D)
    ang = torch.arange(n_pos, device=device, dtype=torch.float32)[:, None] * inv
    ang = torch.cat([ang, ang], dim=-1)
    return ang.cos().to(dtype), ang.sin().to(dtype)


def rotate(x, cos, sin):
    """HF rotate_half RoPE on the last dim; cos/sin broadcast against x."""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


def _rand(*shape, device, dtype, scale=1.0):
    return torch.randn(*shape, device=device, dtype=dtype) * scale


class PrunedAttention:
    """Retained dims per head: lead[h] = the s kept pair leaders j (< D/2); their
    partners j + D/2 sit in the second half of each narrow head."""

    def __init__(self, d, H, D, s, B, T, device="cpu", dtype=torch.float32, lead=None):
        self.H, self.D, self.s, self.w = H, D, s, 2 * s
        if lead is None:
            lead = torch.stack([torch.randperm(D // 2)[:s].sort().values for _ in range(H)])
        self.lead = lead.to(device)
        sd = d ** -0.5
        self.wq, self.wk, self.wv = (_rand(H * self.w, d, device=device, dtype=dtype, scale=sd) for _ in range(3))
        self.wo = _rand(d, H * self.w, device=device, dtype=dtype, scale=sd)
        self.k = torch.zeros(B, H, T, self.w, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        cos, sin = rope_tables(T, D, device=device, dtype=dtype)
        self.cos, self.sin = cos[:, :D // 2][:, self.lead], sin[:, :D // 2][:, self.lead]   # [T, H, s]
        self.cos, self.sin = torch.cat([self.cos] * 2, -1), torch.cat([self.sin] * 2, -1)  # [T, H, 2s]

    def _proj(self, x, pos):
        """x: [B, n, d] tokens at positions pos (a slice) -> rotated q, k and v, each [B, H, n, w]."""
        B, n, _ = x.shape
        q, k, v = ((x @ w.T).view(B, n, self.H, self.w).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        c, sn = self.cos[pos].transpose(0, 1), self.sin[pos].transpose(0, 1)                # [H, n, 2s]
        return rotate(q, c, sn), rotate(k, c, sn), v

    def prefill(self, x):
        _, k, v = self._proj(x, slice(0, x.shape[1]))
        self.k[:, :, :x.shape[1]], self.v[:, :, :x.shape[1]] = k, v

    def step(self, x):
        """x: [B, d], the token at the last cache position T-1. Returns [B, d]."""
        T = self.k.shape[2]
        q, k, v = self._proj(x[:, None], slice(T - 1, T))
        self.k[:, :, -1:], self.v[:, :, -1:] = k, v
        o = F.scaled_dot_product_attention(q, self.k, self.v, scale=self.D ** -0.5)
        return o.reshape(x.shape[0], -1) @ self.wo.T

    def full_weights(self):
        """Dense [H*D, d] equivalents (zero rows for pruned dims) for the reference check."""
        rows = torch.cat([self.lead, self.lead + self.D // 2], 1) + torch.arange(self.H, device=self.lead.device)[:, None] * self.D
        rows = rows.reshape(-1)
        out = []
        for w in (self.wq, self.wk, self.wv):
            full = w.new_zeros(self.H * self.D, w.shape[1])
            full[rows] = w
            out.append(full)
        wo = self.wo.new_zeros(self.wo.shape[0], self.H * self.D)
        wo[:, rows] = self.wo
        return (*out, wo)


class LowRankAttention:
    def __init__(self, d, H, D, r, G, B, T, device="cpu", dtype=torch.float32):
        if H % G or r % G:
            raise ValueError("H and r must be divisible by G")
        self.H, self.D, self.G, self.hg, self.rg = H, D, G, H // G, r // G
        sd = d ** -0.5
        self.wq = _rand(H * D, d, device=device, dtype=dtype, scale=sd)
        self.wo = _rand(d, H * D, device=device, dtype=dtype, scale=sd)
        self.ak, self.av = (_rand(r, d, device=device, dtype=dtype, scale=sd) for _ in range(2))  # rows: group-major
        self.bk = _rand(G, self.rg, self.hg * D, device=device, dtype=dtype, scale=self.rg ** -0.5)
        self.bv = _rand(G, self.hg, D, self.rg, device=device, dtype=dtype, scale=self.rg ** -0.5)
        self.ck = torch.zeros(B, G, T, self.rg, device=device, dtype=dtype)
        self.cv = torch.zeros_like(self.ck)
        self.cos, self.sin = rope_tables(T, D, device=device, dtype=dtype)

    def _latent(self, x, a):
        B, n, _ = x.shape
        return (x @ a.T).view(B, n, self.G, self.rg).transpose(1, 2)                      # [B, G, n, rg]

    def prefill(self, x):
        n = x.shape[1]
        self.ck[:, :, :n], self.cv[:, :, :n] = self._latent(x, self.ak), self._latent(x, self.av)

    def step(self, x):
        B, T = x.shape[0], self.ck.shape[2]
        q = rotate((x @ self.wq.T).view(B, self.G, self.hg, self.D), self.cos[-1], self.sin[-1])
        self.ck[:, :, -1:], self.cv[:, :, -1:] = self._latent(x[:, None], self.ak), self._latent(x[:, None], self.av)
        k = (self.ck @ self.bk).view(B, self.G, T, self.hg, self.D)                        # rebuild all keys
        k = rotate(k, self.cos[:, None], self.sin[:, None])
        scores = torch.einsum("bgthd,bghd->bght", k, q) * self.D ** -0.5
        p = torch.softmax(scores, dim=-1, dtype=torch.promote_types(x.dtype, torch.float32)).to(x.dtype)
        y = p @ self.cv                                                                    # [B, G, hg, rg]
        o = torch.einsum("bghr,ghdr->bghd", y, self.bv)
        return o.reshape(B, -1) @ self.wo.T

    def full_weights(self):
        wk = torch.cat([self.bk[g].T @ self.ak[g * self.rg:(g + 1) * self.rg] for g in range(self.G)])
        wv = torch.cat([self.bv[g].reshape(-1, self.rg) @ self.av[g * self.rg:(g + 1) * self.rg] for g in range(self.G)])
        return self.wq, wk, wv, self.wo


class MLP:
    def __init__(self, d, ffn, device="cpu", dtype=torch.float32):
        self.wg, self.wu = (_rand(ffn, d, device=device, dtype=dtype, scale=d ** -0.5) for _ in range(2))
        self.wd = _rand(d, ffn, device=device, dtype=dtype, scale=ffn ** -0.5)

    def step(self, x):
        return (F.silu(x @ self.wg.T) * (x @ self.wu.T)) @ self.wd.T


def reference_last_token(X, wq, wk, wv, wo, H, D):
    """Plain causal attention output of the last token of X [B, T, d] with full RoPE."""
    B, T, _ = X.shape
    cos, sin = rope_tables(T, D, device=X.device, dtype=X.dtype)
    q, k, v = ((X @ w.T).view(B, T, H, D).transpose(1, 2) for w in (wq, wk, wv))
    q, k = rotate(q, cos, sin), rotate(k, cos, sin)
    p = torch.softmax(q[:, :, -1:] @ k.transpose(-1, -2) / math.sqrt(D), dim=-1)
    return (p @ v).reshape(B, -1) @ wo.T


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_ms(fn, x, iters=50, warmup=10, graph=True):
    """Mean milliseconds per call. On CUDA the call is captured in a CUDA graph so
    kernel-launch overhead from Python does not mask the work being compared."""
    if x.device.type != "cuda":
        for _ in range(warmup):
            fn(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(x)
        return (time.perf_counter() - t0) * 1e3 / iters
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn(x)
    torch.cuda.current_stream().wait_stream(side)
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn(x)
        run = g.replay
    else:
        run = lambda: fn(x)
    for _ in range(warmup):
        run()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _measure(make, x, **kw):
    m = None
    try:
        m = make()
        with torch.no_grad():
            return time_ms(m.step, x, **kw)
    except torch.cuda.OutOfMemoryError:
        return float("nan")
    finally:
        m = None
        if x.is_cuda:
            torch.cuda.empty_cache()


def run_grid(batches=(1, 8, 32), contexts=(1024, 4096, 16384), ratios=(0.1, 0.2, 0.3, 0.4),
             groups=(1, 8, 32), dims=LLAMA2_7B, device="cuda", dtype=torch.float16,
             iters=50, warmup=10, pair_align=4, log=print):
    """Rows of {batch, context, ratio, method, attn_ms, layer_ms, token_ms, kv_kb_per_token}.
    layer_ms adds the MLP of one layer; token_ms = n_layers * layer_ms (no embedding/head).
    s is rounded to a multiple of pair_align: fused attention kernels (Flash) need the head
    width 2s to be a multiple of 8, and other widths fall back to a much slower kernel."""
    d, H, D, ffn, L = (dims[k] for k in ("d", "H", "D", "ffn", "n_layers"))
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    rows = []
    for B in batches:
        x = torch.randn(B, d, device=device, dtype=dtype)
        mlp_ms = _measure(lambda: MLP(d, ffn, device, dtype), x, iters=iters, warmup=warmup)
        for T in contexts:
            def add(ratio, method, attn_ms, kv_values):
                rows.append(dict(batch=B, context=T, ratio=ratio, method=method, attn_ms=attn_ms,
                                 layer_ms=attn_ms + mlp_ms, token_ms=L * (attn_ms + mlp_ms),
                                 kv_kb_per_token=L * kv_values * bytes_per / 1024))
                log(f"B={B:<3} T={T:<6} {method:<16} rho={ratio:.1f}  attn {attn_ms:8.3f} ms")
            dense = _measure(lambda: PrunedAttention(d, H, D, D // 2, B, T, device, dtype), x, iters=iters, warmup=warmup)
            add(0.0, "dense", dense, 2 * H * D)
            for rho in ratios:
                s = pair_align * round((1 - rho) * D / 2 / pair_align)
                add(rho, "obd_rap", _measure(lambda: PrunedAttention(d, H, D, s, B, T, device, dtype), x,
                                             iters=iters, warmup=warmup), 2 * H * 2 * s)
                for G in groups:
                    r = H * 2 * s                      # same cache size as obd_rap
                    add(rho, f"lowrank_G{G}", _measure(lambda: LowRankAttention(d, H, D, r, G, B, T, device, dtype), x,
                                                       iters=iters, warmup=warmup), 2 * r)
    return rows
